import argparse
import json
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import torch
import wandb

from src.utils.experiment import (
    count_model_parameters,
    get_device,
    get_full_path,
    get_summary_writer,
)
from src.utils.io import dump_json
from src.utils.optim import get_optimizer, get_scheduler
from src.utils.scheduler import get_clipping_schedule


class Trainer(metaclass=ABCMeta):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: torch.nn.Module,
    ) -> None:
        self.args = args
        self.model = model
        self.optimizer = get_optimizer(
            args, chain(model.parameters(), loss.parameters())
        )
        self.scheduler = (
            get_scheduler(args, self.optimizer) if self.args.use_lr_schedule else None
        )
        self.loss = loss
        # set up dataloaders
        self.train_dataloader = dataloaders[-1]
        self.test_dataloader = dataloaders[-2] if len(dataloaders) > 1 else None
        self.val_dataloader = dataloaders[-3] if len(dataloaders) > 2 else None
        # number of samples in each dataloader
        self.number_of_samples = {
            "train": len(self.train_dataloader.dataset),
            "test-last-model": len(self.test_dataloader.dataset)
            if len(dataloaders) > 1
            else None,
            "test-best-validation": len(self.test_dataloader.dataset)
            if len(dataloaders) > 1
            else None,
            "val": len(self.val_dataloader.dataset) if len(dataloaders) > 2 else None,
        }
        # record errors at end of training
        self.test_summary = dict()
        self.save_test_summary = args.save_test_summary or args.get_point_estimate
        self.extended_testing = self.save_test_summary
        # zero shot prediction
        self.zero_shot_prediction = (
            self.args.random_zero_shot_prediction_from_rows_and_cols
        )
        # clipping scheduler
        self.clipping_scheduler = (
            get_clipping_schedule(self.args) if self.args.clipping_schedule else None
        )
        # WandB
        if self.args.wandb:
            # watch gradients and parameters of the current model with WandB
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.args.wandb_log_frequency,
                log_graph=True,
            )
        # set up summary writer
        self.summary_writer = get_summary_writer(args)
        # write the hyper-parameters to TensorBoard
        self.summary_writer.add_text(
            # see also https://www.tensorflow.org/tensorboard/text_summaries
            "global",
            "".join(
                "\t" + line
                for line in json.dumps(self.args.__dict__, indent=4).splitlines(True)
            ),
        )
        # keep track of iterations for tensorboard
        self.total_number_of_train_iterations = 0
        self.total_number_of_test_iterations = 0
        self.total_number_of_val_iterations = 0
        self.total_number_of_epochs = 0
        # keep track of validation
        self.best_validation_score = float("inf")
        # count the trainable parameters of this model
        self.number_of_trainable_parameters = count_model_parameters(self.model)
        print(
            f"This model has a total of {self.number_of_trainable_parameters} "
            f"trainable parameters."
        )

    def train(self) -> None:
        for epoch in range(self.total_number_of_epochs, self.args.number_epochs):
            # train epoch
            loss_stats = self.train_epoch()

            # check whether to do validation
            if (
                epoch % self.args.validate_every_epochs == 0
                and self.val_dataloader is not None
                and not self.args.do_not_validate
            ):
                val_stats = self.validate()
                print("-" * 80)
                print(
                    f"epoch {epoch}, iteration {self.total_number_of_train_iterations}"
                )
                print("validating ...")
                for stat, stat_value in val_stats.items():
                    val = stat_value if "weight" in stat else stat_value.item()
                    print("\t" + str(stat) + (" " * (50 - len(str(stat)))) + str(val))
                # early stopping
                validation_loss = val_stats["loss"].item()
                if validation_loss < self.best_validation_score:
                    print(
                        f"received a better validation loss "
                        f"{validation_loss} vs {self.best_validation_score}."
                    )
                self.best_validation_score = validation_loss
                self.checkpoint(identifier="best_validation")
                print("-" * 80)
                print("*" * 80)

            # check whether to checkpoint the current model
            if (self.args.checkpoint_every_epochs > 0) and (
                epoch % self.args.checkpoint_every_epochs == 0
            ):
                self.checkpoint()

            # check whether to print to the console
            if epoch % self.args.print_loss_every_epochs == 0:
                print(
                    f"epoch {epoch}, iteration {self.total_number_of_train_iterations}"
                )
                for stat, stat_value in loss_stats.items():
                    print(
                        "\t"
                        + str(stat)
                        + (" " * (50 - len(str(stat))))
                        + str(stat_value)
                    )
                print("*" * 80)
            self.total_number_of_epochs += 1

        # save final model
        self.checkpoint()
        print("done training.")

        # log final model
        # from here on only testing and validation in eval mode
        self.model.eval()
        self.log()

        # test the current model
        if self.test_dataloader is not None:
            val_stats = self.test("test-last-model")
            print("-" * 80)
            print("testing on current model at last epoch ...")
            for stat, stat_value in val_stats.items():
                val = stat_value if "weight" in stat else stat_value.item()
                print("\t" + str(stat) + (" " * (50 - len(str(stat)))) + str(val))
            print("-" * 80)
            print("*" * 80)
            # test the best validation model
            if self.val_dataloader is not None and not self.args.do_not_validate:
                # make temporary copy of test summary (otherwise would be overwritten)
                test_summary = self.test_summary.copy()
                self.load_checkpoint(
                    get_full_path(
                        self.args, f"checkpoints/checkpoint_best_validation.pth"
                    )
                )
                # restore test summary
                self.test_summary = test_summary
                val_stats = self.test("test-best-validation")
                print("-" * 80)
                print("testing on best validation model ...")
                for stat, stat_value in val_stats.items():
                    val = stat_value if "weight" in stat else stat_value.item()
                    print("\t" + str(stat) + (" " * (50 - len(str(stat)))) + str(val))
                print("-" * 80)
                print("*" * 80)

        # save test summary
        if self.save_test_summary:
            dump_json(
                {str(key): value for key, value in self.test_summary.items()},
                get_full_path(self.args, f"test_summary.json"),
            )

        # close writer to get file events
        self.summary_writer.close()

    def train_epoch(self) -> Dict[str, torch.Tensor]:
        number_of_iterations = len(self.train_dataloader)
        epoch_loss_stats = dict()

        # set model in training mode
        self.model.train()

        for iteration, data in enumerate(self.train_dataloader):
            # preform forward pass and compute loss
            self.optimizer.zero_grad()
            loss_stats = self.forward_model_compute_loss(
                *self.unpack_data(data), split="train"
            )
            # update metrics for tensorboard and logging
            for loss_stat, value in loss_stats.items():
                val = value if "weight" in loss_stat else value.item()
                self.summary_writer.add_scalar(
                    f"train/{loss_stat}",
                    val,
                    self.total_number_of_train_iterations,
                )
                epoch_loss_stat = epoch_loss_stats.get(loss_stat)
                if epoch_loss_stat is None:
                    epoch_loss_stats[loss_stat] = 1 / number_of_iterations * val
                else:
                    epoch_loss_stats[loss_stat] += 1 / number_of_iterations * val
            # log learning rate
            self.summary_writer.add_scalar(
                "train/lr",
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else self.args.lr,
                self.total_number_of_train_iterations,
            )
            # back-prop
            loss_stats["loss"].backward()

            if self.args.wandb:
                # log loss to WandB to be able to watch the model
                wandb.log({"loss": loss_stats["loss"].item()})

            # clip gradient value
            if self.args.clip_grad_value:
                if self.args.clipping_schedule:
                    grad_val, clip = self.clipping_scheduler(
                        self.total_number_of_epochs
                    )
                    if clip:
                        self.model.clip_grad_value(max_grad=grad_val)
                        # log clipping scheduler
                        if self.args.clipping_schedule:
                            self.summary_writer.add_scalar(
                                "train/grad_val_clipping_schedule",
                                self.clipping_scheduler(self.total_number_of_epochs)[0],
                                self.total_number_of_train_iterations,
                            )
                else:
                    self.model.clip_grad_value()

            # optimize
            self.optimizer.step()
            self.total_number_of_train_iterations += 1

        # apply schedule
        if self.args.use_lr_schedule:
            self.scheduler.step()
        return epoch_loss_stats

    def _test(
        self, dataloader: torch.utils.data.dataloader.DataLoader, split: str
    ) -> Dict[str, torch.Tensor]:
        number_of_iterations = len(dataloader)
        epoch_loss_stats = dict()

        # set model in evaluation mode
        self.model.eval()

        for iteration, data in enumerate(dataloader):
            # forward pass without gradients
            with torch.no_grad():
                loss_stats = self.forward_model_compute_loss(
                    *self.unpack_data(data), split=split
                )
            if split.startswith("test"):
                if self.extended_testing:
                    with torch.no_grad():
                        self.summarize_test(*self.unpack_data(data), split=split)
            # update metrics
            for loss_stat, value in loss_stats.items():
                val = value if "weight" in loss_stat else value.item()
                self.summary_writer.add_scalar(
                    f"{split}/{loss_stat}",
                    val,
                    self.total_number_of_val_iterations
                    if split == "val"
                    else self.total_number_of_test_iterations,
                )
                epoch_loss_stat = epoch_loss_stats.get(loss_stat)
                if epoch_loss_stat is None:
                    epoch_loss_stats[loss_stat] = 1 / number_of_iterations * val
                else:
                    epoch_loss_stats[loss_stat] += 1 / number_of_iterations * val
            if split == "val":
                self.total_number_of_val_iterations += 1
            elif split.startswith("test"):
                self.total_number_of_test_iterations += 1
            else:
                raise RuntimeError(f"Unrecognized split {split}.")

        if split.startswith("test") and self.args.get_point_estimate:
            splits = [split] + (
                ["-".join([split, "prior"])] if self.args.predict_from_prior else []
            )
            for split_ in splits:
                mse, mae = self.get_point_estimate_errors(split_)
                self.summary_writer.add_scalar(
                    f"{split_}-point-estimate/mse",
                    mse,
                    self.total_number_of_val_iterations,
                )
                self.summary_writer.add_scalar(
                    f"{split_}-point-estimate/mae",
                    mae,
                    self.total_number_of_val_iterations,
                )
                dump_json(
                    {"mse": mse, "mae": mae},
                    get_full_path(
                        self.args,
                        "test_stats_point_estimate_{}.json".format(
                            split_.replace("-", "_")
                        ),
                    ),
                )

        # save extended test data (plots, errors, ...)
        if split.startswith("test"):
            # save test stats
            dump_json(
                {k: v for k, v in epoch_loss_stats.items()},
                get_full_path(
                    self.args, "test_stats_{}.json".format(split.replace("-", "_"))
                ),
            )

        return loss_stats

    def validate(self) -> Dict[str, torch.Tensor]:
        return self._test(self.val_dataloader, "val")

    def test(self, split: str) -> Dict[str, torch.Tensor]:
        return self._test(self.test_dataloader, split)

    def checkpoint(self, identifier: str = None) -> None:
        # create checkpoint to save
        dict_to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else {},
            "number_of_samples": self.number_of_samples,
            "test_summary": self.test_summary,
            "total_number_of_train_iterations": self.total_number_of_train_iterations,
            "total_number_of_test_iterations": self.total_number_of_test_iterations,
            "total_number_of_val_iterations": self.total_number_of_val_iterations,
            "total_number_of_epochs": self.total_number_of_epochs,
            "best_validation_score": self.best_validation_score,
            "number_of_trainable_parameters": self.number_of_trainable_parameters,
        }
        dict_to_save.update(self.checkpoint_additional_data())
        file_name = (
            f"checkpoints/checkpoint_{self.total_number_of_epochs}.pth"
            if identifier is None
            else f"checkpoints/checkpoint_{identifier}.pth"
        )
        torch.save(
            dict_to_save,
            get_full_path(self.args, file_name),
        )

    def load_checkpoint(self, path: str):
        # load checkpoint
        checkpoint = torch.load(path, map_location=get_device())
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if len(checkpoint["scheduler_state_dict"]) > 0:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.number_of_samples = checkpoint["number_of_samples"]
        self.test_summary = checkpoint["test_summary"]
        self.total_number_of_train_iterations = checkpoint[
            "total_number_of_train_iterations"
        ]
        self.total_number_of_test_iterations = checkpoint[
            "total_number_of_test_iterations"
        ]
        self.total_number_of_val_iterations = checkpoint[
            "total_number_of_val_iterations"
        ]
        self.total_number_of_epochs = checkpoint["total_number_of_epochs"]
        self.best_validation_score = checkpoint["best_validation_score"]
        self.number_of_trainable_parameters = checkpoint[
            "number_of_trainable_parameters"
        ]
        self.load_additional_data_from_checkpoint(checkpoint)

    def summarize_test(self, *args, **kwargs) -> None:
        raise RuntimeWarning(
            f"{self.summarize_test.__name__} is not implemented "
            f"for {self.__class__.__name__}."
        )

    def get_point_estimate_errors(self, split: str):
        raise RuntimeWarning(
            f"{self.get_point_estimate_errors.__name__} is not implemented "
            f"for {self.__class__.__name__}."
        )

    @abstractmethod
    def forward_model(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward_model_compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def unpack_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], ...]:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_additional_data(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_additional_data_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log(self) -> None:
        raise NotImplementedError
