import argparse
from typing import Any, Dict, List, Tuple

import torch

from src.trainer.mf import MFTrainer
from src.utils.experiment import get_device, save_numpy_arrays
from src.utils.torch import tensors_to_numpy


class PMFDDBTrainer(MFTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: torch.nn.Module,
    ) -> None:
        super(PMFDDBTrainer, self).__init__(args, model, dataloaders, loss)

        # weighting of loss values
        self.solutes_weight_numerator = (
            self.args.number_of_solutes - self.args.n_solutes_for_zero_shot
        )
        self.solvents_weight_numerator = (
            self.args.number_of_solvents - self.args.n_solvents_for_zero_shot
        )

    def forward_model(
        self, solute_idx: torch.Tensor, solvent_idx: torch.Tensor, split: str
    ) -> torch.Tensor:
        result = self.model(solute_idx, solvent_idx)

        # update prior parameters in loss
        if self.zero_shot_prediction:
            # update solutes prior factor
            self.loss.update_solutes_prior_factor(
                torch.cat(
                    [
                        self.loss.solutes_prior_factor.get_mean(),
                        self.loss.solutes_prior_factor.get_std().log(),
                    ],
                    -1,
                )[solute_idx.view(-1)]
            )

            # update solvents prior factor
            self.loss.update_solvents_prior_factor(
                torch.cat(
                    [
                        self.loss.solvents_prior_factor.get_mean(),
                        self.loss.solvents_prior_factor.get_std().log(),
                    ],
                    -1,
                )[solvent_idx.view(-1)]
            )

        return result

    def compute_loss(
        self, result: torch.Tensor, target: torch.Tensor, split: str, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self.zero_shot_prediction:
            solutes_idx: torch.Tensor = kwargs["solutes_idx"]
            solvents_idx: torch.Tensor = kwargs["solvents_idx"]
            # get sub-factor of solutes according to solutes_idx
            solutes_factor = self.model.get_solutes_factor().get_sub_factor(
                solutes_idx.view(-1)
            )
            # get sub-factor of solvents according to solvents_idx
            solvents_factor = self.model.get_solvents_factor().get_sub_factor(
                solvents_idx.view(-1)
            )
            b = len(result)
            loss_weights = {
                "data_log_likelihood_weight": self.number_of_samples[split] / b,
                "solutes_weight": (self.solutes_weight_numerator / b)
                if self.args.vi_loss_simple_weighting
                else kwargs["solutes_weight"].view(-1),
                "solvents_weight": (self.solvents_weight_numerator / b)
                if self.args.vi_loss_simple_weighting
                else kwargs["solvents_weight"].view(-1),
                "loss_weight": (1.0 / self.number_of_samples[split])
                if self.args.vi_loss_mean
                else 1.0,
                "likelihood_weight": 1.0,
                "kl_weight": 1.0,
            }
        else:
            loss_weights = {
                "data_log_likelihood_weight": self.number_of_samples[split]
                / len(result),
                "solutes_weight": 1.0,  # no need for weights since KL is computed for
                "solvents_weight": 1.0,  # every batch over all solutes and solvents
                "loss_weight": (1.0 / self.number_of_samples[split])
                if self.args.vi_loss_mean
                else 1.0,
                "likelihood_weight": 1.0,
                "kl_weight": 1.0,
            }
            solutes_factor = self.model.get_solutes_factor()
            solvents_factor = self.model.get_solvents_factor()

        # compute loss
        loss_stats = self.loss(
            target,
            solutes_factor,
            solvents_factor,
            result,
            loss_weights,
        )
        return loss_stats

    def forward_model_compute_loss(
        self,
        solutes_data: torch.Tensor,
        solvents_data: torch.Tensor,
        target: torch.Tensor,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        solutes_idx, solvents_idx = solutes_data["idx"], solvents_data["idx"]
        result = self.forward_model(solutes_idx, solvents_idx, split)

        if (
            self.zero_shot_prediction or self.args.validate_pmf_with_mse
        ) and split == "val":
            result = (
                self.loss.solutes_prior_factor.get_mean()
                * self.loss.solvents_prior_factor.get_mean()
            ).sum(-1, keepdim=True)
            mse, mae, ae_min, ae_max = self.loss.get_loss_metrics(result, target)
            loss_stats = {
                "mse": mse,
                "mae": mae,
                "ae-min": ae_min,
                "ae-max": ae_max,
                # early stopping with MSE
                "loss": mse,
            }
        else:
            loss_stats = self.compute_loss(
                result,
                target,
                split,
                solutes_idx=solutes_idx,
                solvents_idx=solvents_idx,
                solutes_weight=solutes_data["solutes_weight"]
                * self.solvents_weight_numerator
                / solutes_data["unique_solutes"],
                solvents_weight=solvents_data["solvents_weight"]
                * self.solvents_weight_numerator
                / solvents_data["unique_solvents"],
            )
        return loss_stats

    @staticmethod
    def get_likelihood_kl_weights():
        likelihood_weight = 1.0
        kl_weight = 1.0
        return likelihood_weight, kl_weight

    def checkpoint_additional_data(self) -> Dict[str, Any]:
        return dict()

    def unpack_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        target = data["data"].to(get_device())
        solutes_data = {
            "idx": data["solute_idx"].to(get_device()),
            "solutes_weight": data["solute_weight"].to(get_device()),
            "unique_solutes": data["unique_solutes"].to(get_device()),
        }
        solvents_data = {
            "idx": data["solvent_idx"].to(get_device()),
            "solvents_weight": data["solvent_weight"].to(get_device()),
            "unique_solvents": data["unique_solvents"].to(get_device()),
        }
        return solutes_data, solvents_data, target

    def summarize_test(
        self,
        solutes_data: torch.Tensor,
        solvents_data: torch.Tensor,
        target: torch.Tensor,
        split: str,
    ) -> None:
        solutes_idx, solvents_idx = solutes_data["idx"], solvents_data["idx"]
        matrix = self.model.get_matrix_point_estimate()
        self._summarize_test(matrix, solutes_idx, solvents_idx, target, split)

    def load_additional_data_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def log(self) -> None:
        solutes_mean, solutes_std = tensors_to_numpy(
            self.model.get_solutes_factor().get()
        )
        solvents_mean, solvents_std = tensors_to_numpy(
            self.model.get_solvents_factor().get()
        )

        save_numpy_arrays(
            self.args,
            solutes_mean=solutes_mean,
            solutes_std=solutes_std,
            solvents_mean=solvents_mean,
            solvents_std=solvents_std,
        )
