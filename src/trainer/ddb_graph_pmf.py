import argparse
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from src.trainer.ddb_pmf import PMFDDBTrainer
from src.utils.dataset import collate_graph_batch
from src.utils.experiment import get_device, save_numpy_arrays


class GraphScalarPriorPMFDDBTrainer(PMFDDBTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: nn.Module,
    ):
        super(GraphScalarPriorPMFDDBTrainer, self).__init__(
            args, model, dataloaders, loss
        )
        # weighting of loss values
        self.solutes_weight_numerator = (
            self.args.number_of_solutes - self.args.n_solutes_for_zero_shot
        )
        self.solvents_weight_numerator = (
            self.args.number_of_solvents - self.args.n_solvents_for_zero_shot
        )

        # embedding dimension
        self.dim_of_mean_embedding = self.args.dimensionality_of_embedding
        if self.args.model.startswith("Diagonal") or self.args.diagonal_prior:
            self.dim_of_std_embedding = self.args.dimensionality_of_embedding
        else:
            raise RuntimeError(
                f"Invalid combination of input arguments "
                f"({self.args.model}, {self.args.diagonal_prior}, "
                f"{self.args.learn_scalar_prior_factor})."
            )

    def update_molecule_data_for_prior_prediction(
        self, data: Dict[str, Any], idx: torch.Tensor
    ) -> Dict[str, Any]:
        data.update(
            {
                "idx": torch.tensor([idx]),
                "atomic_numbers_mapping": data["atoms_simple_embedding"],
                "bond_types_mapping": data["bonds_simple_embedding"],
            }
        )
        return data

    def get_prior_parameters(
        self, solute_idx: torch.Tensor, solvent_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(solute_idx) == len(solvent_idx)

        # solutes
        solutes_mean = torch.zeros(len(solute_idx), self.dim_of_mean_embedding)
        solutes_std = torch.zeros(len(solute_idx), self.dim_of_std_embedding)
        solvents_mean = torch.zeros(len(solvent_idx), self.dim_of_mean_embedding)
        solvents_std = torch.zeros(len(solvent_idx), self.dim_of_std_embedding)

        # molecule data of solutes and solvents
        (
            solutes_molecule_data,
            solvents_molecule_data,
        ) = self.train_dataloader.dataset.dataset.get_all_molecule_data()

        for i, (su_idx, sv_idx) in enumerate(zip(solute_idx, solvent_idx)):
            with torch.no_grad():
                solutes_data = collate_graph_batch(
                    [solutes_molecule_data[su_idx.item()]]
                )
                solutes_data = self.update_molecule_data_for_prior_prediction(
                    solutes_data, su_idx
                )
                solvents_data = collate_graph_batch(
                    [solvents_molecule_data[sv_idx.item()]]
                )
                solvents_data = self.update_molecule_data_for_prior_prediction(
                    solvents_data, sv_idx
                )

                # push to device
                for data in [solutes_data, solvents_data]:
                    for k, t in data.items():
                        if isinstance(t, torch.Tensor):
                            data.update({k: t.to(get_device())})

                solute_prior_params, solvent_prior_params, _ = self.model(
                    solutes_data,
                    solvents_data,
                )
            # to cpu
            solute_prior_params = solute_prior_params.detach().cpu()
            solvent_prior_params = solvent_prior_params.detach().cpu()
            # solutes
            solutes_mean[i] = solute_prior_params[0, : self.dim_of_mean_embedding]
            solutes_std[i] = solute_prior_params[0, self.dim_of_mean_embedding :]
            # solvents
            solvents_mean[i] = solvent_prior_params[0, : self.dim_of_mean_embedding]
            solvents_std[i] = solvent_prior_params[0, self.dim_of_mean_embedding :]

        return (
            solutes_mean.to(get_device()),
            solutes_std.to(get_device()),
            solvents_mean.to(get_device()),
            solvents_std.to(get_device()),
        )

    def get_prior_point_estimate(
        self, solute_idx: torch.Tensor, solvent_idx: torch.Tensor
    ) -> torch.Tensor:
        solutes_mean, _, solvents_mean, _ = self.get_prior_parameters(
            solute_idx, solvent_idx
        )
        prior_point_estimate = torch.zeros(
            self.args.number_of_solutes, self.args.number_of_solvents
        ).to(get_device())
        for su_idx, sv_idx, ppe in zip(
            solute_idx,
            solvent_idx,
            (solutes_mean * solvents_mean).sum(-1, keepdim=True),
        ):
            prior_point_estimate[su_idx.item(), sv_idx.item()] = ppe
        return prior_point_estimate

    def summarize_test(
        self,
        solute_data: torch.Tensor,
        solvent_data: torch.Tensor,
        target: torch.Tensor,
        split: str,
    ) -> None:
        # index
        solute_idx = solute_data["idx"]
        solvent_idx = solvent_data["idx"]

        # regular prediction
        matrix = self.model.get_matrix_point_estimate()
        self._summarize_test(matrix, solute_idx, solvent_idx, target, split)

        # prediction with prior parameters
        if self.args.predict_from_prior:
            matrix = self.get_prior_point_estimate(solute_idx, solvent_idx)
            self._summarize_test(
                matrix, solute_idx, solvent_idx, target, "-".join([split, "prior"])
            )

    def unpack_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], ...]:
        target = data["data"].to(get_device())
        solutes_data = {
            "idx": data["solute_idx"].to(get_device()),
            "solutes_weight": data["solute_weight"].to(get_device()),
            "unique_solutes": data["unique_solutes"].to(get_device()),
            "node_index": data["solute_node_index"].to(get_device()),
            "edge_index": data["solute_edge_index"].to(get_device()),
            "atomic_numbers_mapping": data["solute_atoms_simple_embedding"].to(
                get_device()
            ),
            "bond_types_mapping": data["solute_bonds_simple_embedding"].to(
                get_device()
            ),
        }
        solvents_data = {
            "idx": data["solvent_idx"].to(get_device()),
            "solvents_weight": data["solvent_weight"].to(get_device()),
            "unique_solvents": data["unique_solvents"].to(get_device()),
            "node_index": data["solvent_node_index"].to(get_device()),
            "edge_index": data["solvent_edge_index"].to(get_device()),
            "atomic_numbers_mapping": data["solvent_atoms_simple_embedding"].to(
                get_device()
            ),
            "bond_types_mapping": data["solvent_bonds_simple_embedding"].to(
                get_device()
            ),
        }
        return solutes_data, solvents_data, target

    def forward_model(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        solutes_prior_params, solvents_prior_params, pmf_estimate = self.model(
            solutes_data, solvents_data
        )
        return solutes_prior_params, solvents_prior_params, pmf_estimate

    def compute_loss(
        self, result: torch.Tensor, target: torch.Tensor, split: str, **kwargs
    ) -> Dict[str, torch.Tensor]:
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

        likelihood_weight, kl_weight = self.get_likelihood_kl_weights()

        b = len(result)
        loss_weights = {
            "data_log_likelihood_weight": self.number_of_samples[split] / b,
            "solutes_weight": kwargs["solutes_weight"].view(-1),
            "solvents_weight": kwargs["solvents_weight"].view(-1),
            "loss_weight": (1.0 / self.number_of_samples[split]),
            "likelihood_weight": likelihood_weight,
            "kl_weight": kl_weight,
        }

        # compute loss
        loss_stats = self.loss(
            target, solutes_factor, solvents_factor, result, loss_weights
        )
        return loss_stats

    def forward_model_compute_loss(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        target: torch.Tensor,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        # check whether to fix the parameters of the model
        self.check_and_fix_parameters()

        solutes_params, solvents_params, result = self.forward_model(
            solutes_data, solvents_data, split
        )
        # learn single scalar for mean and std for embedding dimension
        b, n = solutes_params.shape
        assert n == 2

        # repeat parameters into shape
        solutes_prior_params = torch.repeat_interleave(
            solutes_params.unsqueeze(-1),
            self.args.dimensionality_of_embedding,
            -1,
        ).view(b, -1)
        solvents_prior_params = torch.repeat_interleave(
            solvents_params.unsqueeze(-1),
            self.args.dimensionality_of_embedding,
            -1,
        ).view(b, -1)
        # update solutes parameters in the prior factor of the loss
        self.loss.update_solutes_prior_factor(solutes_prior_params)

        # update solvents parameters in the prior factor of the loss
        self.loss.update_solvents_prior_factor(solvents_prior_params)

        if (self.zero_shot_prediction or self.args.validate_pmf_with_mse) and (
            split == "val" or split.startswith("test")
        ):
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
            # compute the loss (with updated prior parameters)
            loss_stats = self.compute_loss(
                result,
                target,
                split,
                solutes_idx=solutes_data["idx"],
                solvents_idx=solvents_data["idx"],
                solutes_weight=solutes_data["solutes_weight"]
                * self.solvents_weight_numerator
                / solutes_data["unique_solutes"],
                solvents_weight=solvents_data["solvents_weight"]
                * self.solvents_weight_numerator
                / solvents_data["unique_solvents"],
            )
        return loss_stats

    def checkpoint_additional_data(self) -> Dict[str, Any]:
        # get checkpoint of super
        checkpoint_dict = super().checkpoint_additional_data()

        # update checkpoint by own parameters
        checkpoint_dict.update(
            {
                "solutes_weight_numerator": self.solutes_weight_numerator,
                "solvents_weight_numerator": self.solvents_weight_numerator,
            }
        )
        return checkpoint_dict

    def load_additional_data_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().load_additional_data_from_checkpoint(checkpoint)
        self.solutes_weight_numerator = checkpoint["solutes_weight_numerator"]
        self.solvents_weight_numerator = checkpoint["solvents_weight_numerator"]

    def log(self) -> None:
        # use logging of super
        super().log()

        # log additional prior mean and prior std
        (
            solute_idx,
            solvent_idx,
        ) = self.train_dataloader.dataset.dataset.get_index_vector_for_dataset()
        (
            solutes_prior_mean,
            solutes_prior_std,
            solvents_prior_mean,
            solvents_prior_std,
        ) = self.get_prior_parameters(
            torch.tensor(solute_idx).to(get_device()),
            torch.tensor(solvent_idx).to(get_device()),
        )

        # save
        save_numpy_arrays(
            self.args,
            solutes_prior_mean=solutes_prior_mean.cpu().numpy(),
            solutes_prior_std=solutes_prior_std.cpu().numpy(),
            solvents_prior_mean=solvents_prior_mean.cpu().numpy(),
            solvents_prior_std=solvents_prior_std.cpu().numpy(),
        )


class GraphPriorPMFDDBTrainer(GraphScalarPriorPMFDDBTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: nn.Module,
    ):
        super(GraphPriorPMFDDBTrainer, self).__init__(args, model, dataloaders, loss)

    def forward_model_compute_loss(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        target: torch.Tensor,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        # get prior parameters for solutes and solvents
        solutes_prior_params, solvents_prior_params, result = self.forward_model(
            solutes_data, solvents_data, split
        )

        # update solutes parameters in the prior factor of the loss
        self.loss.update_solutes_prior_factor(solutes_prior_params)
        # update solvents parameters in the prior factor of the loss
        self.loss.update_solvents_prior_factor(solvents_prior_params)

        if (self.zero_shot_prediction or self.args.validate_pmf_with_mse) and (
            split == "val" or split.startswith("test")
        ):
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
            # compute the loss (with updated prior parameters)
            loss_stats = self.compute_loss(
                result,
                target,
                split,
                solutes_idx=solutes_data["idx"],
                solvents_idx=solvents_data["idx"],
                solutes_weight=solutes_data["solutes_weight"]
                * self.solvents_weight_numerator
                / solutes_data["unique_solutes"],
                solvents_weight=solvents_data["solvents_weight"]
                * self.solvents_weight_numerator
                / solvents_data["unique_solvents"],
            )
        return loss_stats


class GraphPriorPMFWBDDBTrainer(GraphPriorPMFDDBTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: nn.Module,
    ):
        super(GraphPriorPMFWBDDBTrainer, self).__init__(args, model, dataloaders, loss)
        self.use_solutes_bias = args.use_solutes_bias
        self.use_solvents_bias = args.use_solvents_bias
        self.use_global_bias = args.use_global_bias

        self.solutes_bias_weight_numerator = (
            self.args.number_of_solutes - self.args.n_solutes_for_zero_shot
        )
        self.solvents_bias_weight_numerator = (
            self.args.number_of_solvents - self.args.n_solvents_for_zero_shot
        )
        self.global_bias_weight_numerator = 1.0

    def get_prior_parameters(self, **kwargs) -> None:
        raise NotImplementedError

    def compute_loss(
        self, result: torch.Tensor, target: torch.Tensor, split: str, **kwargs
    ) -> Dict[str, torch.Tensor]:
        solutes_idx = kwargs["solutes_idx"]
        solvents_idx = kwargs["solvents_idx"]
        # get sub-factor of solutes according to solutes_idx
        solutes_factor = self.model.get_solutes_factor().get_sub_factor(
            solutes_idx.view(-1)
        )
        # get sub-factor of solvents according to solvents_idx
        solvents_factor = self.model.get_solvents_factor().get_sub_factor(
            solvents_idx.view(-1)
        )
        # get sub-factor of solutes bias according to solutes_idx
        solutes_bias_factor = self.model.get_solutes_bias_factor().get_sub_factor(
            solutes_idx.view(-1)
        )
        # get sub-factor of solvents bias according to solutes_idx
        solvents_bias_factor = self.model.get_solvents_bias_factor().get_sub_factor(
            solvents_idx.view(-1)
        )
        # get global bias factor
        global_bias_factor = self.model.get_global_bias_factor().get_sub_factor(
            torch.zeros(len(result)).long()
        )

        b = len(result)
        solutes_weight = (
            (self.solutes_weight_numerator / b)
            if self.args.vi_loss_simple_weighting
            else kwargs["solutes_weight"].view(-1)
        )
        solvents_weight = (
            (self.solvents_weight_numerator / b)
            if self.args.vi_loss_simple_weighting
            else kwargs["solvents_weight"].view(-1)
        )

        likelihood_weight, kl_weight = self.get_likelihood_kl_weights()

        loss_weights = {
            "data_log_likelihood_weight": self.number_of_samples[split] / b,
            "solutes_weight": solutes_weight,
            "solvents_weight": solvents_weight,
            "solutes_bias_weight": solutes_weight,
            "solvents_bias_weight": solvents_weight,
            "global_bias_weight": self.global_bias_weight_numerator / b,
            "loss_weight": (1.0 / self.number_of_samples[split])
            if self.args.vi_loss_mean
            else 1.0,
            "likelihood_weight": likelihood_weight,
            "kl_weight": kl_weight,
        }

        # compute loss
        loss_stats = self.loss(
            target,
            solutes_factor,
            solvents_factor,
            solutes_bias_factor,
            solvents_bias_factor,
            global_bias_factor,
            result,
            loss_weights,
        )
        return loss_stats

    def forward_model_compute_loss(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        target: torch.Tensor,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        # get params
        solutes_params, solvents_params, result = self.forward_model(
            solutes_data, solvents_data, split
        )
        # unpack params
        solutes_prior_params = (
            solutes_params[:, : -(2 + 2)]
            if self.use_solutes_bias or self.use_global_bias
            else solutes_params
        )
        solvents_prior_params = (
            solvents_params[:, : -(2 + 2)]
            if self.use_solvents_bias or self.use_global_bias
            else solvents_params
        )
        solutes_bias_prior_params = (
            solutes_params[:, -(2 + 2) : -2] if self.use_solutes_bias else None
        )
        solvents_bias_prior_params = (
            solvents_params[:, -(2 + 2) : -2] if self.use_solvents_bias else None
        )
        # add params for global bias (consider both, solutes and solvents)
        global_bias_params = (
            (solutes_params[:, -2:] + solvents_params[:, -2:])
            if self.use_global_bias
            else None
        )

        # update solutes parameters in the prior factor of the loss
        self.loss.update_solutes_prior_factor(solutes_prior_params)
        # update solvents parameters in the prior factor of the loss
        self.loss.update_solvents_prior_factor(solvents_prior_params)

        if self.use_solutes_bias:
            # update solutes prior parameters in the prior factor of the loss
            self.loss.update_solutes_bias_prior_factor(solutes_bias_prior_params)
        if self.use_solvents_bias:
            # update solvents prior parameters in the prior factor of the loss
            self.loss.update_solvents_bias_prior_factor(solvents_bias_prior_params)
        if self.use_global_bias:
            # update global prior parameters in the prior factor of the loss
            self.loss.update_global_bias_prior_factor(global_bias_params)

        # compute the loss (with updated prior parameters)
        loss_stats = self.compute_loss(
            result,
            target,
            split,
            solutes_idx=solutes_data["idx"],
            solvents_idx=solutes_data["idx"],
            solutes_weight=solutes_data["solutes_weight"]
            * self.solvents_weight_numerator
            / solutes_data["unique_solutes"],
            solvents_weight=solvents_data["solvents_weight"]
            * self.solvents_weight_numerator
            / solvents_data["unique_solvents"],
        )
        return loss_stats

    def checkpoint_additional_data(self) -> Dict[str, Any]:
        # get checkpoint of super
        checkpoint_dict = super().checkpoint_additional_data()

        # update checkpoint by own parameters
        checkpoint_dict.update(
            {
                "solutes_bias_weight_numerator": self.solutes_bias_weight_numerator,
                "solvents_bias_weight_numerator": self.solvents_bias_weight_numerator,
                "global_bias_weight_numerator": self.global_bias_weight_numerator,
            }
        )
        return checkpoint_dict

    def load_additional_data_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().load_additional_data_from_checkpoint(checkpoint)
        self.solutes_bias_weight_numerator = checkpoint["solutes_bias_weight_numerator"]
        self.solvents_bias_weight_numerator = checkpoint[
            "solvents_bias_weight_numerator"
        ]
        self.global_bias_weight_numerator = checkpoint["global_bias_weight_numerator"]
