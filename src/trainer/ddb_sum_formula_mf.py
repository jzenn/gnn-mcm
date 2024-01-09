import argparse
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from src.trainer.mf import MFTrainer
from src.utils.dataset import collate_graph_batch
from src.utils.experiment import get_device, save_numpy_arrays


class SumFormulaMFDDBTrainer(MFTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: nn.Module,
    ):
        super(SumFormulaMFDDBTrainer, self).__init__(args, model, dataloaders, loss)

    def update_molecule_data_for_prediction(
        self, data: Dict[str, Any], idx: torch.Tensor
    ) -> Dict[str, Any]:
        data.update(
            {
                "idx": torch.tensor([idx]),
                "embedding": data["sum_formula_embedding"],
            }
        )
        return data

    def get_parameters(
        self, solute_idx: torch.Tensor, solvent_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # molecule data of solutes and solvents
        (
            solutes_molecule_data,
            solvents_molecule_data,
        ) = self.train_dataloader.dataset.dataset.get_all_molecule_data()

        solutes_embedding = torch.zeros(
            len(solute_idx), self.args.dimensionality_of_embedding
        )
        solvents_embedding = torch.zeros(
            len(solvent_idx), self.args.dimensionality_of_embedding
        )

        for i, (su_idx, sv_idx) in enumerate(zip(solute_idx, solvent_idx)):
            with torch.no_grad():
                solutes_data = collate_graph_batch(
                    [solutes_molecule_data[su_idx.item()]]
                )
                solutes_data = self.update_molecule_data_for_prediction(
                    solutes_data, su_idx
                )
                solvents_data = collate_graph_batch(
                    [solvents_molecule_data[sv_idx.item()]]
                )
                solvents_data = self.update_molecule_data_for_prediction(
                    solvents_data, sv_idx
                )

                # push to device
                for data in [solutes_data, solvents_data]:
                    for k, t in data.items():
                        if isinstance(t, torch.Tensor):
                            data.update({k: t.to(get_device())})

                solute_embedding_params, solvent_embedding_params, _ = self.model(
                    solutes_data,
                    solvents_data,
                )

            # to cpu
            solute_embedding_params = solute_embedding_params.detach().cpu()
            solvent_embedding_params = solvent_embedding_params.detach().cpu()
            # solutes
            solutes_embedding[i] = solute_embedding_params
            solvents_embedding[i] = solvent_embedding_params

        return solutes_embedding, solvents_embedding

    def summarize_test(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        target: torch.Tensor,
        split: str,
    ) -> None:
        # index
        solute_idx = solutes_data["idx"]
        solvent_idx = solvents_data["idx"]

        # forward model, create matrix to use _summarize_test
        result = self.forward_model(solutes_data, solvents_data, split)
        matrix = torch.zeros(self.args.number_of_solutes, self.args.number_of_solvents)
        for i, (su_idx, sv_idx) in enumerate(zip(solute_idx, solvent_idx)):
            matrix[su_idx.item(), sv_idx.item()] = result[i]

        # regular prediction
        self._summarize_test(
            matrix.to(solute_idx.device), solute_idx, solvent_idx, target, split
        )

        # prior prediction not possible

    def unpack_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], ...]:
        target = data["data"].to(get_device())
        solutes_data = {
            "idx": data["solute_idx"].to(get_device()),
            "solutes_weight": data["solute_weight"].to(get_device()),
            "node_index": data["solute_node_index"].to(get_device()),
            "embedding": data["solute_sum_formula_embedding"].to(get_device()),
        }
        solvents_data = {
            "idx": data["solvent_idx"].to(get_device()),
            "solvents_weight": data["solvent_weight"].to(get_device()),
            "node_index": data["solvent_node_index"].to(get_device()),
            "embedding": data["solvent_sum_formula_embedding"].to(get_device()),
        }
        return solutes_data, solvents_data, target

    def forward_model(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        split: str,
    ) -> torch.Tensor:
        _, _, mf_estimate = self.model(solutes_data, solvents_data)
        mf_estimate = mf_estimate.unsqueeze(-1)
        return mf_estimate

    def compute_loss(
        self, result: torch.Tensor, target: torch.Tensor, split: str, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # compute loss
        loss_stats = self.loss(target, result)
        return loss_stats

    def forward_model_compute_loss(
        self,
        solutes_data: Dict[str, torch.Tensor],
        solvents_data: Dict[str, torch.Tensor],
        target: torch.Tensor,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        result = self.forward_model(solutes_data, solvents_data, split)
        loss_stats = self.compute_loss(result, target, split)
        return loss_stats

    def checkpoint_additional_data(self) -> Dict[str, Any]:
        # get checkpoint of super
        return dict()

    def load_additional_data_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def log(self) -> None:
        # use logging of super
        super().log()

        # log additional prior mean and prior std
        (
            solute_idx,
            solvent_idx,
        ) = self.train_dataloader.dataset.dataset.get_index_vector_for_dataset()

        # log additional prior mean and prior std
        solutes_embedding, solvents_embedding = self.get_parameters(
            torch.tensor(solute_idx).to(get_device()),
            torch.tensor(solvent_idx).to(get_device()),
        )

        # save
        save_numpy_arrays(
            self.args,
            solutes_embedding=solutes_embedding.numpy(),
            solvents_embedding=solvents_embedding.numpy(),
        )
