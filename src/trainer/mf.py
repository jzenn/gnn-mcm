import argparse
from abc import ABC
from typing import Any, Dict, List, Tuple

import torch

from src.trainer.trainer import Trainer
from src.utils.torch import mse_mae_from_differences


class MFTrainer(Trainer, ABC):
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloaders: List[torch.utils.data.dataloader.DataLoader],
        loss: torch.nn.Module,
    ):
        super(MFTrainer, self).__init__(args, model, dataloaders, loss)

    def log(self) -> None:
        pass

    def _summarize_test(
        self,
        matrix: torch.Tensor,
        solute_idx: torch.Tensor,
        solvent_idx: torch.Tensor,
        target: torch.Tensor,
        split: str,
    ) -> None:
        dataset = self.train_dataloader.dataset.dataset
        for i, (su_idx, sv_idx) in enumerate(zip(solute_idx, solvent_idx)):
            pred = matrix[su_idx, sv_idx]
            gt = target[i]
            self.test_summary.update(
                {
                    (su_idx.item(), sv_idx.item(), split): {
                        "split": split,
                        "solute_idx": su_idx.item(),
                        "solvent_idx": sv_idx.item(),
                        "prediction": pred.tolist(),
                        "target": gt.tolist(),
                        "difference": (pred - gt).tolist(),
                        # zero-shot
                        "solute_zero_shot": dataset.solute_is_zero_shot(su_idx).item()
                        if self.zero_shot_prediction
                        else "NotImplemented",
                        "solvent_zero_shot": dataset.solvent_is_zero_shot(sv_idx).item()
                        if self.zero_shot_prediction
                        else "NotImplemented",
                        "zero_shot": dataset.is_zero_shot(su_idx, sv_idx).item()
                        if self.zero_shot_prediction
                        else "NotImplemented",
                    }
                }
            )

    def get_point_estimate_errors(self, split: str) -> Tuple[float, float]:
        differences = list()
        for datum in self.test_summary.values():
            if datum["split"] == split:
                differences.append(datum["difference"])
        mse, mae = mse_mae_from_differences(differences)
        return mse, mae
