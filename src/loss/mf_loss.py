import argparse
from typing import Dict

import torch

from src.loss.metric_collection import MetricCollection


class NoPriorMFMSELoss(MetricCollection):
    def __init__(self, args: argparse.Namespace) -> None:
        super(NoPriorMFMSELoss, self).__init__(args)

    def forward(
        self, target: torch.Tensor, result: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        mse_loss = self.get_mse_loss(result.double(), target)
        mae_loss = self.get_mae_loss(result, target)
        min_abs_loss, max_abs_loss = self.get_min_max_abs_loss(result, target)

        return {
            "mse": mse_loss,
            "mae": mae_loss,
            "ae-min": min_abs_loss,
            "ae-max": max_abs_loss,
            "loss": mse_loss,
        }
