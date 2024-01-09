import argparse
from typing import Tuple

import torch
from torch import nn as nn


class MetricCollection(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(MetricCollection, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mae_loss = nn.L1Loss(reduction="mean")
        self.ae_loss_no_reduction = nn.L1Loss(reduction="none")

        data_likelihood_std_tensor = torch.tensor([args.data_likelihood_std])
        self.register_buffer("data_likelihood_std", data_likelihood_std_tensor)

    def get_mse_loss(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        return self.mse_loss(prediction, ground_truth)

    def get_mae_loss(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        return self.mae_loss(prediction, ground_truth)

    def get_min_max_abs_loss(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entrywise_mae_loss = self.ae_loss_no_reduction(prediction, ground_truth)
        return entrywise_mae_loss.min(), entrywise_mae_loss.max()

    def get_loss_metrics(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.get_mse_loss(prediction, ground_truth),
            self.get_mae_loss(prediction, ground_truth),
            *self.get_min_max_abs_loss(prediction, ground_truth),
        )
