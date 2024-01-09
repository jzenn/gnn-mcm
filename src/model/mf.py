import argparse

import torch
from torch import nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(MatrixFactorization, self).__init__()

    def forward(
        self, solutes_features: torch.Tensor, solvents_features: torch.Tensor
    ) -> torch.Tensor:
        # solute_features shape, solvents_features shape: b x (s) x K
        result = (solutes_features * solvents_features).sum(-1)
        return result
