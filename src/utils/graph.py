import argparse
from typing import Callable

import torch
import torch.nn as nn

from src.utils.scatter import degree, scatter_add, scatter_mean


class LayerNorm(nn.Module):
    # takes inspiration from Pytorch Geometric
    # adapted to this frameworks data structure
    def __init__(
        self, in_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super(LayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.Tensor([in_channels]))
            self.bias = nn.Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        norm = degree(node_index).clamp_(min=1)  # assure that we do not devide by 0
        norm = norm.mul_(x.shape[-1]).view(-1, 1)  # normalization (#elements)

        mean = scatter_add(x, node_index).sum(dim=-1, keepdim=True)
        mean = mean / norm

        x = x - mean[node_index]

        var = scatter_add(x**2, node_index).sum(dim=-1, keepdim=True)
        var = var / norm

        out = x / (var + self.eps).sqrt()[node_index]

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out


def get_aggregator(
    args: argparse.Namespace, agg: str, in_channels: int = None
) -> Callable:
    if agg == "mean":

        def aggregator(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return scatter_mean(x, idx)

    else:
        raise RuntimeError(f"Unknown aggregator {agg}.")
    return aggregator
