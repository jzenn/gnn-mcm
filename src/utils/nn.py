import argparse
from typing import List, Union

import torch
import torch.nn as nn
import torch_geometric

from src.utils.graph import LayerNorm


def initialize_bias_to_zero(module: Union[nn.Module, List[nn.Module]]) -> None:
    if isinstance(module, (nn.Linear, torch_geometric.nn.dense.linear.Linear)):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (list, nn.ModuleList, nn.Sequential)):
        for m in module:
            initialize_bias_to_zero(m)


def get_act(args: argparse.Namespace, act: str) -> nn.Module:
    if act == "ReLU":
        act_ = nn.ReLU()
    elif act == "LeakyReLU":
        act_ = nn.LeakyReLU(negative_slope=args.leaky_relu_negative_slope)
    elif act == "tanh":
        act_ = nn.Tanh()
    elif act == "ELU":
        act_ = nn.ELU()
    else:
        raise RuntimeError(f"Unrecognized activation {act}.")
    return act_


def get_norm(
    args: argparse.Namespace, norm: str, num_features: int, graph=True
) -> nn.Module:
    if graph:
        print("Using norm for graph structure.")
    if norm == "none":
        norm_ = None
    elif norm == "LayerNorm":
        if graph:
            norm_ = LayerNorm(num_features)
        else:
            norm_ = nn.LayerNorm(num_features)
    else:
        raise RuntimeError(f"Unrecognized norm {norm}.")
    return norm_
