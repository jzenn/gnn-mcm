import torch
from torch_scatter import scatter


def scatter_add(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    x_reduced = scatter(x, idx, dim=0, reduce="sum")
    return x_reduced


def scatter_mean(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    x_reduced = scatter(x, idx, dim=0, reduce="mean")
    return x_reduced


def degree(idx: torch.Tensor) -> torch.Tensor:
    return scatter_add(torch.ones(idx.shape[0], device=idx.device), idx)
