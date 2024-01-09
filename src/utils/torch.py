from typing import Iterable, List, Tuple

import numpy as np
import torch


def tensors_to_numpy(tensors: Iterable[torch.Tensor]) -> Iterable[np.ndarray]:
    return type(tensors)(t.detach().clone().cpu().numpy() for t in tensors)


def reduce_tensor(tensor: torch.Tensor, reduction: str = "sum"):
    if reduction == "sum":
        reduced_tensor = tensor.sum(-1)
    elif reduction == "none":
        reduced_tensor = tensor
    else:
        raise RuntimeError(f"Unrecognized reduction {reduction}.")
    return reduced_tensor


def mse_mae_from_differences(differences: List[float]) -> Tuple[float, float]:
    mse = torch.tensor(differences).norm(2) ** 2 / len(differences)
    mae = torch.tensor(differences).norm(1) / len(differences)
    return mse.item(), mae.item()
