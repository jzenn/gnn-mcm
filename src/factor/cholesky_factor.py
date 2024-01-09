from typing import List, Union

import torch
from torch import nn as nn

from src.utils.experiment import get_device


def tril_vec_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    k = matrix.shape[-1]
    tril_row_idx, tril_col_idx = torch.tril_indices(k, k).unbind()
    return matrix[..., tril_row_idx, tril_col_idx]


class CholeskyFactor(nn.Module):
    def __init__(
        self,
        data: Union[torch.Tensor, None],
        dimensions: List[int],
        requires_grad: bool = True,
    ) -> None:
        super(CholeskyFactor, self).__init__()
        self.dimensions = dimensions
        self.req_grad = requires_grad
        if len(dimensions) == 2:
            self.b, self.d = dimensions
        else:
            raise RuntimeError(f"Cannot handle {len(dimensions)} dimensions.")

        # indices for row, col, diag (of data)
        self.tril_row_idx, self.tril_col_idx = torch.tril_indices(
            self.d, self.d
        ).unbind()
        self.tril_diag_idx = self.tril_row_idx == self.tril_col_idx

        # actual data
        if data is not None:
            if self.req_grad:
                self._data = nn.Parameter(data)
            else:
                self.register_buffer("_data", data)
        else:
            self._data = data

    def get_sub_factor(self, index: torch.Tensor) -> "CholeskyFactor":
        dimensions = self.dimensions
        dimensions[0] = len(index)
        # create new factor ("new view")
        factor = CholeskyFactor(None, dimensions)
        # set data to view
        factor._data = self._data.clone()[index]
        return factor

    def update_raw(self, data: torch.Tensor, diagonal: bool = False) -> None:
        # update batch dimension
        self.b = data.shape[0]
        self.dimensions[0] = self.b
        # update data
        if diagonal:
            self._data = tril_vec_from_matrix(
                torch.diag_embed(data.view(self.dimensions))
            )
        else:
            self._data = data.view(data.shape[0], self._data.shape[-1])

    def get_raw(self) -> torch.Tensor:
        return self._data

    def transform(self) -> torch.Tensor:
        data = self.get_raw().clone()
        data[..., self.tril_diag_idx] = data[..., self.tril_diag_idx].abs()
        return data

    def get(self) -> torch.Tensor:
        return self.transform()

    def tril(self) -> torch.Tensor:
        L = torch.zeros(self.b, self.d, self.d).to(get_device())
        L[..., self.tril_row_idx, self.tril_col_idx] = self.get()
        return L

    def diag(self) -> torch.Tensor:
        return self.get_raw().clone()[..., self.tril_diag_idx].abs()

    def __len__(self):
        return self._data.shape[-1]
