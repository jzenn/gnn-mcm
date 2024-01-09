from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
from torch import nn as nn

from src.factor.cholesky_factor import CholeskyFactor
from src.utils.distribution import univariate_gaussian_log_prob


class PriorGaussianFactor(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_mean(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_std(self, **kwargs) -> Union[torch.Tensor, CholeskyFactor]:
        raise NotImplementedError

    @abstractmethod
    def update_params(self, params: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiagonalPriorGaussianFactor(PriorGaussianFactor):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        diagonal: bool = True,
        std_update_transform: str = "exp",
    ) -> None:
        super(DiagonalPriorGaussianFactor, self).__init__()
        self.std_update_transform = std_update_transform
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def update_params(self, params: torch.Tensor, **kwargs) -> None:
        self.mean = (
            params[:, : self.mean.shape[-1]]
            .clone()
            .view(params.shape[0], self.mean.shape[-1])
        )
        self.std = (
            params[:, self.mean.shape[-1] :]
            .clone()
            .view(params.shape[0], self.std.shape[-1])
        )
        if self.std_update_transform == "exp":
            self.std = self.std.exp()
        elif self.std_update_transform == "abs":
            self.std = self.std.abs()
        else:
            raise RuntimeError(
                f"Unrecognized update transform for std: "
                f"{self.std_update_transform}."
            )

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std

    def get_mean(self) -> torch.Tensor:
        return self.mean

    def get_std(self, **kwargs) -> torch.Tensor:
        get_as = kwargs.get("get_as")
        if get_as == "factor" or get_as is None:
            std = self.std
        elif get_as == "tril":
            std = torch.diag_embed(self.std)
        else:
            raise RuntimeError(f"Unrecognized type {get_as}.")
        return std

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return univariate_gaussian_log_prob(x, self.mean, self.std)
