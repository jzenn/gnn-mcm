import math
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Union

import torch
from torch import nn as nn

from src.factor.cholesky_factor import CholeskyFactor
from src.utils.distribution import (
    kl_divergence_from_sample_log_probabilities,
    sample_univariate_normal_gaussian,
    univariate_gaussian_entropy,
    univariate_gaussian_kl,
    univariate_gaussian_log_prob,
)
from src.utils.torch import reduce_tensor


class VariationalGaussianFactor(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get_sub_factor(self, index: torch.Tensor) -> "VariationalGaussianFactor":
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_mean(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_kl_div(
        self,
        right_mean: torch.Tensor,
        right_std: Union[torch.Tensor, CholeskyFactor],
        reduction: str = "sum",
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_kl_div(
        self,
        right_log_prob: torch.Tensor,
        m: int = -1,
        reduction: str = "sum",
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_entropy(self, reduction: str = "sum") -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, m: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_last_sample(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_and_get(self, m: int, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiagonalVariationalGaussianFactor(VariationalGaussianFactor):
    def __init__(self, shape: List[int], **kwargs) -> None:
        super(DiagonalVariationalGaussianFactor, self).__init__()
        assert len(shape) == 2

        self.n, self.k = shape
        self.last_reparametrized_sample = None

        # see: Jirasek et al. (2020)
        self.variational_mean = nn.Parameter(torch.full(shape, 0.0))
        self.variational_log_std = nn.Parameter(
            torch.full(shape, math.log(0.8 * 10 ** (-1)))
        )

    def get_sub_factor(self, index: torch.Tensor) -> "VariationalGaussianFactor":
        # create new factor ("new view")
        factor = DiagonalVariationalGaussianFactor(
            [len(index), self.k],
            initialize=False,
            variational_mean=self.variational_mean[index].clone(),
            variational_log_std=self.variational_log_std[index].clone(),
        )
        # set last sample
        factor.last_reparametrized_sample = self.last_reparametrized_sample[index]
        return factor

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.variational_mean, self.variational_log_std.exp()

    def get_mean(self) -> torch.Tensor:
        return self.variational_mean

    def compute_entropy(self, reduction: str = "sum") -> torch.Tensor:
        entropy = (
            univariate_gaussian_entropy(self.variational_log_std.exp())
            .view(self.n, -1)
            .sum(-1)
        )

        # reduction
        reduced_entropy = reduce_tensor(entropy, reduction)
        return reduced_entropy

    def compute_kl_div(
        self, right_mean: torch.Tensor, right_std: torch.Tensor, reduction: str = "sum"
    ) -> torch.Tensor:
        assert self.variational_mean.shape == right_mean.shape
        assert self.variational_log_std.shape == right_std.shape
        assert isinstance(right_mean, torch.Tensor)

        # sum over embedding dimension
        kl = (
            univariate_gaussian_kl(
                self.variational_mean.view(-1),
                self.variational_log_std.exp().view(-1),
                right_mean.view(-1),
                right_std.view(-1),
            )
            .view(self.n, -1)
            .sum(-1)
        )

        # reduction
        reduced_kl = reduce_tensor(kl, reduction)
        return reduced_kl

    def sample_kl_div(
        self, right_log_prob_function: Callable, m: int = -1, reduction: str = "sum"
    ) -> torch.Tensor:
        # use last sample if m is not provided
        if m < 0:
            z = self.get_last_sample().transpose(0, 1)  # change sampling dimension to 0
        else:
            z = self.sample(m).transpose(0, 1)  # change sampling dimension to 0

        kl = kl_divergence_from_sample_log_probabilities(
            self.log_prob(z), right_log_prob_function(z)
        ).sum(-1)

        # reduction
        reduced_kl = reduce_tensor(kl, reduction)
        return reduced_kl

    def sample(self, m: int) -> torch.Tensor:
        shape = (self.n, m, self.k)
        normal_sample = sample_univariate_normal_gaussian(shape)
        reparametrized_sample = (
            self.variational_mean.unsqueeze(1)
            + self.variational_log_std.exp().unsqueeze(1) * normal_sample
        )
        self.last_reparametrized_sample = reparametrized_sample.clone()
        return reparametrized_sample

    def get_last_sample(self):
        return self.last_reparametrized_sample

    def sample_and_get(self, m: int, idx: torch.Tensor) -> torch.Tensor:
        sample = self.sample(m)
        return sample[idx]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return univariate_gaussian_log_prob(
            x, self.variational_mean, self.variational_log_std.exp()
        )
