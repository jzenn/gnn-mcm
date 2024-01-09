import math

import torch
import torch.distributions as dist

from src.factor.cholesky_factor import CholeskyFactor
from src.utils.experiment import get_device

# constant for entropy
LN_SQRT_2_PI = 0.5 * math.log(2 * math.pi)


def kl_divergence_from_sample_log_probabilities(
    left_sample_log_prob: torch.Tensor, right_sample_log_prob: torch.Tensor
) -> torch.Tensor:
    per_sample_kl = left_sample_log_prob - right_sample_log_prob
    kl = per_sample_kl.mean(0)
    return kl


def univariate_gaussian_kl(
    left_mean: torch.Tensor,
    left_std: torch.Tensor,
    right_mean: torch.Tensor,
    right_std: torch.Tensor,
) -> torch.Tensor:
    left = torch.log(right_std / left_std)
    right = (left_std**2 + (left_mean - right_mean) ** 2) / (2 * right_std**2)
    result = left + right - 1 / 2
    return result


def univariate_gaussian_entropy(std: torch.Tensor):
    return 0.5 + torch.log(std) + LN_SQRT_2_PI


def univariate_gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    d = dist.normal.Normal(mean, std)
    return d.log_prob(x)


def multivariate_gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, cholesky_factor_tril: torch.Tensor
) -> torch.Tensor:
    # internal operations are based on scale_tril
    d = dist.MultivariateNormal(mean, scale_tril=cholesky_factor_tril)
    return d.log_prob(x)


def sample_univariate_normal_gaussian(shape: tuple) -> torch.Tensor:
    return torch.randn(shape, device=get_device())
