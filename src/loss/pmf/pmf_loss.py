import argparse
from abc import abstractmethod

import torch

from src.factor.gaussian.prior_gaussian import DiagonalPriorGaussianFactor
from src.loss.metric_collection import MetricCollection
from src.utils.distribution import (
    multivariate_gaussian_log_prob,
    univariate_gaussian_log_prob,
)


class PMFLoss(MetricCollection):
    def __init__(self, args: argparse.Namespace) -> None:
        super(PMFLoss, self).__init__(args)
        # prior model (Gaussian or Cauchy)
        self.diagonal_prior = args.diagonal_prior
        self.sample_kl = args.sample_kl
        # diagonal or block-diagonal factor to use
        self.factor = DiagonalPriorGaussianFactor
        # solutes prior
        shape = (args.number_of_solutes, args.dimensionality_of_embedding)
        self.solutes_prior_factor = self.factor(
            torch.full(shape, 0.0),  # see: Jirasek et al. (2020)
            torch.full(shape, 0.8),  # see: Jirasek et al. (2020)
            self.diagonal_prior,
            "exp",
        )
        # solvents prior
        shape = (args.number_of_solvents, args.dimensionality_of_embedding)
        self.solvents_prior_factor = self.factor(
            torch.full(shape, 0.0),  # see: Jirasek et al. (2020)
            torch.full(shape, 0.8),  # see: Jirasek et al. (2020)
            self.diagonal_prior,
            "exp",
        )
        # data log-likelihood
        self.data_log_likelihood = univariate_gaussian_log_prob
        # log-prior
        self.log_prior = multivariate_gaussian_log_prob
        # number of samples
        self.number_of_samples_for_expectation = args.number_of_samples_for_expectation

    def get_data_likelihood_std(self):
        return self.data_likelihood_std.repeat(
            1, self.number_of_samples_for_expectation
        )

    def update_solutes_prior_factor(self, params: torch.Tensor):
        self.solutes_prior_factor.update_params(params)

    def update_solvents_prior_factor(self, params: torch.Tensor):
        self.solvents_prior_factor.update_params(params)

    @abstractmethod
    def compute_elbo(self, *args):
        raise NotImplementedError
