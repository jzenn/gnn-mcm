import argparse
from typing import Tuple

import torch
import torch.nn as nn

from src.factor.gaussian.variational_gaussian import (
    DiagonalVariationalGaussianFactor,
    VariationalGaussianFactor,
)
from src.model.mf import MatrixFactorization


class GaussianProbabilisticMatrixFactorization(nn.Module):
    def __init__(self, args: argparse.Namespace, block_diagonal: bool = False) -> None:
        super(GaussianProbabilisticMatrixFactorization, self).__init__()
        self.matrix_factorization = MatrixFactorization(args)
        # solutes
        shape = [args.number_of_solutes, args.dimensionality_of_embedding]
        self.solutes_factor = DiagonalVariationalGaussianFactor(shape=shape)
        # solvents
        shape = [args.number_of_solvents, args.dimensionality_of_embedding]
        self.solvents_factor = DiagonalVariationalGaussianFactor(shape=shape)
        # other parameters
        self.dimensionality_of_embedding = args.dimensionality_of_embedding
        self.number_of_samples_for_expectation = args.number_of_samples_for_expectation
        self.number_of_solutes = args.number_of_solutes
        self.number_of_solvents = args.number_of_solvents

    def get_solutes_factor(self) -> VariationalGaussianFactor:
        return self.solutes_factor

    def get_solvents_factor(self) -> VariationalGaussianFactor:
        return self.solvents_factor

    def get_matrix_point_estimate(self) -> torch.Tensor:
        return (
            self.solutes_factor.get_mean().detach().clone()
            @ self.solvents_factor.get_mean().detach().clone().T
        )

    def forward(
        self, solutes_idx: torch.Tensor, solvents_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        solutes_sample = self.solutes_factor.sample_and_get(
            self.number_of_samples_for_expectation, solutes_idx.view(-1)
        )
        solvents_sample = self.solvents_factor.sample_and_get(
            self.number_of_samples_for_expectation, solvents_idx.view(-1)
        )

        return solutes_sample, solvents_sample


class GaussianProbabilisticMatrixFactorizationWithoutBias(
    GaussianProbabilisticMatrixFactorization
):
    def __init__(self, args: argparse.Namespace, block_diagonal: bool = False) -> None:
        super(GaussianProbabilisticMatrixFactorizationWithoutBias, self).__init__(
            args, block_diagonal=block_diagonal
        )

    def forward(
        self, solutes_idx: torch.Tensor, solvents_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        solutes_sample, solvents_sample = super().forward(solutes_idx, solvents_idx)
        # perform "inner product"
        result = self.matrix_factorization(solutes_sample, solvents_sample)
        return result
