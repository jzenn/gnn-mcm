import argparse
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn

from src.factor.gaussian.variational_gaussian import VariationalGaussianFactor
from src.featurizer.featurizer import Featurizer
from src.utils.nn import get_act


class SumFormulaPriorProbabilisticMatrixFactorization(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        featurizer: Featurizer,
        solutes_sum_formula_model: nn.Module,
        solvents_sum_formula_model: nn.Module,
        pmf_model: nn.Module,
        channels: int,
    ):
        super(SumFormulaPriorProbabilisticMatrixFactorization, self).__init__()
        self.args = args
        self.featurizer = featurizer
        self.solutes_sum_formula_model = solutes_sum_formula_model
        self.solvents_sum_formula_model = solvents_sum_formula_model
        self.pmf_model = pmf_model

        # linear layer after GNNs
        self.channels = channels
        self.lin = (
            nn.Sequential(nn.Linear(channels, channels))
            if args.graph_pmf_lin_single_layer
            else nn.Sequential(
                nn.Linear(channels, channels),
                get_act(args, args.graph_activation),
                nn.Linear(channels, channels),
            )
        )
        self.lin = self.lin if args.graph_pmf_lin else None

    def forward(
        self, solutes_data: torch.Tensor, solvents_data: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # parameters for solute and solvent
        solutes_prior_params = self.solutes_sum_formula_model(solutes_data["embedding"])
        solvents_prior_params = self.solvents_sum_formula_model(
            solvents_data["embedding"]
        )

        # use linear layer
        if self.lin is not None:
            embedding = self.lin(
                torch.cat([solutes_prior_params, solvents_prior_params], -1)
            )
            solutes_prior_params, solvents_prior_params = embedding.split(
                int(self.lin[0].in_features / 2), -1
            )

        # get pmf result
        pmf_estimate = self.pmf_model(solutes_data["idx"], solvents_data["idx"])
        return solutes_prior_params, solvents_prior_params, pmf_estimate

    def get_solutes_factor(self) -> VariationalGaussianFactor:
        return self.pmf_model.get_solutes_factor()

    def get_solvents_factor(self) -> VariationalGaussianFactor:
        return self.pmf_model.get_solvents_factor()

    def clip_grad_value(self, max_grad: float = None):
        def clipper(*args):
            nn.utils.clip_grad_value_(*args)

        self._clip_grad(clipper, max_grad)

    def _clip_grad(self, clipper: Callable, max_grad: float = None):
        max_grad = self.args.max_grad if max_grad is None else max_grad

        # solutes graph model
        clipper(self.solutes_sum_formula_model.parameters(), max_grad)

        # solvents graph model
        clipper(self.solvents_sum_formula_model.parameters(), max_grad)

        # PMF model
        clipper(self.pmf_model.parameters(), max_grad)

    def get_matrix_point_estimate(self) -> torch.Tensor:
        return self.pmf_model.get_matrix_point_estimate()
