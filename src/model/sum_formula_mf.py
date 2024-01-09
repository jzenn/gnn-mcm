import argparse
from typing import Callable, Tuple

import torch
import torch.nn as nn

from src.featurizer.featurizer import Featurizer
from src.utils.nn import get_act


class SumFormulaMatrixFactorization(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        featurizer: Featurizer,
        solutes_sum_formula_model: nn.Module,
        solvents_sum_formula_model: nn.Module,
        mf_model: nn.Module,
        channels: int,
    ):
        super(SumFormulaMatrixFactorization, self).__init__()
        self.args = args
        self.featurizer = featurizer
        self.solutes_sum_formula_model = solutes_sum_formula_model
        self.solvents_sum_formula_model = solvents_sum_formula_model
        self.mf_model = mf_model

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
        solutes_embedding = self.solutes_sum_formula_model(solutes_data["embedding"])
        solvents_embedding = self.solvents_sum_formula_model(solvents_data["embedding"])

        # use linear layer
        if self.lin is not None:
            embedding = self.lin(torch.cat([solutes_embedding, solvents_embedding], -1))
            solutes_embedding, solvents_embedding = embedding.split(
                int(self.lin[0].in_features / 2), -1
            )

        # get pmf result
        mf_estimate = self.mf_model(solutes_embedding, solvents_embedding)
        return solutes_embedding, solvents_embedding, mf_estimate

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
        clipper(self.mf_model.parameters(), max_grad)

    def get_matrix_point_estimate(self) -> torch.Tensor:
        return self.mf_model.get_matrix_point_estimate()
