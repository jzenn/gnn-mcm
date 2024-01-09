import argparse
from collections import OrderedDict
from itertools import chain
from typing import Callable, Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn

from src.factor.gaussian.variational_gaussian import VariationalGaussianFactor
from src.featurizer.featurizer import Featurizer
from src.utils.nn import get_act


class GraphPriorProbabilisticMatrixFactorization(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        featurizer: Featurizer,
        solutes_graph_model: nn.Module,
        solvents_graph_model: nn.Module,
        pmf_model: nn.Module,
        channels: int,
    ):
        super(GraphPriorProbabilisticMatrixFactorization, self).__init__()
        self.args = args
        self.featurizer = featurizer
        self.solutes_graph_model = solutes_graph_model
        self.solvents_graph_model = solvents_graph_model
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

    def toggle_analyze(self) -> None:
        self.solutes_graph_model.toggle_analyze()
        self.solvents_graph_model.toggle_analyze()

    def set_analyze_function(self, analyze_function: Callable) -> None:
        self.solutes_graph_model.set_analyze_function(analyze_function)
        self.solvents_graph_model.set_analyze_function(analyze_function)

    def forward(
        self, solutes_data: torch.Tensor, solvents_data: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # featurize bonds and atoms
        (
            solutes_bond_types_mapping,
            solutes_x,
            solutes_x_edge,
            solutes_edge_index,
            solutes_node_index,
        ) = self.featurizer(solutes_data)
        (
            solvents_bond_types_mapping,
            solvents_x,
            solvents_x_edge,
            solvents_edge_index,
            solvents_node_index,
        ) = self.featurizer(solvents_data)

        # parameters for solute and solvent
        solutes_prior_params = self.solutes_graph_model(
            x=solutes_x,
            edge_index=solutes_edge_index,
            edge_features=solutes_x_edge,
            edge_label=solutes_bond_types_mapping,
            node_index=solutes_node_index,
        )
        solvents_prior_params = self.solvents_graph_model(
            x=solvents_x,
            edge_index=solvents_edge_index,
            edge_features=solvents_x_edge,
            edge_label=solvents_bond_types_mapping,
            node_index=solvents_node_index,
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

    def load_pmf_model(self, checkpoint: OrderedDict[str, torch.Tensor]):
        self.pmf_model.load_state_dict(checkpoint)
        print("Loaded state_dict of pmf_model.")

    def load_model(self, checkpoint: Dict[str, OrderedDict[str, torch.Tensor]]):
        for model_part, model_part_name in zip(
            [
                self.featurizer,
                self.solutes_graph_model,
                self.solvents_graph_model,
                self.pmf_model,
            ]
            + ([] if self.lin is None else [self.lin]),
            ["featurizer", "solutes_graph_model", "solvents_graph_model", "pmf_model"]
            + ([] if self.lin is None else ["lin"]),
        ):
            model_part.load_state_dict(
                OrderedDict(
                    [
                        (k[k.find(".") + 1 :], v)
                        for k, v in checkpoint["model_state_dict"].items()
                        if k.startswith(f"{model_part_name}.")
                    ]
                )
            )
            print(f"Loaded state_dict of {model_part_name}.")

    def freeze_pmf_model(self) -> None:
        self._freeze_part_of_model(lambda x: "pmf_model" in x, False)

    def un_freeze_pmf_model(self) -> None:
        self._freeze_part_of_model(lambda x: "pmf_model" in x, True)

    def _freeze_part_of_model(
        self, decide_freeze: Callable, requires_grad: bool
    ) -> None:
        for param_name, param_value in self.named_parameters():
            if decide_freeze(param_name):
                param_value.requires_grad_(requires_grad)

    def split_parameters(self) -> List[Iterator[nn.Parameter]]:
        return [
            self.pmf_model.parameters(),
            chain(
                self.solutes_graph_model.parameters(),
                self.solvents_graph_model.parameters(),
                self.featurizer.parameters(),
            )
            if self.lin is None
            else chain(
                self.solutes_graph_model.parameters(),
                self.solvents_graph_model.parameters(),
                self.featurizer.parameters(),
                self.lin.parameters(),
            ),
        ]

    def clip_grad_norm(self, max_grad: float = None):
        def clipper(*args):
            nn.utils.clip_grad_norm_(*args, error_if_nonfinite=True)

        self._clip_grad(clipper, max_grad)

    def clip_grad_value(self, max_grad: float = None):
        def clipper(*args):
            nn.utils.clip_grad_value_(*args)

        self._clip_grad(clipper, max_grad)

    def _clip_grad(self, clipper: Callable, max_grad: float = None) -> None:
        max_grad = self.args.max_grad if max_grad is None else max_grad

        # featurizer
        clipper(self.featurizer.parameters(), max_grad)

        # solutes graph model
        clipper(self.solutes_graph_model.parameters(), max_grad)

        # solvents graph model
        clipper(self.solvents_graph_model.parameters(), max_grad)

        # PMF model
        clipper(self.pmf_model.parameters(), max_grad)

    def get_matrix_point_estimate(self) -> torch.Tensor:
        return self.pmf_model.get_matrix_point_estimate()

    def get_solutes_factor(self) -> VariationalGaussianFactor:
        return self.pmf_model.get_solutes_factor()

    def get_solvents_factor(self) -> VariationalGaussianFactor:
        return self.pmf_model.get_solvents_factor()
