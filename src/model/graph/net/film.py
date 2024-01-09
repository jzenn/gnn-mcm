import argparse
import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.basic_gnn import BasicGNN

from src.model.graph.net.film_conv import FiLMConv
from src.utils.graph import get_aggregator
from src.utils.nn import get_act, get_norm


class FiLM(BasicGNN):
    def __init__(
        self,
        args: argparse.Namespace,
        in_channels: int = None,
        edge_in_channels: int = None,
        hidden_node_channels: int = None,
        hidden_edge_channels: int = None,
        out_channels: int = None,
    ) -> None:
        in_channels = args.graph_in_channels if in_channels is None else in_channels
        hidden_node_channels = (
            args.graph_hidden_channels
            if hidden_node_channels is None
            else hidden_node_channels
        )
        out_channels = args.graph_out_channels if out_channels is None else out_channels

        num_layers = args.graph_number_of_layers
        act = get_act(args, args.graph_activation)
        norm = get_norm(args, args.graph_norm, hidden_node_channels)

        super().__init__(
            in_channels,
            hidden_node_channels,
            num_layers,
            out_channels,
            args.graph_dropout,
            act,
            norm,
            args.graph_jumping_knowledge,
        )

        # residual connections
        self.residual_every_layer = (
            args.graph_residual_every_layer
            if args.graph_residual_every_layer is not None
            else -1
        )

        # aggregator for final prediction given all node values over batch
        self.aggregator = get_aggregator(
            args, args.graph_final_prediction_agg, out_channels
        )

        net = (
            nn.Linear(hidden_node_channels, 2 * hidden_node_channels)
            if args.graph_lin_single_layer
            else nn.Sequential(
                nn.Linear(hidden_node_channels, hidden_node_channels),
                copy.deepcopy(act),
                nn.Linear(hidden_node_channels, 2 * hidden_node_channels),
            )
        )

        # convolutional operators
        for i in range(num_layers):
            self.convs.append(
                FiLMConv(
                    hidden_node_channels,
                    hidden_node_channels,
                    num_relations=args.graph_num_relations,
                    nn=net,
                    act=act,
                    aggr=args.graph_message_agg,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x_edge_label = kwargs["edge_label"]
        # copied from super
        xs: List[torch.Tensor] = []
        x_id = None
        for i in range(self.num_layers):
            if self.residual_every_layer > 0 and i % self.residual_every_layer == 0:
                x_id = x.clone()
            x = self.convs[i](x, edge_index, edge_type=x_edge_label.view(-1))
            if self.norms is not None:
                x = self.norms[i](x, kwargs["node_index"])
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if (
                self.residual_every_layer > 0
                and (i + 1) % self.residual_every_layer == 0
            ):
                x = x + x_id
            if hasattr(self, "jk"):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, "jk") else x
        x = self.lin(x) if hasattr(self, "lin") else x

        x = self.aggregator(x, kwargs["node_index"])
        return x
