import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.nn import get_act, get_norm, initialize_bias_to_zero


class MLP(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        in_channels: int = None,
        edge_in_channels: int = None,
        hidden_node_channels: int = None,
        hidden_edge_channels: int = None,
        out_channels: int = None,
    ):
        super(MLP, self).__init__()
        assert in_channels is not None
        assert hidden_node_channels is not None
        assert out_channels is not None

        self.num_layers = args.graph_number_of_layers
        self.act = get_act(args, args.graph_activation)
        self.dropout = args.graph_dropout
        self.jk = True if args.graph_jumping_knowledge == "cat" else False

        # residual connections
        self.residual_every_layer = (
            args.graph_residual_every_layer
            if args.graph_residual_every_layer is not None
            else -1
        )

        # network
        net = [nn.Linear(in_channels, hidden_node_channels, bias=args.graph_bias)] + [
            nn.Linear(hidden_node_channels, hidden_node_channels, bias=args.graph_bias)
            for _ in range(self.num_layers - 1)
        ]
        self.net = nn.ModuleList(net)

        # norms
        norms = nn.ModuleList(
            [
                get_norm(args, args.graph_norm, hidden_node_channels, graph=False)
                for _ in range(self.num_layers)
            ]
        )
        self.norms = None if norms[-1] is None else norms

        # lin of BasicGNN (final layer)
        self.lin = nn.Linear(
            self.num_layers * hidden_node_channels if self.jk else hidden_node_channels,
            out_channels,
            bias=args.graph_bias,
        )

        if args.graph_initialize_bias_to_zero:
            initialize_bias_to_zero(self.lin)
            initialize_bias_to_zero(self.net)

    def forward(self, x: torch.Tensor):
        if self.jk:
            xs = list()
        x_id = None
        for i in range(self.num_layers):
            # first network layer has different size than rest of the network
            if (
                i > 0
                and self.residual_every_layer > 0
                and i % self.residual_every_layer == 0
            ):
                x_id = x.clone()
            x = self.net[i](x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # first network layer has different size than rest of the network
            if (
                i > 1
                and self.residual_every_layer > 0
                and (i + 1) % self.residual_every_layer == 0
            ):
                x = x + x_id
            if self.jk:
                xs.append(x)

        x = torch.cat(xs, dim=-1) if self.jk else x  # JumpingKnowledge (concat)
        x = self.lin(x)
        return x
