import argparse
from typing import List, Tuple


def compute_out_channels_for_pmf(args: argparse.Namespace) -> Tuple[int, int]:
    if args.model == "No-Prior-MF":
        # Sum-Formula prior, just return the embedding dimensions
        solutes_out_channels = args.dimensionality_of_embedding
        solvents_out_channels = args.dimensionality_of_embedding
    else:
        k = args.dimensionality_of_embedding
        if args.model.startswith("Diagonal") or args.diagonal_prior:
            solutes_out_channels = 2 * k
            solvents_out_channels = 2 * k
        elif args.model.startswith("Block-Diagonal") and not args.diagonal_prior:
            solutes_out_channels = k + k * (k + 1) / 2
            solvents_out_channels = k + k * (k + 1) / 2
        elif "-MF" in args.model:
            solutes_out_channels, solvents_out_channels = k, k
        else:
            raise RuntimeError(f"Unrecognized model {args.model}.")
    return int(solutes_out_channels), int(solvents_out_channels)


def compute_graph_dimensions_for_pmf(
    args: argparse.Namespace,
) -> Tuple[int, int, int, int]:
    # node in-channels and hidden-channels
    hidden_node_channels = args.graph_featurizer_atom_embedding_dim
    in_channels = hidden_node_channels

    # edge in-channels and hidden-channels
    hidden_edge_channels = args.graph_featurizer_bond_embedding_dim
    edge_in_channels = hidden_edge_channels

    return in_channels, edge_in_channels, hidden_node_channels, hidden_edge_channels
