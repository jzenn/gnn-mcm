import argparse
from typing import List

import torch
import torch.nn as nn

from src.arguments import parse_args
from src.dataset.dataset_factory import (
    get_graph_ddb_dataloaders,
    get_graph_medina_dataloaders,
    get_zero_shot_graph_ddb_dataloaders,
)
from src.dataset.datasets.ddb_graph_dataset import GraphDDBDataset
from src.dataset.datasets.medina_graph_dataset import GraphMedinaDataset
from src.featurizer.featurizer import Featurizer
from src.featurizer.simple_featurizer import SimpleAtomFeaturizer
from src.loss.mf_loss import NoPriorMFMSELoss
from src.loss.pmf.pmf_entropy_loss import GaussianPMFVIEntropyLoss
from src.loss.pmf.pmf_kl_loss import GaussianPMFVIKLLoss
from src.model.graph.net.film import FiLM
from src.model.graph_pmf import GraphPriorProbabilisticMatrixFactorization
from src.model.mf import MatrixFactorization
from src.model.mlp import MLP
from src.model.pmf import (
    GaussianProbabilisticMatrixFactorization,
    GaussianProbabilisticMatrixFactorizationWithoutBias,
)
from src.model.sum_formula_mf import SumFormulaMatrixFactorization
from src.model.sum_formula_pmf import SumFormulaPriorProbabilisticMatrixFactorization
from src.trainer.ddb_graph_pmf import GraphPriorPMFDDBTrainer
from src.trainer.ddb_sum_formula_mf import SumFormulaMFDDBTrainer
from src.trainer.ddb_sum_formula_pmf import SumFormulaPriorPMFDDBTrainer
from src.trainer.trainer import Trainer
from src.utils.architecture import (
    compute_graph_dimensions_for_pmf,
    compute_out_channels_for_pmf,
)
from src.utils.experiment import create_experiment, get_device


def main(args: argparse.Namespace) -> None:
    train(args)


def train(args: argparse.Namespace) -> None:
    # set up experiment
    create_experiment(args)

    # build trainer
    trainer = build_trainer(args)

    # train
    trainer.train()


def build_trainer(args: argparse.Namespace) -> Trainer:
    # set up model
    model = get_model(args)
    # set up loss
    loss = get_loss(args)
    # set up dataloaders
    dataloaders = get_dataloaders(args)
    # set up trainer (depending on data + model)
    trainer = get_trainer(args, dataloaders, loss, model)
    return trainer


def get_model(args: argparse.Namespace) -> nn.Module:
    # learned prior
    assert args.graph_model is not None
    solutes_out_channels, solvents_out_channels = compute_out_channels_for_pmf(args)
    graph_dimensions = list(compute_graph_dimensions_for_pmf(args))
    # get sub-parts of model
    mf_model = get_mf_model(args)
    solutes_graph_model = get_graph_model(
        args,
        args.graph_model,
        *graph_dimensions,
        solutes_out_channels,
    )
    solvents_graph_model = get_graph_model(
        args,
        args.graph_model,
        *graph_dimensions,
        solvents_out_channels,
    )
    featurizer = get_featurizer(args)
    # combine
    if args.graph_model == "Sum-Formula":
        if args.model == "No-Prior-MF":
            model = SumFormulaMatrixFactorization(
                args,
                featurizer,
                solutes_graph_model,
                solvents_graph_model,
                mf_model,
                solutes_out_channels + solvents_out_channels,
            ).to(get_device())
        else:
            model = SumFormulaPriorProbabilisticMatrixFactorization(
                args,
                featurizer,
                solutes_graph_model,
                solvents_graph_model,
                mf_model,
                solutes_out_channels + solvents_out_channels,
            ).to(get_device())
    elif "-PMF" in args.model:
        model = GraphPriorProbabilisticMatrixFactorization(
            args,
            featurizer,
            solutes_graph_model,
            solvents_graph_model,
            mf_model,
            solutes_out_channels + solvents_out_channels,
        ).to(get_device())
    else:
        raise RuntimeError(
            f"Invalid combination of model and graph-model "
            f"({args.model}, {args.graph_model})."
        )
    return model


def get_mf_model(args: argparse.Namespace) -> GaussianProbabilisticMatrixFactorization:
    block_diagonal = args.model.startswith("Block-Diagonal")
    if args.model.endswith("Gaussian-PMF-VI"):
        model = GaussianProbabilisticMatrixFactorizationWithoutBias(
            args, block_diagonal=block_diagonal
        ).to(get_device())
    elif (args.model == "No-Prior-MF") and (args.graph_model is not None):
        model = MatrixFactorization(args).to(get_device())
    else:
        raise RuntimeError(f"Unrecognized PMF model {args.model}.")
    return model


def get_graph_model(
    args: argparse.Namespace,
    model_name: str,
    in_channels: int = None,
    edge_in_channels: int = None,
    hidden_node_channels: int = None,
    hidden_edge_channels: int = None,
    out_channels: int = None,
) -> nn.Module:
    if model_name.startswith("FiLM"):
        model = FiLM(
            args,
            in_channels,
            edge_in_channels,
            hidden_node_channels,
            hidden_edge_channels,
            out_channels,
        ).to(get_device())
    # this is no *graph* model but acts in combination with the PMF as if it is
    elif model_name.startswith("Sum-Formula"):
        in_channels = args.number_of_distinct_atoms
        model = MLP(
            args,
            in_channels,
            edge_in_channels,
            hidden_node_channels,
            hidden_edge_channels,
            out_channels,
        ).to(get_device())
    else:
        raise RuntimeError(f"Unrecognized graph model {args.model}.")
    return model


def get_featurizer(args: argparse.Namespace) -> Featurizer:
    if args.graph_featurizer == "simple-atom":
        featurizer = SimpleAtomFeaturizer(args, args.data_path)
    elif args.graph_featurizer == "sum-formula":
        # no featurizer needed for Sum-Formula data
        featurizer = None
    else:
        raise RuntimeError(f"Unrecognized featurizer {args.graph_featurizer}")
    return featurizer


def get_loss(args: argparse.Namespace) -> nn.Module:
    maximize_entropy = args.maximize_entropy
    if args.model.endswith("Gaussian-PMF-VI"):
        if maximize_entropy:
            loss = GaussianPMFVIEntropyLoss(args).to(get_device())
        else:
            loss = GaussianPMFVIKLLoss(args).to(get_device())
    elif args.model == "No-Prior-MF":
        loss = NoPriorMFMSELoss(args).to(get_device())
    else:
        raise RuntimeError(f"Could not find any loss for model {args.model}.")
    return loss


def get_dataset(args: argparse.Namespace) -> torch.utils.data.dataset.Dataset:
    if args.data == "DDB" and args.graph_model is not None:
        dataset = GraphDDBDataset(args.data_path, args.seed)
    elif args.data == "Medina" and args.graph_model is not None:
        dataset = GraphMedinaDataset(args.data_path, args.seed)
    else:
        raise RuntimeError(f"Unrecognized data {args.data}.")
    return dataset


def get_dataloaders(
    args: argparse.Namespace,
) -> List[torch.utils.data.dataloader.DataLoader]:
    if (
        args.data == "DDB"
        and args.graph_model is not None
        and not args.random_zero_shot_prediction_from_rows_and_cols
    ):
        dataloaders = get_graph_ddb_dataloaders(args)
    elif (
        args.data == "Medina"
        and args.graph_model is not None
        and not args.random_zero_shot_prediction_from_rows_and_cols
    ):
        dataloaders = get_graph_medina_dataloaders(args)
    elif (
        args.data == "DDB"
        and args.graph_model is not None
        and args.random_zero_shot_prediction_from_rows_and_cols
    ):
        dataloaders = get_zero_shot_graph_ddb_dataloaders(args)
    else:
        raise RuntimeError(f"Unrecognized data {args.data}.")
    return dataloaders


def get_data_to_summarize(args: argparse.Namespace) -> List[str]:
    if args.data == "DDB":
        keys_to_summarize = [
            "excluded_solutes_solvents_test",
            "test_stats_test_last_model",
            "test_stats_test_best_validation",
            "test_stats_point_estimate_test_last_model",
            "test_stats_point_estimate_test_last_model_prior",
            "test_stats_point_estimate_test_best_validation",
            "test_stats_point_estimate_test_best_validation_prior",
        ]
    else:
        raise RuntimeError(f"Unrecognized data for cross validation {args.data}.")
    return keys_to_summarize


def get_trainer(
    args: argparse.Namespace,
    dataloaders: List[torch.utils.data.dataloader.DataLoader],
    loss: nn.Module,
    model: nn.Module,
) -> Trainer:
    if (args.data == "DDB" or args.data == "Medina") and isinstance(
        model, GraphPriorProbabilisticMatrixFactorization
    ):
        trainer = GraphPriorPMFDDBTrainer(args, model, dataloaders, loss)
    elif (args.data == "DDB" or args.data == "Medina") and isinstance(
        model, SumFormulaPriorProbabilisticMatrixFactorization
    ):
        trainer = SumFormulaPriorPMFDDBTrainer(args, model, dataloaders, loss)
    elif (args.data == "DDB" or args.data == "Medina") and isinstance(
        model, SumFormulaMatrixFactorization
    ):
        trainer = SumFormulaMFDDBTrainer(args, model, dataloaders, loss)
    else:
        raise RuntimeError(
            f"No matching trainer found for "
            f"{args.data}, {model.__class__.__name__}, {loss.__class__.__name__}."
        )
    return trainer


if __name__ == "__main__":
    args = parse_args()
    main(args)
