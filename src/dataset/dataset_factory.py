import argparse
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset.datasets.ddb_graph_dataset import GraphDDBDataset
from src.dataset.datasets.medina_graph_dataset import GraphMedinaDataset
from src.utils.dataset import (
    collate_graph_batch,
    split_dataset_randomly,
    split_dataset_zero_shot,
)


def collate_fn_with_batch_weights(batch):
    return collate_graph_batch(batch, calculate_batch_weights=True)


def get_zero_shot_graph_ddb_dataloaders(
    args: argparse.Namespace,
) -> List[torch.utils.data.dataloader.DataLoader]:
    dataset = GraphDDBDataset(args.data_path, args.seed)
    dataloaders = split_zero_shot_and_get_dataloaders(
        args, dataset, collate_fn_with_batch_weights
    )
    return dataloaders


def get_graph_ddb_dataloaders(
    args: argparse.Namespace,
) -> List[torch.utils.data.dataloader.DataLoader]:
    dataset = GraphDDBDataset(args.data_path, args.seed)

    dataloaders = split_randomly_and_get_dataloaders(
        args, dataset, collate_fn_with_batch_weights
    )
    return dataloaders


def get_graph_medina_dataloaders(
    args: argparse.Namespace,
) -> List[torch.utils.data.dataloader.DataLoader]:
    dataset = GraphMedinaDataset(args.data_path, args.seed)
    indices = dataset.get_val_test_train_index(args.data_ensemble_id)
    # with Medina data there is multi-shot and zero-shot prediction at the same time
    dataset.prepare_zero_shot_prediction(indices[-1], indices[-2])
    datasets = [Subset(dataset, idx) for idx in indices if len(idx) > 0]
    dataloaders = get_dataloaders(args, collate_fn_with_batch_weights, datasets)
    return dataloaders


def split_randomly_and_get_dataloaders(
    args: argparse.Namespace,
    dataset: torch.utils.data.dataset.Dataset,
    collate_fn: Callable = None,
) -> List[torch.utils.data.dataloader.DataLoader]:
    datasets = (
        split_dataset_randomly(args, dataset) if args.split_dataset else [dataset]
    )
    dataloaders = get_dataloaders(args, collate_fn, datasets)
    return dataloaders


def split_zero_shot_and_get_dataloaders(
    args: argparse.Namespace,
    dataset: torch.utils.data.dataset.Dataset,
    collate_fn: Callable = None,
) -> List[torch.utils.data.dataloader.DataLoader]:
    datasets = (
        split_dataset_zero_shot(args, dataset) if args.split_dataset else [dataset]
    )
    dataloaders = get_dataloaders(args, collate_fn, datasets)
    return dataloaders


def get_dataloaders(
    args: argparse.Namespace,
    collate_fn: Callable,
    datasets: List[torch.utils.data.dataset.Dataset],
) -> List[torch.utils.data.dataloader.DataLoader]:
    dataloaders = [
        DataLoader(
            ds,
            batch_size=len(ds) if args.batch_size == -1 else args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )
        for ds in datasets
    ]
    return dataloaders
