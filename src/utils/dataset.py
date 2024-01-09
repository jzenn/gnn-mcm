import argparse
from functools import reduce
from typing import Any, Dict, List, Set, Tuple, Union

import torch
from torch.utils.data import Subset, random_split
from torch.utils.data.dataloader import default_collate

from src.dataset.datasets.ddb_dataset import DDBDataset
from src.utils.experiment import get_full_path
from src.utils.io import dump_json
from src.utils.std import flatten, merge_dicts_of_lists


class Data:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.data}"


class GraphAttribute(Data):
    def __init__(self, data: torch.Tensor) -> None:
        super(GraphAttribute, self).__init__(data)


class AdditionalData(Data):
    def __init__(self, data: Any) -> None:
        super(AdditionalData, self).__init__(data)


class NumberOfNodes(GraphAttribute):
    pass


class NodeEmbedding(GraphAttribute):
    pass


class EdgeEmbedding(GraphAttribute):
    pass


class NodeFeatures(GraphAttribute):
    pass


class NodePositions(GraphAttribute):
    pass


class EdgeFeatures(GraphAttribute):
    pass


class GraphRegressionTargets(GraphAttribute):
    pass


class EdgeIndex(GraphAttribute):
    pass


class NodeIndex(GraphAttribute):
    pass


def split_dataset_randomly(
    args: argparse.Namespace, dataset: torch.utils.data.dataset.Dataset
) -> List[torch.utils.data.dataset.Dataset]:
    # val%test%train%
    split_sizes = [
        int(int(s) / 100 * len(dataset)) for s in args.split_sizes.split("%")[:-1]
    ]
    if sum(split_sizes) != len(dataset):
        split_sizes[-1] = len(dataset) - sum(split_sizes[:-1])
    generator = torch.Generator().manual_seed(args.seed) if args.use_seed else None
    datasets = random_split(dataset, split_sizes, generator=generator)
    # convert internal index to tensor
    for ds in datasets:
        ds.indices = torch.tensor(ds.indices)
    # postprocess datasets (considering solutes index and solvents index)
    datasets = postprocess_dataset(args, datasets)
    return datasets


def split_dataset_zero_shot(
    args: argparse.Namespace, dataset: torch.utils.data.dataset.Dataset
) -> List[torch.utils.data.dataset.Dataset]:
    if not isinstance(dataset, DDBDataset):
        raise NotImplementedError

    # need predict from prior to be able to do zero-shot prediction
    assert (
        (
            args.predict_from_prior
            and (
                args.random_zero_shot_prediction_from_rows_and_cols
                or args.random_zero_shot_prediction_from_matrix
            )
        )
        or (args.model == "No-Prior-MF" and args.graph_model is not None)
        or args.graph_model is None  # just use the zero-shot dataset
    )

    # test and validation index
    generator = torch.Generator().manual_seed(args.seed) if args.use_seed else None
    test_val_index = dataset.get_data_index_of_random_solutes_solvents(
        args.n_solutes_for_zero_shot, args.n_solvents_for_zero_shot
    )
    test_val_index = test_val_index[
        torch.randperm(len(test_val_index), generator=generator)
    ].unique()

    # create split index
    split_index = [
        torch.tensor([idx for idx in dataset.get_index() if idx not in test_val_index])
    ]
    split_sizes = [int(i) for i in args.split_sizes.split("%")[:-1]]
    if len(split_sizes) > 2:
        # create validation dataset based on ratio |val| / (|val| + |test|)
        split_factor = int(
            (1 - split_sizes[-3] / (split_sizes[-3] + split_sizes[-2]))
            * len(test_val_index)
        )
        split_index.append(test_val_index[:split_factor])
        split_index.append(test_val_index[split_factor:])
    else:
        split_index.append(test_val_index)
    split_index = split_index[::-1]

    split_index_lens = [len(spi) for spi in split_index]
    print(
        f"Created datasets of lengths {split_index_lens} "
        f"({[spil / sum(split_index_lens) for spil in split_index_lens]})"
    )

    # create index structures based on indices
    dataset.prepare_zero_shot_prediction(split_index[-1], split_index[-2])

    # create datasets
    datasets = [Subset(dataset, idx) for idx in split_index]
    return datasets


def postprocess_dataset(args, datasets):
    # treat arguments exclusively
    assert args.exclude_solutes_solvents_not_present_in_train is not (
        args.random_zero_shot_prediction_from_rows_and_cols
    )
    # exclude data that is not present in the training dataset
    if args.exclude_solutes_solvents_not_present_in_train:
        datasets = exclude_solutes_solvents_not_present_in_train(args, datasets)
    return datasets


def exclude_solutes_solvents_not_present_in_train(
    args: argparse.Namespace, datasets: List[torch.utils.data.dataset.Dataset]
) -> List[torch.utils.data.dataset.Dataset]:
    train_dataset = datasets[-1]
    # a dataset in datasets is wrapped in Subset()
    val_test_datasets = datasets[:-1]
    split_names = ["val", "test"] if len(val_test_datasets) > 1 else ["test"]
    train_dataset_solutes_solvents_indices = (
        train_dataset.dataset.get_all_indices_of_solutes_and_solvents(
            train_dataset.indices
        )
    )
    for i, (dataset, split_name) in enumerate(zip(val_test_datasets, split_names)):
        if not args.random_zero_shot_prediction_from_rows_and_cols:
            # just delete the solutes and solvents from val and test
            delete = True
            (
                number_of_solutes_excluded,
                idx_of_solutes_excluded,
                number_of_solvents_excluded,
                idx_of_solvents_excluded,
            ) = exclude_solvents_solutes_with_idx_from_subset(
                dataset, train_dataset_solutes_solvents_indices, delete
            )
        else:
            # for 0-shot prediction just look at the validation set
            delete = False
            if (
                split_name == "val"
                and args.move_data_from_val_not_included_in_train_to_test
            ):
                (
                    number_of_solutes_excluded,
                    idx_of_solutes_excluded,
                    number_of_solvents_excluded,
                    idx_of_solvents_excluded,
                ) = exclude_solvents_solutes_with_idx_from_subset(
                    dataset, train_dataset_solutes_solvents_indices, delete
                )
                # add the indices to the test dataset
                test_dataset = val_test_datasets[-1]
                test_dataset.indices = torch.cat(
                    [
                        test_dataset.indices,
                        idx_of_solutes_excluded,
                        idx_of_solvents_excluded,
                    ]
                )
            else:
                number_of_solutes_excluded, number_of_solvents_excluded = 0, 0
                idx_of_solutes_excluded = torch.tensor([])
                idx_of_solvents_excluded = torch.tensor([])
        # save file to indicate how many and which solutes/solvents are excluded
        dump_json(
            {
                "split": split_name,
                "number_of_solutes_excluded": number_of_solutes_excluded,
                "index_of_solutes_excluded": idx_of_solutes_excluded.tolist(),
                "number_of_solvents_excluded": number_of_solvents_excluded,
                "index_of_solvents_excluded": idx_of_solvents_excluded.tolist(),
                "deleted": delete,
            },
            get_full_path(args, f"excluded_solutes_solvents_{split_name}.json"),
        )
    return datasets


def exclude_solvents_solutes_with_idx_from_subset(
    subset: torch.utils.data.dataset.Dataset,
    train_dataset_solutes_solvents_indices: Dict[str, List[int]],
    delete: bool = True,
) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
    excluded_str = "Deleted" if delete else "Moved"
    subset_solutes_solvents_indices = (
        subset.dataset.get_all_indices_of_solutes_and_solvents(subset.indices)
    )
    number_of_solutes_excluded, idx_of_solutes_excluded = exclude_indices_from_subset(
        subset,
        subset_solutes_solvents_indices["solutes"],
        set(train_dataset_solutes_solvents_indices["solutes"]),
        delete=delete,
    )
    print(f"{excluded_str} {number_of_solutes_excluded} solutes from the subset.")
    subset_solutes_solvents_indices = (
        subset.dataset.get_all_indices_of_solutes_and_solvents(subset.indices)
    )
    number_of_solvents_excluded, idx_of_solvents_excluded = exclude_indices_from_subset(
        subset,
        subset_solutes_solvents_indices["solvents"],
        set(train_dataset_solutes_solvents_indices["solvents"]),
        delete=delete,
    )
    print(f"{excluded_str} {number_of_solvents_excluded} solvents from the subset.")
    print("*" * 80)
    return (
        number_of_solutes_excluded,
        idx_of_solutes_excluded,
        number_of_solvents_excluded,
        idx_of_solvents_excluded,
    )


def exclude_indices_from_subset(
    subset: torch.utils.data.dataset.Dataset,
    subset_sol_indices: List[int],
    train_indices: Set[int],
    delete: bool = True,
) -> Tuple[int, torch.Tensor]:
    number_of_indices_to_delete = 0
    indices_to_keep = list()
    indices_to_delete = list()
    for idx, subset_sol_idx in enumerate(subset_sol_indices):
        if subset_sol_idx in train_indices:
            indices_to_keep.append(idx)
        else:
            indices_to_delete.append(subset.indices[idx])
            number_of_indices_to_delete += 1
    if delete:
        subset.indices = subset.indices[torch.tensor(indices_to_keep)]
    return number_of_indices_to_delete, torch.tensor(indices_to_delete)


def collate_graph_batch(
    batch: List[Dict[str, GraphAttribute]], calculate_batch_weights: bool = False
) -> Dict[str, Union[torch.Tensor, str]]:
    merged_batch = reduce(merge_dicts_of_lists, batch, {key: [] for key in batch[0]})
    batch_collate = dict()
    # walk through batch and collate
    for key, value in merged_batch.items():
        if (
            isinstance(value[0], NodeFeatures)
            or isinstance(value[0], NodePositions)
            or isinstance(value[0], EdgeFeatures)
            or isinstance(value[0], GraphRegressionTargets)
        ):
            batch_collate.update({key: torch.vstack([v.data for v in value])})
        elif isinstance(value[0], EdgeIndex):
            edge_indices = list([value[0].data])
            max_node_idx = edge_indices[0].max()
            for i in range(1, len(value)):
                edge_indices.append(value[i].data + (1 + max_node_idx))
                max_node_idx = edge_indices[-1].max()
            batch_collate.update({key: torch.hstack(edge_indices)})
        elif isinstance(value[0], NodeIndex):
            node_indices = [value[i].data + i for i in range(len(value))]
            batch_collate.update({key: torch.hstack(node_indices)})
        elif isinstance(value[0], (NumberOfNodes, NodeEmbedding, EdgeEmbedding)):
            batch_collate.update({key: torch.hstack([v.data for v in value])})
        elif isinstance(value[0], AdditionalData):
            batch_collate.update({key: flatten([v.data for v in value])})
        elif isinstance(value[0], torch.Tensor):
            batch_collate.update({key: torch.vstack(value)})
        else:
            raise RuntimeError(f"No collate specified for type {type(value[0])}")

    # calculate batch weights
    if calculate_batch_weights:
        batch_collate = calculate_per_sample_batch_weights(batch_collate)

    return batch_collate


def calculate_per_sample_batch_weights(
    batch: Dict[str, Union[torch.Tensor, str]]
) -> Dict[str, Union[torch.Tensor, str]]:
    solute_idx = batch["solute_idx"].view(-1)
    solvent_idx = batch["solvent_idx"].view(-1)
    solute_weight = torch.ones_like(solute_idx, dtype=torch.float)
    solvent_weight = torch.ones_like(solvent_idx, dtype=torch.float)

    # compute weights for solutes
    unique_solutes = torch.unique(solute_idx)
    for idx in unique_solutes:
        indexer = solute_idx == idx
        solute_weight[indexer] /= indexer.sum()

    # compute weights for solvents
    unique_solvents = torch.unique(solvent_idx)
    for idx in unique_solvents:
        indexer = solvent_idx == idx
        solvent_weight[indexer] /= indexer.sum()

    # update batch
    batch.update(
        {
            "solute_weight": solute_weight.view(-1, 1),
            "solvent_weight": solvent_weight.view(-1, 1),
            "unique_solutes": torch.tensor([len(unique_solutes)]).view(1, 1),
            "unique_solvents": torch.tensor([len(unique_solvents)]).view(1, 1),
        }
    )
    return batch
