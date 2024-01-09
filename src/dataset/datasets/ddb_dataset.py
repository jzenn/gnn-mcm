from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.experiment import get_device
from src.utils.std import flatten


class DDBDataset(Dataset):
    def __init__(self, data_path: str, seed: int) -> None:
        super(DDBDataset, self).__init__()
        self.seed = seed
        self.data = pd.read_csv(data_path, sep=",")

        # mapping index of dataset to index in DDB database
        self.solute_idx_to_ddb_idx, self.solvent_idx_to_dbb_idx = self.index_to_ddb()

        # data structure for 0-shot prediction
        self.solute_idx_zero_shot = torch.tensor([])
        self.solvent_idx_zero_shot = torch.tensor([])

        # size of the full matrix (#solutes, #solvents)
        self.N, self.M = (
            max(self.data["solute_idx"]) + 1,
            max(self.data["solvent_idx"]) + 1,
        )
        self.matrix = self.create_matrix()

    def get_index(self):
        return torch.tensor(list(self.data.index))

    def __getitem__(self, index) -> Dict[str, Any]:
        datum = self.data.loc[int(index)]

        # datum has attributes
        # - DDB Solute
        # - Name Solute
        # - CAS Solute
        # - DDB Solvent
        # - Name Solvent
        # - CAS Solvent
        # - gamma_exp
        # - gamma_UNIFAC
        # - gamma_MCM
        # - solute_idx
        # - solvent_idx
        # - log_gamma_exp
        # - log_gamma_UNIFAC
        # - log_gamma_MCM

        return {
            "data": torch.tensor([datum["log_gamma_exp"]]),
            "solute_idx": torch.tensor([datum["solute_idx"]]).type(torch.LongTensor),
            "solvent_idx": torch.tensor([datum["solvent_idx"]]).type(torch.LongTensor),
        }

    def __len__(self):
        return len(self.data)

    def get_index_vector_for_dataset(self):
        return self.data["solute_idx"].tolist(), self.data["solvent_idx"].tolist()

    def get_all_indices_of_solutes_and_solvents(
        self, subset_index: torch.Tensor
    ) -> Dict[str, List[int]]:
        return {
            "solutes": [self.data.loc[int(i)]["solute_idx"] for i in subset_index],
            "solvents": [self.data.loc[int(i)]["solvent_idx"] for i in subset_index],
        }

    def create_matrix(self) -> torch.Tensor:
        # all missing entries are filled with (-1)
        matrix = torch.ones(self.N, self.M) * -1
        for _, row in self.data.iterrows():
            matrix[row["solute_idx"], row["solvent_idx"]] = row["log_gamma_exp"]
        return matrix

    def get_matrix(self) -> torch.Tensor:
        return self.matrix

    def get_number_of_rows(self) -> int:
        return self.N

    def get_number_of_cols(self) -> int:
        return self.M

    def get_data_index_of_random_solutes_solvents(
        self, n_solutes: int, n_solvents: int
    ) -> torch.Tensor:
        assert n_solutes is not None and n_solvents is not None
        solutes_generator = torch.Generator().manual_seed(self.seed + 1)
        solvents_generator = torch.Generator().manual_seed(self.seed + 2)
        solutes_perm = torch.randperm(self.N, generator=solutes_generator)
        solvents_perm = torch.randperm(self.M, generator=solvents_generator)

        # get indices
        data_indices = list()
        solutes_subset = solutes_perm[:n_solutes]
        solvents_subset = solvents_perm[:n_solvents]
        for subset, key in zip(
            [solutes_subset, solvents_subset], ["solute_idx", "solvent_idx"]
        ):
            for idx in subset:
                data_indices.append(list(self.data[self.data[key] == idx.item()].index))
        return torch.tensor(flatten(data_indices))

    def index_to_ddb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        solute_idx_to_ddb_idx = list()
        solvent_idx_to_ddb_idx = list()

        # index for solutes
        for _, row in self.data[["solute_idx", "DDB Solute"]].iterrows():
            if row["solute_idx"] not in solute_idx_to_ddb_idx:
                solute_idx_to_ddb_idx.append(row["DDB Solute"])

        # index for solvents
        for _, row in self.data[["solvent_idx", "DDB Solvent"]].iterrows():
            if row["solvent_idx"] not in solvent_idx_to_ddb_idx:
                solvent_idx_to_ddb_idx.append(row["DDB Solvent"])

        return torch.tensor(solute_idx_to_ddb_idx), torch.tensor(solvent_idx_to_ddb_idx)

    def prepare_zero_shot_prediction(
        self, train_indices: torch.Tensor, test_indices: torch.Tensor
    ) -> None:
        solute_solvent_indices_train = self.get_all_indices_of_solutes_and_solvents(
            train_indices
        )
        solute_solvent_indices_test = self.get_all_indices_of_solutes_and_solvents(
            test_indices
        )

        # collect indices not in train
        # matrix entries of (solute, solvent) are one-shot or zero-shot
        test_solute_solvent_indices_not_in_train = {"solutes": [], "solvents": []}
        for s in ["solutes", "solvents"]:
            for idx in solute_solvent_indices_test[s]:
                if idx not in solute_solvent_indices_train[s]:
                    test_solute_solvent_indices_not_in_train[s].append(idx)

        # save
        self.solute_idx_zero_shot = torch.unique(
            torch.tensor(test_solute_solvent_indices_not_in_train["solutes"])
        )
        self.solvent_idx_zero_shot = torch.unique(
            torch.tensor(test_solute_solvent_indices_not_in_train["solvents"])
        )

    def is_zero_shot(self, solutes_idx: torch.Tensor, solvents_idx: torch.Tensor):
        return torch.logical_and(
            self.solute_is_zero_shot(solutes_idx),
            self.solvent_is_zero_shot(solvents_idx),
        )

    def solute_is_zero_shot(self, solutes_idx: torch.Tensor):
        return (
            torch.tensor(
                [idx.item() in self.solute_idx_zero_shot for idx in solutes_idx]
            )
            .view(solutes_idx.shape)
            .to(get_device())
        )

    def solvent_is_zero_shot(self, solvents_idx: torch.Tensor):
        return (
            torch.tensor(
                [idx.item() in self.solvent_idx_zero_shot for idx in solvents_idx]
            )
            .view(solvents_idx.shape)
            .to(get_device())
        )
