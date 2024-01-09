import os
from typing import Dict, List, Tuple

import torch

from src.dataset.datasets.ddb_graph_dataset import GraphDDBDataset
from src.utils.io import load_json


class GraphMedinaDataset(GraphDDBDataset):
    def __init__(self, data_path: str, seed: int) -> None:
        super(GraphMedinaDataset, self).__init__(data_path, seed)

    def load_molecule_data(self, data_path: str) -> Dict:
        return load_json(
            os.path.join(os.path.dirname(data_path), "medina_featurized_molecules.json")
        )

    def get_val_test_train_index(
        self, data_ensemble_id: int
    ) -> Tuple[List[int], List[int], List[int]]:
        col_name = f"Ensemble_{data_ensemble_id}"
        col = self.data[col_name]
        train_idx = self.data[col == "Train"].index
        val_idx = self.data[col == "Valid"].index
        test_idx = self.data[col == "Test"].index
        return list(val_idx), list(test_idx), list(train_idx)

    def __getitem__(self, index):
        datum = self.data.loc[int(index)]
        # pmf data
        log_gamma_exp = datum["log_gamma_exp"]
        solute_idx = datum["solute_idx"]
        solvent_idx = datum["solvent_idx"]
        # graph data
        solute_molecule = self.get_molecule_data(datum["solute_smiles"])
        solvent_molecule = self.get_molecule_data(datum["solvent_smiles"])
        # construct base data
        data = {
            "data": torch.tensor([log_gamma_exp]),
            "solute_idx": torch.tensor([solute_idx]).type(torch.LongTensor),
            "solvent_idx": torch.tensor([solvent_idx]).type(torch.LongTensor),
            # whether the network has ever seen the solute and/or solvent
            # (zero-shot-prediction)
            "solute_zero_shot": torch.tensor([solute_idx in self.solute_idx_zero_shot]),
            "solvent_zero_shot": torch.tensor(
                [solvent_idx in self.solvent_idx_zero_shot]
            ),
        }
        # enrich with molecular data
        data.update({f"solute_{key}": val for key, val in solute_molecule.items()})
        data.update({f"solvent_{key}": val for key, val in solvent_molecule.items()})
        return data
