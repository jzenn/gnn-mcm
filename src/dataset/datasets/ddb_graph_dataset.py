import os
from typing import Dict

import torch

from src.dataset.datasets.ddb_dataset import DDBDataset
from src.utils.dataset import (
    EdgeEmbedding,
    EdgeIndex,
    NodeEmbedding,
    NodeIndex,
    NumberOfNodes,
)
from src.utils.io import load_json


class GraphDDBDataset(DDBDataset):
    def __init__(self, data_path: str, seed: int) -> None:
        super(GraphDDBDataset, self).__init__(data_path, seed)
        # molecule data
        self.molecule_data = self.load_molecule_data(data_path)

    def load_molecule_data(self, data_path: str) -> Dict:
        return load_json(
            os.path.join(os.path.dirname(data_path), "featurized_molecules.json")
        )

    def get_molecule_data(self, smiles):
        data = self.molecule_data[smiles]

        # atoms have attributes:
        # - atom_index
        # - atomic_number
        # - atomic_symbol
        # - atomic_degree
        # - atomic_chiral_tag
        # - atomic_explicit_valence
        # - atomic_formal_charge
        # - atomic_implicit_valence
        # - atomic_hybridization_type
        # - atom_is_aromatic
        # - atomic_isotope
        # - atomic_mass
        # - atomic_radical_electrons
        # - atomic_total_valence
        # - num_explicit_hs
        # - num_implicit_hs
        # - atom_is_in_ring

        # bonds have attributes:
        # - bond_dir
        # - bond_type
        # - bond_type_as_double
        # - bond_is_aromatic
        # - bond_is_conjugated
        # - bond_stereo
        # - bond_valence_contrib

        # additional data:
        # - name
        # - chemical_formula
        # - cas_number

        processed_molecule_data = {
            "number_of_nodes": NumberOfNodes(
                torch.tensor([len(data["atoms"]["atom_index"])])
            ),
            "node_index": NodeIndex(
                torch.zeros([len(data["atoms"]["atom_index"])]).type(torch.LongTensor)
            ),
            "edge_index": EdgeIndex(torch.tensor(data["edge_index"])),
        }
        # update with embedding vectors
        processed_molecule_data.update(
            {
                "atoms_simple_embedding": NodeEmbedding(
                    torch.tensor(data["atoms_simple_featurized_vec"])
                ),
                "bonds_simple_embedding": EdgeEmbedding(
                    torch.tensor(data["bonds_simple_featurized_vec"])
                ),
                # fixed size embeddings
                "sum_formula_embedding": torch.tensor([data["sum_formula_vec"]]),
            }
        )
        return processed_molecule_data

    def get_all_molecule_data(self):
        # get molecule data for solutes
        all_solutes_molecule_data = dict()
        solute_group = ["solute_smiles", "solute_idx"]
        for _, (solute_smiles, solute_idx) in (
            self.data.copy()
            .groupby(solute_group, as_index=False)
            .first()[solute_group]
            .sort_values(by="solute_idx")
            .iterrows()
        ):
            all_solutes_molecule_data.update(
                {solute_idx: self.get_molecule_data(solute_smiles)}
            )

        # get molecule data for solvents
        all_solvents_molecule_data = dict()
        solvent_group = ["solvent_smiles", "solvent_idx"]
        for _, (solvent_smiles, solvent_idx) in (
            self.data.copy()
            .groupby(solvent_group, as_index=False)
            .first()[solvent_group]
            .sort_values(by="solvent_idx")
            .iterrows()
        ):
            all_solvents_molecule_data.update(
                {solvent_idx: self.get_molecule_data(solvent_smiles)}
            )

        return all_solutes_molecule_data, all_solvents_molecule_data

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
        # add molecular data
        data.update({f"solute_{key}": val for key, val in solute_molecule.items()})
        data.update({f"solvent_{key}": val for key, val in solvent_molecule.items()})
        return data
