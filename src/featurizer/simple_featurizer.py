from typing import Any, Dict

import torch.nn as nn

from src.featurizer.featurizer import Featurizer


class SimpleAtomFeaturizer(Featurizer):
    """embed atomic number"""

    def __init__(self, args, data_path: str):
        super(SimpleAtomFeaturizer, self).__init__(args, data_path)
        # embedding dimensions
        self.atom_embedding_dim = args.graph_featurizer_atom_embedding_dim
        self.bond_embedding_dim = args.graph_featurizer_bond_embedding_dim

        # embeddings
        self.atom_embedding = nn.Embedding(
            len(self.aggregated_atoms_mappings["atomic_number"]),
            self.atom_embedding_dim,
        )

    def forward(self, data: Dict[str, Any]):
        atomic_numbers_mapping = data["atomic_numbers_mapping"]
        bond_types_mapping = data["bond_types_mapping"]
        edge_index = data["edge_index"]
        node_index = data["node_index"]
        # embedding
        atoms_embedding = self.atom_embedding(atomic_numbers_mapping)
        return (
            bond_types_mapping,
            atoms_embedding,
            None,  # bond_embedding not needed, prevent code duplication
            edge_index,
            node_index,
        )
