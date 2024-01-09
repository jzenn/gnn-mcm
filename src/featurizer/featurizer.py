import os

import torch.nn as nn

from src.utils.io import load_json


class Featurizer(nn.Module):
    def __init__(self, args, data_path: str):
        super(Featurizer, self).__init__()
        if args.data == "DDB":
            aggregated_atoms_mappings_name = "aggregated_atoms_mappings.json"
            aggregated_bonds_mappings_name = "aggregated_bonds_mappings.json"
        elif args.data == "Medina":
            aggregated_atoms_mappings_name = "medina_aggregated_atoms_mappings.json"
            aggregated_bonds_mappings_name = "medina_aggregated_bonds_mappings.json"
        else:
            raise RuntimeError(f"Unknown data for featurizer {args.data}.")
        self.aggregated_atoms_mappings = load_json(
            os.path.join(os.path.dirname(data_path), aggregated_atoms_mappings_name)
        )
        self.aggregated_bonds_mappings = load_json(
            os.path.join(os.path.dirname(data_path), aggregated_bonds_mappings_name)
        )
