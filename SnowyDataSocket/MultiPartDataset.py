import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .PartDataset import PartDataset
from .EnergyRange import EnergyRange

class MultiPartDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 subdirectory_parts: dict, 
                 sample_weights: list = None, 
                 selection: list = None) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            subdirectory_parts (dict): Dictionary with subdirectory names as keys and part lists as values.
            sample_weights (list, optional): Weights for sampling from each PartDataset.
            selection (list, optional): List of event numbers to include.
        """
        self.root_dir = root_dir
        self.subdirectory_parts = subdirectory_parts
        self.selection = selection

        # Initialise each PartDataset
        self.datasets = [
            PartDataset(
                root_dir=self.root_dir,
                subdirectory_no=int(subdir), 
                part=part, 
                selection=selection)
            for subdir, parts in subdirectory_parts.items() for part in parts
        ]

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets])

        # Define sampling weights
        if sample_weights:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = [1] * len(self.datasets)
        self.sample_weights = np.array(self.sample_weights) / sum(self.sample_weights)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        # 1. Determine the dataset index using the cumulative lengths
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        # 2. Calculate the local index within the selected dataset
        local_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]
        # 3. Return the event from the selected dataset
        return self.datasets[dataset_idx][local_idx]

    @classmethod
    def from_energy_ranges(cls, root_dir, energy_ranges, parts, sample_weights=None, selection=None):
        subdirectory_parts = {subdir: parts for energy_range in energy_ranges for subdir in energy_range.get_subdirs_energy()}
        return cls(root_dir, subdirectory_parts, sample_weights, selection)
