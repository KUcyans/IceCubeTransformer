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

        # ✅ Initialise each PartDataset (vectorised, faster than looping)
        self.datasets = [
            PartDataset(
                root_dir=self.root_dir,
                subdirectory_no=int(subdir), 
                part=part, 
                selection=selection)
            for subdir, parts in subdirectory_parts.items() for part in parts
        ]

        # ✅ Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets])

        # ✅ Define sampling weights
        if sample_weights:
            self.sample_weights = np.array(sample_weights)
        else:
            self.sample_weights = np.ones(len(self.datasets))
        self.sample_weights = self.sample_weights / self.sample_weights.sum()


    def __len__(self):
        """Return total number of events."""
        return sum(len(dataset) for dataset in self.datasets)


    def __getitem__(self, idx):
        """Retrieve item from correct dataset using deterministic mixing and searchsorted for efficiency."""
        idx = int(idx)
        dataset_idx = idx % len(self.datasets)  # ✅ Cycle over datasets sequentially
        
        # ✅ Calculate where this index would land within the combined datasets using searchsorted
        adjusted_idx = idx // len(self.datasets)  # ✅ Distribute indices evenly
        local_idx = adjusted_idx if dataset_idx == 0 else adjusted_idx - self.cumulative_lengths[dataset_idx - 1]

        # ✅ Ensure index wraps around if it's out of range (for smaller datasets)
        local_idx = local_idx % len(self.datasets[dataset_idx])

        return self.datasets[dataset_idx][local_idx]

