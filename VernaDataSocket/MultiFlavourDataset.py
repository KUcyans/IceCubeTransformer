import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .MonoFlavourDataset import MonoFlavourDataset
from Enum.EnergyRange import EnergyRange
from Enum.Flavour import Flavour

class MultiFlavourDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 er: EnergyRange,
                 N_events_nu_e: int,
                 N_events_nu_mu: int,
                 N_events_nu_tau: int,
                 selection=None) -> None:
        """
        Dataset that stacks three MonoFlavourDatasets in a cyclic way.

        Args:
            root_dir (str): Root directory of the dataset.
            N_events_nu_e (int): Number of electron neutrino events.
            N_events_nu_mu (int): Number of muon neutrino events.
            N_events_nu_tau (int): Number of tau neutrino events.
            selection (list, optional): List of event numbers to include.
        """
        self.root_dir = root_dir
        self.selection = selection

        # ✅ Step 1: Load individual MonoFlavourDatasets (each handles its own caching)
        self.nu_e_dataset = MonoFlavourDataset(
            root_dir=root_dir, er=er, flavour=Flavour.E, 
            N_events_monodataset=N_events_nu_e, selection=selection
        )
        self.nu_mu_dataset = MonoFlavourDataset(
            root_dir=root_dir, er=er, flavour=Flavour.MU, 
            N_events_monodataset=N_events_nu_mu, selection=selection
        )
        self.nu_tau_dataset = MonoFlavourDataset(
            root_dir=root_dir, er=er, flavour=Flavour.TAU, 
            N_events_monodataset=N_events_nu_tau, selection=selection
        )

        # ✅ Step 2: Stack the datasets in a cyclic fashion
        self.datasets = [self.nu_e_dataset, self.nu_mu_dataset, self.nu_tau_dataset]
        self.flavour_mapped_indices = self._create_cyclic_index()

    def _create_cyclic_index(self):
        """Creates a global cyclic index mapping events from different datasets."""
        cyclic_indices = []
        dataset_lengths = [len(ds) for ds in self.datasets]
        max_length = max(dataset_lengths)

        # ✅ Interleave events cyclically
        for i in range(max_length):
            for ds_idx, ds in enumerate(self.datasets):
                if i < dataset_lengths[ds_idx]:
                    cyclic_indices.append((ds_idx, i))

        return cyclic_indices

    def __len__(self):
        return len(self.flavour_mapped_indices)

    def __getitem__(self, idx):
        """Retrieve cyclically stacked event based on global index."""
        ds_idx, local_idx = self.flavour_mapped_indices[idx]
        return self.datasets[ds_idx][local_idx]
