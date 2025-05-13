import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .MonoFlavourDataset import MonoFlavourDataset
from .NoiseDataset import NoiseDataset
from Enum.EnergyRange import EnergyRange
from Enum.Flavour import Flavour
from Enum.ClassificationMode import ClassificationMode


class MultiFlavourDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 er: EnergyRange,
                 N_events_nu_e: int,
                 N_events_nu_mu: int,
                 N_events_nu_tau: int,
                 N_events_noise: int,
                 classification_mode: ClassificationMode = ClassificationMode.MULTIFLAVOUR,
                 root_dir_corsika: str = None,
                 selection=None) -> None:
        self.classification_mode = classification_mode
        self.selection = selection
        self.root_dir = root_dir
        self.root_dir_corsika = root_dir_corsika
        
        self.er = er    
        self.N_events_nu_e = N_events_nu_e
        self.N_events_nu_mu = N_events_nu_mu
        self.N_events_nu_tau = N_events_nu_tau
        self.N_events_noise = N_events_noise

        self._build_dataset()

        self.flavour_mapped_indices = self._create_index()
        # [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), ...]
        # where 0, 1, 2 are the dataset indices and 0, 1, ... are the local indices within each dataset
        # This will be used to access the datasets in a round-robin fashion 
        # flavour_mapped_indices[0] = (0, 0) -> first event from the first dataset
        # flavour_mapped_indices[1] = (1, 0) -> first event from the second dataset
        # flavour_mapped_indices[2] = (2, 0) -> first event from the third dataset

    def _build_dataset(self):
        flavour_event_map = {
            Flavour.E: self.N_events_nu_e,
            Flavour.MU: self.N_events_nu_mu,
            Flavour.TAU: self.N_events_nu_tau
        }

        # Determine which flavours to include
        if self.classification_mode == ClassificationMode.TRACK_CASCADE_BINARY:
            selected_flavours = [Flavour.E, Flavour.MU]
        elif self.classification_mode in (ClassificationMode.MULTIFLAVOUR, ClassificationMode.SIGNAL_NOISE_BINARY):
            selected_flavours = [Flavour.E, Flavour.MU, Flavour.TAU]
        else:
            raise ValueError(f"Unsupported classification mode: {self.classification_mode}")

        self.datasets = [
            MonoFlavourDataset(
                root_dir=self.root_dir,
                er=self.er,
                flavour=flavour,
                N_events_monodataset=flavour_event_map[flavour],
                classification_mode=self.classification_mode,
                selection=self.selection
            )
            for flavour in selected_flavours
        ]

        # Add noise dataset only for SIGNAL_NOISE_BINARY
        if self.classification_mode == ClassificationMode.SIGNAL_NOISE_BINARY:
            self.noise_dataset = NoiseDataset(
                root_dir=self.root_dir_corsika,
                N_events_noise=self.N_events_noise,
                selection=self.selection
            )

    def _create_index(self):
        """Interleave indices based on classification mode."""
        if self.classification_mode == ClassificationMode.SIGNAL_NOISE_BINARY:
            return self._interleave_signal_noise()
        else:
            return self._cyclic_interleave()

    def _cyclic_interleave(self):
        """Standard round-robin interleaving."""
        cyclic_indices = []
        dataset_lengths = [len(ds) for ds in self.datasets]
        common_length = min(dataset_lengths)
        for i in range(common_length):
            for ds_idx, ds_len in enumerate(dataset_lengths):
                if i < ds_len:
                    cyclic_indices.append((ds_idx, i))
        return cyclic_indices

    def _interleave_signal_noise(self):
        """Interleaves e, noise, mu, noise, tau, noise..."""
        signal_datasets = self.datasets
        noise_dataset = self.noise_dataset

        signal_lengths = [len(ds) for ds in signal_datasets]
        noise_length = len(noise_dataset)

        # Determine max available balanced interleave steps
        max_steps = min(min(signal_lengths), noise_length // len(signal_datasets))

        interleaved = []
        for i in range(max_steps):
            for j, signal_ds in enumerate(signal_datasets):
                interleaved.append((j, i))                      # signal
                interleaved.append((len(signal_datasets), i * len(signal_datasets) + j))  # corresponding noise

        return interleaved

    def __len__(self):
        return len(self.flavour_mapped_indices)

    def __getitem__(self, idx):
        ds_idx, local_idx = self.flavour_mapped_indices[idx]
        
        if ds_idx < len(self.datasets):
            sample = self.datasets[ds_idx][local_idx]
        else:
            sample = self.noise_dataset[local_idx]

        # event_no = int(sample[2][0])  # first column of analysis_truth
        # print(f"ds_idx: {ds_idx}, local_idx: {local_idx}, event_no: {event_no}")
        
        return sample
