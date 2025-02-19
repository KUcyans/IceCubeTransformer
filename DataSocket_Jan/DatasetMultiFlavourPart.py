import os
import numpy as np
from torch.utils.data import Dataset

from .EnergyRange import EnergyRange
from .DatasetMultiFlavourShard import DatasetMultiFlavourShard
from .DatasetMonoFlavourShard import DatasetMonoFlavourShard
from .MaxNDOMFinder import MaxNDOMFinder

class DatasetMultiFlavourPart(Dataset):
    def __init__(self, root_dir: str, 
                 energy_band: EnergyRange, 
                 part: int, 
                 event_length: int = None,
                 verbosity: int = 0,
                 n_classes: int = 3,
                 ):
        """
        Args:
            root_dir (str): The root directory of the dataset.
            energy_band (EnergyRange): The energy band (enum) defining the subdirectories.
            part (int): The part number to collect.
            verbosity (int): The verbosity level.
        """
        self.root_dir = root_dir
        self.energy_band = energy_band
        self.part = part
        self.verbosity = verbosity
        
        if event_length is None:
            event_length_finder = MaxNDOMFinder(root_dir, energy_band, part, verbosity=verbosity)
            self.event_length = event_length_finder()
        else:
            self.event_length = event_length
        self.n_classes = n_classes
        
        # Collect all shards for the part from each flavour
        self.datasets = self._collect_shards()

        # Compute cumulative lengths for indexing
        self.cumulative_lengths = self._compute_cumulative_lengths()

        if verbosity > 0:
            self._show_info()

    def _collect_shards(self):
        datasets = []
        subdirectories = self.energy_band.get_subdirs()
        common_shards, unique_shards = self._get_common_and_unique_shard_numbers(subdirectories)
        
        for subdir in subdirectories:
            for shard in common_shards:
                datasets.append(
                    DatasetMultiFlavourShard(
                        root_dir=self.root_dir,
                        energy_band=self.energy_band,
                        part=self.part,
                        shard=shard,
                        event_length=self.event_length,
                        verbosity=self.verbosity - 1,
                        n_classes=self.n_classes
                    )
                )
            for shard in unique_shards[subdir]:
                datasets.append(
                    DatasetMonoFlavourShard(
                        root_dir=self.root_dir,
                        subdirectory_no=int(subdir),
                        part=self.part,
                        shard=shard,
                        event_length=self.event_length,
                        verbosity=self.verbosity - 1
                    )
                )
        return datasets
        
    def _get_common_and_unique_shard_numbers(self, subdirectories):
        shard_sets = []
        all_shard_numbers = {}
        
        for subdir in subdirectories:
            shard_dir = os.path.join(self.root_dir, subdir, str(self.part))
            shard_numbers = {
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir(shard_dir) if f.startswith("PMTfied_") and f.endswith(".parquet")
            }
            shard_sets.append(shard_numbers)
            all_shard_numbers[subdir] = shard_numbers

        common_shards = sorted(set.intersection(*shard_sets))

        unique_shards = {
            subdir: sorted(shard_numbers - set(common_shards))
            for subdir, shard_numbers in all_shard_numbers.items()
        }

        return common_shards, unique_shards

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self._global_to_local_index(idx)
        return self.datasets[dataset_idx][local_idx]

    def _compute_cumulative_lengths(self):
        lengths = [len(dataset) for dataset in self.datasets]
        return [0] + list(np.cumsum(lengths))

    def _global_to_local_index(self, idx):
        for dataset_idx, start in enumerate(self.cumulative_lengths[:-1]):
            if start <= idx < self.cumulative_lengths[dataset_idx + 1]:
                local_idx = idx - start
                return dataset_idx, local_idx
        raise IndexError(f"Index {idx} is out of range.")

    def _show_info(self):
        print(f"------------- Multi-Flavour Part (Energy Band: {self.energy_band.name}, Part: {self.part}) -------------")
        for dataset in self.datasets:
            dataset._show_info()
