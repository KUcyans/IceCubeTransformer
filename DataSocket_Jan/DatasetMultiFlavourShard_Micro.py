import numpy as np
from torch.utils.data import Dataset

from .EnergyRange import EnergyRange
from .DatasetMonoFlavourShard_Micro import DatasetMonoFlavourShard_Micro
from .MaxNDOMFinder import MaxNDOMFinder

class DatasetMultiFlavourShard_Micro(Dataset):
    def __init__(self, root_dir: str, 
                 energy_band: EnergyRange, 
                 part: int, 
                 shard: int = None,
                 event_length: int = None,
                 verbosity: int = 0,
                 n_classes: int = 3,
                 first_n_events: int = 10,
                 ):
        """
        Args:
            root_dir (str): The root directory of the dataset.
            energy_band (EnergyRange): The energy band (enum) defining the subdirectories.
            part (int): The part number to collect.
            shard (int): The shard number to collect.
            event_length (int): The maximum number of DOMs in the dataset.
            verbosity (int): The verbosity level.
        """
        self.root_dir = root_dir
        self.energy_band = energy_band
        self.part = part
        self.shard = shard
        self.verbosity = verbosity
        self.first_n_events = first_n_events
        self.num_classes = n_classes
        
        if event_length is None:
            event_length_finder = MaxNDOMFinder(root_dir, energy_band, part, shard, verbosity=self.verbosity)
            self.event_length = event_length_finder()
        else:
            self.event_length = event_length
        
        self.datasets = self._collect_shards()
        self.cumulative_lengths = self._compute_cumulative_lengths()
        
        if verbosity > 0:
            self._show_info()
    
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        dataset_idx, local_idx = self._global_to_local_index(idx)
        return self.datasets[dataset_idx][local_idx]
    
    def _collect_shards(self):
        datasets = []
        
        for subdir in self.energy_band.get_subdirs():
            dataset = DatasetMonoFlavourShard_Micro(
                root_dir = self.root_dir, 
                subdirectory_no = int(subdir),
                part = self.part,
                shard = self.shard,
                event_length = self.event_length,
                verbosity = self.verbosity - 1,
                first_n_events = self.first_n_events,
                n_classes = self.num_classes
            )
            datasets.append(dataset)
            
        return datasets
    
    def _global_to_local_index(self, idx):
        for dataset_idx, start in enumerate(self.cumulative_lengths[:-1]):
            if start <= idx < self.cumulative_lengths[dataset_idx + 1]:
                local_idx = idx - start
                return dataset_idx, local_idx
        raise IndexError(f"Index {idx} is out of range.")
    
    def _compute_cumulative_lengths(self):
        lengths = [len(dataset) for dataset in self.datasets]
        return [0] + list(np.cumsum(lengths))
    
    def _show_info(self):
        print(f"------------- Multi-Flavour Shard (Energy Band: {self.energy_band.name}, Part: {self.part}, Shard: {self.shard}) -------------")