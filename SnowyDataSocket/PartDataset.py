import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset
from .PseudoNormaliser import PseudoNormaliser

class PartDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 subdirectory_no: int, 
                 part: int, 
                 selection: list = None) -> None:
        """Dataset class to load and process IceCube data."""
        self.root_dir = root_dir
        self.subdirectory_no = subdirectory_no
        self.part = part
        self.truth_file = os.path.join(self.root_dir, f"{self.subdirectory_no}", f"truth_{self.part}.parquet")
        self.feature_dir = os.path.join(self.root_dir, f"{self.subdirectory_no}", f"{self.part}")
        self.transform = PseudoNormaliser()
        self.selection = selection

        # Initialize variables
        self.current_truth = None
        self.current_feature_path = None
        self.column_indices = None

        self.total_events = self._count_events()
        self._preload_column_indices()
        
        
    def _count_events(self):
        """Count the number of events in the truth file without loading everything."""
        truth = pq.read_table(self.truth_file)
        if self.selection is not None:
            mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
            truth = truth.filter(mask)
        return len(truth)


    def __len__(self):
        """Return total number of events."""
        return self.total_events


    def __getitem__(self, idx):
        """Load a single event based on index."""
        idx = int(idx)

        # ✅ Read truth only when needed (mimics old dataset logic)
        if self.current_truth is None:
            self.current_truth = pq.read_table(self.truth_file, memory_map=True)  # ✅ Memory Mapping
            if self.selection is not None:
                mask = pc.is_in(self.current_truth['event_no'], value_set=pa.array(self.selection))
                self.current_truth = self.current_truth.filter(mask)

        # ✅ Extract event details (on-demand using cached truth)
        row = self.current_truth.slice(idx, 1)
        offset = int(row.column('offset')[0].as_py())
        n_doms = int(row.column('N_doms')[0].as_py())
        shard_no = int(row.column('shard_no')[0].as_py())
        pid = int(row.column('pid')[0].as_py())

        # ✅ Build feature path using shard number
        feature_path = os.path.join(self.feature_dir, f"PMTfied_{shard_no}.parquet")

        # ✅ Load feature file only when needed
        if feature_path != self.current_feature_path:
            self.current_feature_path = feature_path
            self.current_features = pq.read_table(feature_path, memory_map=True)

        features_table = self.current_features.slice(offset, n_doms).drop(['event_no', 'original_event_no'])

        # ✅ Vectorised Normaliser with NumPy
        features_np = np.column_stack([np.array(features_table[col]) for col in features_table.column_names])
        features_np = self.transform(features_np, features_table.column_names)
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        target = self._encode_target(pid)

        return features_tensor, target


    def _encode_target(self, pid):
        """Encode particle ID as a one-hot vector."""
        pid_to_one_hot = {
            12: [1, 0, 0], -12: [1, 0, 0],
            14: [0, 1, 0], -14: [0, 1, 0],
            16: [0, 0, 1], -16: [0, 0, 1]
        }
        return torch.tensor(pid_to_one_hot.get(pid, [0, 0, 0]), dtype=torch.float32)


    def _preload_column_indices(self):
        """Load feature columns from the first shard and store their indices."""
        first_shard_path = os.path.join(self.feature_dir, f"PMTfied_1.parquet")
        if os.path.exists(first_shard_path):
            first_shard = pq.read_table(first_shard_path, memory_map=True)
            features_table = first_shard.drop(['event_no', 'original_event_no'])
            self.column_indices = {name: idx for idx, name in enumerate(features_table.column_names)}
            self.column_names = list(self.column_indices.keys())  # ✅ Store names as a list
        else:
            raise FileNotFoundError(f"First shard file not found: {first_shard_path}")

    def get_column_names(self):
        """Return the list of column names in the feature files."""
        return self.column_names
    
    def get_column_indices(self):
        """Return the dictionary of column indices in the feature files."""
        return self.column_indices
