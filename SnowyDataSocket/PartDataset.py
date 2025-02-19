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

        # Load metadata
        self.metadata = self._load_metadata()
        self.total_events = len(self.metadata)
        self._preload_column_indices()

    def _preload_column_indices(self):
        first_meta = self.metadata[0]  # ✅ Use first event metadata
        self.current_feature_path = first_meta['shard_file']
        self.current_features = pq.read_table(self.current_feature_path)

        offset, n_doms = first_meta['offset'], first_meta['n_doms']
        features_table = self.current_features.slice(offset - n_doms, n_doms).drop(['event_no', 'original_event_no'])

        # ✅ Save column indices once
        self.column_indices = {name: idx for idx, name in enumerate(features_table.column_names)}

    def _load_metadata(self):
        """Load event metadata from the truth file."""
        metadata = []
        if self.current_truth is None:
            self.current_truth = pq.read_table(self.truth_file)
            if self.selection is not None:
                mask = pc.is_in(self.current_truth['event_no'], value_set=pa.array(self.selection))
                self.current_truth = self.current_truth.filter(mask)

        # ✅ Extract columns once for efficiency
        event_no_col = self.current_truth.column('event_no').to_numpy()
        offset_col = self.current_truth.column('offset').to_numpy()
        n_doms_col = self.current_truth.column('N_doms').to_numpy()
        shard_no_col = self.current_truth.column('shard_no').to_numpy()
        pid_col = self.current_truth.column('pid').to_numpy()

        # ✅ Populate metadata using numpy arrays
        for event_no, offset, n_doms, shard_no, pid in zip(event_no_col, offset_col, n_doms_col, shard_no_col, pid_col):
            metadata.append({
                'event_no': event_no,
                'offset': offset,
                'n_doms': n_doms,
                'shard_file': os.path.join(self.feature_dir, f"PMTfied_{shard_no}.parquet"),
                'target': pid
            })

        return metadata

    def __len__(self):
        """Return total number of events."""
        return self.total_events

    def __getitem__(self, idx):
        idx = int(idx)
        event_meta = self.metadata[idx]
        features = self._load_event(event_meta)
        target = self._encode_target(event_meta['target'])
        return features, target

    def _load_event(self, event_meta):
        """Load event features from a shard file."""
        offset, n_doms = event_meta['offset'], event_meta['n_doms']
        if event_meta['shard_file'] != self.current_feature_path:
            self.current_feature_path = event_meta['shard_file']
            self.current_features = pq.read_table(self.current_feature_path)

        features_table = self.current_features.slice(offset, n_doms).drop(['event_no', 'original_event_no'])
        if self.transform:
            features_table = self.transform(features_table)

        # Save column indices once
        if self.column_indices is None:
            self.column_indices = {name: idx for idx, name in enumerate(features_table.column_names)}

        features = np.stack([col.to_numpy() for col in features_table.columns], axis=1)
        return torch.tensor(features, dtype=torch.float32)

    def _encode_target(self, pid):
        """Encode particle ID as a one-hot vector."""
        pid_to_one_hot = {
            12: [1, 0, 0], -12: [1, 0, 0],
            14: [0, 1, 0], -14: [0, 1, 0],
            16: [0, 0, 1], -16: [0, 0, 1]
        }
        return torch.tensor(pid_to_one_hot.get(pid, [0, 0, 0]), dtype=torch.float32)
