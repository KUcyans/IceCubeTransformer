import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset

from .PseudoNormaliser import PseudoNormaliser
from .MaxNDOMFinder import MaxNDOMFinder

class DatasetMonoFlavourShard(Dataset):
    def __init__(self, root_dir: str, 
                 subdirectory_no: int,
                 part: int, 
                 shard: int, 
                 event_length: int,
                 verbosity: int = 0,
                 n_classes: int = 3,
                 ):
        """
        Args:
            root_dir (str): The root directory of the flavour.
            part (int): The part of the dataset.
            shard (int): The shard number.
            verbosity (int): The verbosity level.
        """
        self.root_dir = root_dir
        self.subdirectory_no = subdirectory_no
        self.part = part
        self.shard = shard
        self.verbosity = verbosity
        self.event_length = event_length
        self.transform = PseudoNormaliser()
        self.num_classes = n_classes
 
        self.feature_file = os.path.join(self.root_dir, f"{self.subdirectory_no}", f"{self.part}", f"PMTfied_{self.shard}.parquet")
        self.truth_file = os.path.join(self.root_dir, f"{self.subdirectory_no}", f"truth_{self.part}.parquet")

        self.truth_data = self._load_truth_data()
        self.feature_data = self._load_feature_data()

        if verbosity > 0:
            self._show_info()

    def __len__(self):
        return len(self.truth_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve truth data
        truth_row = self.truth_data.slice(idx, 1)
        event_no = truth_row.column("event_no").to_pylist()[0]
        original_event_no = truth_row.column("original_event_no").to_pylist()[0]
        offset = truth_row.column("offset").to_pylist()[0]
        n_doms = truth_row.column("N_doms").to_pylist()[0]
        flavour = torch.tensor(truth_row.column("flavour").to_numpy()[0], dtype=torch.float32) # [1, 0, 0], [0, 1, 0] or [0, 0, 1]
        """
        the flavour is a one-hot encoded tensor, where the index of the non-zero element corresponds to the flavour of the event
        NuE: [1, 0, 0]
        NuMu: [0, 1, 0]
        NuTau: [0, 0, 1]
        """

        # Extract and pad features
        features = self._extract_features(offset, n_doms)
        features_padded = np.zeros((self.event_length, features.shape[1]), dtype=np.float32)
        features_padded[:features.shape[0], :] = features

        # Create the mask
        mask = np.zeros((self.event_length,), dtype=np.float32)
        mask[:features.shape[0]] = 1.0

        # Convert to tensors
        event_no_tensor = torch.tensor([event_no, original_event_no], dtype=torch.int64)
        features_tensor = torch.tensor(features_padded, dtype=torch.float32)
        flavour = torch.tensor(truth_row.column("flavour").to_numpy()[0], dtype=torch.float32)
        target_tensor = flavour

        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {
            "event_no": event_no_tensor,
            "features": features_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "event_lengths": torch.tensor(self.event_length, dtype=torch.int64),
        }

    def _load_feature_data(self):
        table = pq.read_table(self.feature_file)
        table = self.transform(table)
        return table
    
    def _load_truth_data(self):
        """
        Load and filter the truth data for the specific shard.
        Dynamically create the 'flavour' column if it is missing.
        """
        # Read the truth data
        truth_table = pq.read_table(self.truth_file)

        # Filter rows matching the shard number
        shard_mask = pc.equal(truth_table.column("shard_no").combine_chunks(), self.shard)
        shard_filter = truth_table.filter(shard_mask)

        # Check if 'flavour' column exists; if not, create it
        if 'flavour' not in shard_filter.column_names:
            if 'pid' not in shard_filter.column_names:
                raise ValueError("The truth data is missing both 'flavour' and 'pid' columns. Cannot determine flavours.")

            # Define PID to flavour mapping
            UNKNOWN_FLAVOUR = -1
            
            pid_to_one_hot = {
                12: [1, 0, 0],  # NuE
                -12: [1, 0, 0],  # NuE
                14: [0, 1, 0],  # NuMu
                -14: [0, 1, 0],  # NuMu
                16: [0, 0, 1],  # NuTau
                -16: [0, 0, 1],  # NuTau
            }

            # Create 'flavour' column based on 'pid'
            pid_column = shard_filter.column("pid").combine_chunks().to_numpy()
            flavour_array = [
                pid_to_one_hot.get(pid, UNKNOWN_FLAVOUR) 
                for pid in pid_column
            ]

            # Convert to PyArrow Array and append as 'flavour'
            # flavour_arrow_array = pa.array(flavour_array, type=pa.int64())
            flavour_arrow_array = pa.array(flavour_array, type=pa.list_(pa.int64()))

            shard_filter = shard_filter.append_column("flavour", flavour_arrow_array)

        return shard_filter

    def _extract_features(self, offset, n_rows):
        """
        Extract a specific slice of features based on offset and number of rows using PyArrow.
        """
        features_slice = self.feature_data.slice(offset, n_rows)

        # Drop columns "event_no" and "original_event_no" if present
        columns_to_keep = [
            col for col in features_slice.column_names if col not in ["event_no", "original_event_no"]
        ]
        features_slice = features_slice.select(columns_to_keep)

        # Convert PyArrow table to a NumPy array efficiently
        features_slice = np.stack([features_slice.column(col).to_numpy() for col in columns_to_keep], axis=1)
        return features_slice

    def _show_info(self):
        print(f"------------- Statistics (subdirectory {self.subdirectory_no}, part {self.part}, shard {self.shard}) -------------")
        num_events = len(self.truth_data)
        print(f"Total {num_events} events from shard {self.shard}")
