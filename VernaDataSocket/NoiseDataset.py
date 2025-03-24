import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset
from .PseudoNormaliser import PseudoNormaliser
from Enum.EnergyRange import EnergyRange
from Enum.Flavour import Flavour

class NoiseDataset(Dataset):
    IDENTIFICATION = ["event_no", "offset", "shard_no", "N_doms"]
    TARGET = ["pid"]
    REQUIRED_COLUMNS = IDENTIFICATION + TARGET

    def __init__(self, 
                 root_dir: str, # CORSIKA
                 N_events_noise: int,
                 selection: list = None) -> None:
        self.root_dir = root_dir
        self.N_events_noise = N_events_noise
        self.subdirectory_no = "0003000-0003999"
        self.truth_file_dir = os.path.join(self.root_dir, f"{self.subdirectory_no}")
        self.transform = PseudoNormaliser()
        self.selection = selection

        self.truth_files = sorted(
            [os.path.join(self.truth_file_dir, f) for f in os.listdir(self.truth_file_dir) 
            if f.startswith("truth_") and f.endswith(".parquet")]
        )
        self.event_index = self._build_event_index()
        self.selected_events = self._select_events()

        self.truth_current = None
        self.truth_next = None
        self.current_truth_file = None
        self.next_truth_file = None

    def _build_event_index(self):
        """Scans all truth files and builds an event index."""
        event_index = []

        def extract_part_number(filepath):
            """Extracts part number from `truth_X.parquet` (only uses filename)."""
            filename = os.path.basename(filepath)  # ✅ Extract just the filename
            parts = filename.split("_")
            if len(parts) < 2 or not parts[1].split(".")[0].isdigit():
                return float("inf")  # Push invalid files to the end of sorting
            return int(parts[1].split(".")[0])

        # ✅ Ensure sorting is based on extracted part numbers
        self.truth_files = sorted(
            [os.path.join(self.truth_file_dir, f) for f in os.listdir(self.truth_file_dir)
            if f.startswith("truth_") and f.endswith(".parquet")],
            key=extract_part_number
        )

        # ✅ Remove incorrectly sorted files (if they exist)
        self.truth_files = [f for f in self.truth_files if extract_part_number(os.path.basename(f)) != float("inf")]
        for truth_file in self.truth_files:
            truth_table = pq.read_table(truth_file, columns=self.REQUIRED_COLUMNS, memory_map=True)

            # ✅ Extract only the required columns
            event_nos = np.array(truth_table.column("event_no"))
            shard_nos = np.array(truth_table.column("shard_no"))
            offsets = np.array(truth_table.column("offset"))
            N_doms = np.array(truth_table.column("N_doms"))

            for row_idx, event_no in enumerate(event_nos):
                event_index.append(
                    (event_no, truth_file, row_idx, shard_nos[row_idx], offsets[row_idx], N_doms[row_idx])
                )

        event_index.sort(key=lambda x: x[0])  # Sort for deterministic access
        return event_index

    def _select_events(self):
        """Selects the first N_events_noise events from the event index."""
        if not self.event_index:
            return []
        return self.event_index[:self.N_events_noise]

    def _load_truth_file(self, truth_file):
        """Loads a truth file and manages cache efficiently."""
        if truth_file == self.current_truth_file:
            return self.truth_current  # ✅ Return cached file

        # ✅ If `truth_next` is ready, move it to `truth_current`
        if self.truth_next is not None and truth_file == self.next_truth_file:
            self.truth_current = self.truth_next
            self.current_truth_file = self.next_truth_file
        else:
            # ✅ Directly load truth file if cache is empty
            self.truth_current = pq.read_table(truth_file, columns=self.REQUIRED_COLUMNS, memory_map=True)
            self.current_truth_file = truth_file

        # ✅ Preload next truth file
        try:
            next_idx = self.truth_files.index(truth_file) + 1
            if next_idx < len(self.truth_files):
                self.next_truth_file = self.truth_files[next_idx]
                self.truth_next = pq.read_table(self.next_truth_file, columns=self.REQUIRED_COLUMNS, memory_map=True)
            else:
                self.next_truth_file = None
                self.truth_next = None
        except Exception:
            self.next_truth_file = None
            self.truth_next = None

        return self.truth_current

    def __len__(self):
        return len(self.selected_events)

    def __getitem__(self, idx):
        """Retrieve the event's features and target."""
        event_no, truth_file, row_idx, shard_no, offset, N_doms = self.selected_events[idx]

        # ✅ Load truth file (only keeping two in memory)
        truth_table = self._load_truth_file(truth_file)
        row = truth_table.slice(row_idx, 1)

        # ✅ Correctly locate the feature file
        part_no = int(os.path.basename(truth_file).split("_")[1].split(".")[0])
        feature_dir = os.path.join(self.truth_file_dir, str(part_no))
        feature_file = os.path.join(feature_dir, f"PMTfied_{shard_no}.parquet")

        # ✅ Load feature file if needed
        if not hasattr(self, 'current_feature_file') or self.current_feature_file != feature_file:
            self.current_feature_file = feature_file
            self.current_features = pq.read_table(feature_file, memory_map=True)

        # ✅ Extract event features
        features_table = self.current_features.slice(offset, N_doms).drop(['event_no', 'original_event_no'])
        features_np = np.column_stack([np.array(features_table[col]) for col in features_table.column_names])
        if np.isnan(features_np).any():
            print(f"⚠️ NaN detected in event {event_no} from file {feature_file}")
            raise ValueError(f"NaN detected in event {event_no}!")
        features_np = self.transform(features_np, features_table.column_names)
        if np.isnan(features_np).any():
            print(f"⚠️ NaN introduced after normalisation! Event: {event_no}")
            raise ValueError(f"NaN introduced in normalisation!")
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        # ✅ Encode target
        target = self._encode_target_signal_noise_binary(row.column("pid")[0].as_py())

        return features_tensor, target

    
    def _encode_target_signal_noise_binary(self, pid):
        pid_to_one_hot = {
            12: [1, 0], -12: [1, 0],
            14: [1, 0], -14: [1, 0],
            16: [1, 0], -16: [1, 0],
        }
        return torch.tensor(pid_to_one_hot.get(pid, [0, 1]), dtype=torch.float32)