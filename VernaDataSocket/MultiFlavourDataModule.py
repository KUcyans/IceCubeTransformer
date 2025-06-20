import numpy as np
import torch
from .MultiFlavourDataset import MultiFlavourDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Enum.Flavour import Flavour
from Enum.EnergyRange import EnergyRange
from Enum.ClassificationMode import ClassificationMode

class MultiFlavourDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_dir, 
                 er: EnergyRange,
                 N_events_nu_e: int,
                 N_events_nu_mu: int,
                 N_events_nu_tau: int,
                 N_events_noise: int,
                 event_length: int,
                 inference_event_length: int,
                 batch_size: int,
                 num_workers: int, 
                 frac_train: float,
                 frac_val: float,
                 frac_test: float,
                 classification_mode: ClassificationMode = ClassificationMode.MULTIFLAVOUR,
                 root_dir_corsika=None,
                 selection=None, 
                 order_by_this_column="Qtotal"):
        super().__init__()
        self.root_dir = root_dir
        self.er = er
        self.N_events_nu_e = N_events_nu_e
        self.N_events_nu_mu = N_events_nu_mu
        self.N_events_nu_tau = N_events_nu_tau
        self.N_events_noise = N_events_noise
        self.event_length = event_length
        self.inference_event_length = inference_event_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._build_frac(frac_train, frac_val, frac_test)
        self.classification_mode = classification_mode
        self.root_dir_corsika = root_dir_corsika
        self.selection = selection
        self.order_by_this_column = order_by_this_column

        self.dataset = None  # ✅ Store dataset globally and split later

    def setup(self, stage=None):
        """Loads dataset once and splits it into train, validation, and test."""
        if self.dataset is None:
            self.dataset = MultiFlavourDataset(
                root_dir=self.root_dir,
                er=self.er,
                N_events_nu_e=self.N_events_nu_e,
                N_events_nu_mu=self.N_events_nu_mu,
                N_events_nu_tau=self.N_events_nu_tau,
                N_events_noise=self.N_events_noise,
                classification_mode=self.classification_mode,
                root_dir_corsika=self.root_dir_corsika,
                selection=self.selection
            )

            # ✅ Compute split sizes
            total_size = len(self.dataset)
            train_size = int(self.frac_train * total_size)
            val_size = int(self.frac_val * total_size)

            self.train_dataset = torch.utils.data.Subset(self.dataset, range(0, train_size))
            self.val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
            self.test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + val_size, total_size))

            self.remove_duplicate_noise_events()

            # ✅ Get column index dynamically
            first_event = self.train_dataset[0][0]
            self.index_order_by = self._get_order_by_index()
            print(f"Feature Dimension: {first_event.shape[1]}")
    
    def remove_duplicate_noise_events(self):
        """Ensures no duplicate noise events across train/val/test splits."""

        # Gather event_no for each subset
        def get_event_nos(subset):
            event_nos = []
            for i in range(len(subset)):
                _, _, analysis = subset[i]
                event_no = analysis[0]
                event_nos.append(event_no)
            return set(event_nos)

        train_event_nos = get_event_nos(self.train_dataset)
        val_event_nos = get_event_nos(self.val_dataset)
        test_event_nos = get_event_nos(self.test_dataset)

        # Remove overlaps from test dataset
        overlap = (train_event_nos | val_event_nos) & test_event_nos
        if overlap:
            print(f"⚠️ Removing {len(overlap)} duplicate noise events from test set...")

            # Keep only unique test event_nos
            unique_indices = [i for i in range(len(self.test_dataset))
                            if self.test_dataset[i][2][0] not in overlap]

            self.test_dataset = torch.utils.data.Subset(self.dataset, unique_indices)
            print("✅ Duplicate noise events removed from test set.")
            
    def _get_order_by_index(self):
        """Finds the correct column index for ordering."""
        try:
            col_names = self.dataset.datasets[0].current_features.column_names
            return col_names.index(self.order_by_this_column)
        except ValueError:
            raise KeyError(f"Column '{self.order_by_this_column}' not found in feature set.")

    def pad_or_truncate(self, event: torch.Tensor):
        """Pads or truncates events to `event_length` based on sorting by `order_by_this_column`."""
        seq_length = event.size(0)
        event = event[event[:, self.index_order_by].argsort(descending=True)]

        # Truncate if too long
        if seq_length > self.event_length:
            event = event[:self.event_length]
        else:
            padding = torch.zeros((self.event_length - seq_length, event.size(1)))
            event = torch.cat([event, padding], dim=0)

        return event, seq_length
    
    def pad_or_truncate_inference(self, event: torch.Tensor):
        seq_length = event.size(0)
        event = event[event[:, self.index_order_by].argsort(descending=True)]

        if seq_length > self.inference_event_length:
            # event = event[event[:, self.index_order_by].argsort(descending=True)][:self.inference_event_length]
            event = event[:self.inference_event_length]
        else:
            padding = torch.zeros((self.inference_event_length - seq_length, event.size(1)))
            event = torch.cat([event, padding], dim=0)

        return event, seq_length
    
    def train_validate_collate_fn(self, batch):
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        batch_events, event_length = zip(*[self.pad_or_truncate(event) for event in features])
        batch_events = torch.stack(batch_events)
        batch_targets = torch.stack(targets)
        batch_event_length = torch.tensor(event_length, dtype=torch.int64)
        
        return batch_events, batch_targets, batch_event_length

    # def predict_collate_fn(self, batch):
    #     features = [item[0] for item in batch]
    #     targets = [item[1] for item in batch]
    #     batch_events, event_length = zip(*[self.pad_or_truncate(event) for event in features])
    #     batch_events = torch.stack(batch_events)
    #     batch_targets = torch.stack(targets)
    #     batch_event_length = torch.tensor(event_length, dtype=torch.int64)

    #     return batch_events, batch_targets, batch_event_length

    def long_predict_collate_fn(self, batch):
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        batch_events, event_length = zip(*[self.pad_or_truncate_inference(event) for event in features])
        batch_events = torch.stack(batch_events)
        batch_targets = torch.stack(targets)
        batch_event_length = torch.tensor(event_length, dtype=torch.int64)
        return batch_events, batch_targets, batch_event_length


    def _build_frac(self, frac_train, frac_val, frac_test):
        """Builds the fraction for each dataset."""
        total_frac = frac_train + frac_val + frac_test
        self.frac_train = frac_train / total_frac
        self.frac_val = frac_val / total_frac
        self.frac_test = frac_test / total_frac
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.train_validate_collate_fn,
            persistent_workers=True,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.train_validate_collate_fn,
            persistent_workers=True,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.long_predict_collate_fn,
            persistent_workers=False,
            pin_memory=False
        )