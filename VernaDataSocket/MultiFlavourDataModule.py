import torch
from .MultiFlavourDataset import MultiFlavourDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Enum.Flavour import Flavour
from Enum.EnergyRange import EnergyRange

class MultiFlavourDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_dir, 
                 er: EnergyRange,
                 N_events_nu_e: int,
                 N_events_nu_mu: int,
                 N_events_nu_tau: int,
                 event_length: int,
                 batch_size: int,
                 num_workers: int, 
                 frac_train: float,
                 frac_val: float,
                 frac_test: float,
                 selection=None, 
                 order_by_this_column="Qtotal"):
        super().__init__()
        self.root_dir = root_dir
        self.er = er
        self.N_events_nu_e = N_events_nu_e
        self.N_events_nu_mu = N_events_nu_mu
        self.N_events_nu_tau = N_events_nu_tau
        self.event_length = event_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._build_frac(frac_train, frac_val, frac_test)
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
                selection=self.selection
            )

            # ✅ Compute split sizes
            total_size = len(self.dataset)
            train_size = int(self.frac_train * total_size)
            val_size = int(self.frac_val * total_size)

            self.train_dataset = torch.utils.data.Subset(self.dataset, range(0, train_size))
            self.val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
            self.test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + val_size, total_size))

            # ✅ Get column index dynamically
            first_event, _ = self.train_dataset[0]
            self.index_order_by = self._get_order_by_index()
            print(f"Feature Dimension: {first_event.shape[1]}")
    
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

        # Truncate if too long
        if seq_length > self.event_length:
            event = event[event[:, self.index_order_by].argsort(descending=True)][:self.event_length]
        else:
            padding = torch.zeros((self.event_length - seq_length, event.size(1)))
            event = torch.cat([event, padding], dim=0)

        return event, seq_length

    def custom_collate_fn(self, batch):
        """Collate function to pad or truncate sequences."""
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Pad or truncate using the specified column name
        padded_events, event_length = zip(*[self.pad_or_truncate(event) for event in features])

        # Stack everything into tensors
        batch_events = torch.stack(padded_events)
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
            collate_fn=self.custom_collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
