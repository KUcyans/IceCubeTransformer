import torch
from .MultiPartDataset import MultiPartDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class MultiPartDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_dir, 
                 subdirectory_parts_train, 
                 subdirectory_parts_val, 
                 subdirectory_parts_test=None,
                 event_length = 128, 
                 batch_size=64, 
                 num_workers=4, 
                 sample_weights_train=None, 
                 sample_weights_val=None,
                 selection=None, 
                 order_by_this_column="Qtotal"):
        super().__init__()
        self.root_dir = root_dir
        self.subdirectory_parts_train = subdirectory_parts_train
        self.subdirectory_parts_val = subdirectory_parts_val
        self.subdirectory_parts_test = subdirectory_parts_test if subdirectory_parts_test else []
        self.event_length = event_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_weights_train = sample_weights_train
        self.sample_weights_val = sample_weights_val
        self.selection = selection
        self.order_by_this_column = order_by_this_column


    def setup(self, stage=None):
        """Initialises datasets for training, validation, and testing."""
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiPartDataset(
                root_dir=self.root_dir,
                subdirectory_parts=self.subdirectory_parts_train,
                sample_weights=self.sample_weights_train,
                selection=self.selection
            )
            self.val_dataset = MultiPartDataset(
                root_dir=self.root_dir,
                subdirectory_parts=self.subdirectory_parts_val,
                sample_weights=self.sample_weights_val,
                selection=self.selection
            )
            first_event, _ = self.train_dataset[0]
            print("Feature Dimension (Train):", first_event.shape[1])
            self.index_order_by = self.train_dataset.datasets[0].column_indices[self.order_by_this_column]

        elif stage == 'predict':
            self.predict_dataset = MultiPartDataset(
                root_dir=self.root_dir,
                subdirectory_parts=self.subdirectory_parts_test,
                sample_weights=self.sample_weights_val,
                selection=self.selection
            )
            first_event, _ = self.predict_dataset[0]
            self.index_order_by = self.predict_dataset.datasets[0].column_indices[self.order_by_this_column]
            print("Feature Dimension (Predict):", first_event.shape[1])

        elif stage == 'test':
            self.test_dataset = MultiPartDataset(
                root_dir=self.root_dir,
                subdirectory_parts=self.subdirectory_parts_test,
                sample_weights=self.sample_weights_val,
                selection=self.selection
            )
            first_event, _ = self.test_dataset[0]
            self.index_order_by = self.test_dataset.datasets[0].column_indices[self.order_by_this_column]
            print("Feature Dimension (Test):", first_event.shape[1])


    def pad_or_truncate(self, event: torch.Tensor):
        seq_length = event.size(0)

        # Truncate if too long
        if seq_length > self.event_length:
            event = event[event[:, self.index_order_by].argsort(descending=True)][:self.event_length]
        else:
            padding = torch.zeros((self.event_length - seq_length, event.size(1)))
            event = torch.cat([event, padding], dim=0)

        return event, seq_length


    def custom_collate_fn(self, batch):
        """Custom collate function to pad or truncate each event in the batch."""
        # batch returns = featurs, targets, masks, event_length
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Pad or truncate using the specified column name
        padded_events, event_length = zip(*[self.pad_or_truncate(event) for event in features])

        # Stack everything into tensors
        batch_events = torch.stack(padded_events) # (batch_size, seq_length, n_features=32)
        batch_targets = torch.stack(targets)
        batch_event_length = torch.tensor(event_length, dtype=torch.int64)

        return batch_events, batch_targets, batch_event_length

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=True,
            pin_memory=True
        ) # leng(dataloader) = len(dataset) // batch_size + 1

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=True,
            pin_memory=True
        )


    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=True,
            pin_memory=True
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            persistent_workers=True,
            pin_memory=True
        )