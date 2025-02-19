import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from .EnergyRange import EnergyRange

class PMTfiedDataModule(LightningDataModule):
    def __init__(self, root_dir: str, 
                 energy_band: EnergyRange, 
                 dataset: Dataset, 
                 batch_size: int = 32, 
                 num_workers: int = 8, 
                 split_ratios=(0.8, 0.1, 0.1), 
                 n_classes: int = 3,
                 verbosity=0):
        super().__init__()
        self.root_dir = root_dir
        self.energy_band = energy_band
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.n_classes = n_classes
        self.verbosity = verbosity

    def setup(self, stage=None):
        for i, sample in enumerate(self.dataset):
            if torch.isnan(sample["features"]).any() or torch.isnan(sample["target"]).any():
                print(f"⚠️ NaN found in dataset at index {i}!")
        total_len = len(self.dataset)
        train_len = int(total_len * self.split_ratios[0])
        val_len = int(total_len * self.split_ratios[1])
        test_len = total_len - train_len - val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        targets = [torch.argmax(sample["target"]).item() for sample in self.train_dataset]
        
        class_counts = torch.bincount(torch.tensor(targets), minlength=self.n_classes)
        self.class_weights = 1.0 / class_counts.float()

        if self.verbosity > 0:
            print(f"Dataset split into train ({train_len}), val ({val_len}), and test ({test_len})")
            print(f"Class weights: {self.class_weights}")

    def _collate_fn(self, batch):
        features = [item["features"] for item in batch]
        targets = [item["target"] for item in batch]
        masks = [item["mask"] for item in batch]

        max_seq_length = max(f.shape[0] for f in features)
        padded_features = torch.zeros((len(features), max_seq_length, features[0].shape[1]), dtype=torch.float32)
        padded_masks = torch.zeros((len(masks), max_seq_length), dtype=torch.float32)

        for i, (feature, mask) in enumerate(zip(features, masks)):
            seq_length = feature.shape[0]
            padded_features[i, :seq_length, :] = feature
            padded_masks[i, :seq_length] = mask

        targets = torch.stack(targets)
        if torch.isnan(padded_features).any():
            print("⚠️ NaN detected in padded features!")
        if torch.isnan(targets).any():
            print("⚠️ NaN detected in targets!")
        if torch.isnan(padded_masks).any():
            print("⚠️ NaN detected in masks!")
            
        return {
            "features": padded_features,
            "target": targets,
            "mask": padded_masks,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True, 
                          collate_fn=self._collate_fn, 
                          pin_memory=True,
                          # skip the last batch if it is not full
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True, 
                        #   collate_fn=self._collate_fn, 
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False, 
                        #   collate_fn=self._collate_fn, 
                          pin_memory=True)
