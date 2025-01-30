import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from pytorch_lightning import LightningModule

from Model.EncoderBlock import EncoderBlock

class FlavourClassificationTransformerEncoder(LightningModule):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 d_f, 
                 num_layers, 
                 d_input,
                 num_classes, 
                 dropout=0.1, 
                 nan_logger=None,
                 train_logger=None,
                 learning_rate=1e-4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.num_layers = num_layers
        self.d_input = d_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout

        # Input projection layer
        self.input_projection = nn.Linear(self.d_input, self.d_model)
        self.nan_logger = nan_logger
        self.train_logger = train_logger

        # Stacked encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(
                d_model = self.d_model, 
                n_heads = self.n_heads, 
                d_f = self.d_f, 
                dropout = self.dropout, 
                nan_logger = self.nan_logger,
                layer_idx=i) for i in range(self.num_layers)]
        )

        # Classification head
        self.classification_output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x, mask=None):
        # Input projection
        x = self.input_projection(x)

        # Encoder blocks
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)

        # Classification head: Mean pooling across sequence and output logits
        x = x.mean(dim=1)
        logits = self.classification_output_layer(x)
        self.nan_logger.info(f"---------Classification Head-----------")
        self.nan_logger.info(f"logits hasn't nan: {not torch.isnan(logits).any()}")
        return logits

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Reduce LR if validation loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Reduce LR when validation loss stops improving
            }
        }

    def training_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["target"]  # For classification, this is the class index
        mask = batch["mask"]

        logits = self(x, mask)
        loss = F.cross_entropy(logits, y)

        self.train_logger.info(f"Epoch {self.current_epoch}: train_loss={loss.item():.4f}")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["target"]
        mask = batch["mask"]

        logits = self(x, mask)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.train_logger.info(f"Epoch {self.current_epoch}: val_loss={loss.item():.4f}, val_acc={acc.item() * 100:.2f}%")
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["target"]
        mask = batch["mask"]

        logits = self(x, mask)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.train_logger.info(f"Epoch {self.current_epoch}: test_loss={loss.item():.4f}, test_acc={acc.item() * 100:.2f}%")
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def on_train_epoch_start(self):
        self.nan_logger.info(f"####################Training epoch {self.current_epoch}####################")
        self.train_logger.info(f"####################Training epoch {self.current_epoch}####################")