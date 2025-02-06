import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from pytorch_lightning import LightningModule

from Model.EncoderBlock import EncoderBlock
from Model.BuildingBlocks.Pooling import Pooling
from Model.BuildingBlocks.OutputProjection import OutputProjection

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
                 learning_rate=1e-5,
                 profiler=None
                 ):
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
        self.profiler = profiler

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
        self.pooling = Pooling(pooling_type="mean")
        
        # Classification head
        self.classification_output_layer = OutputProjection(
            input_dim=self.d_model,  # ✅ Ensure it matches d_model
            hidden_dim=self.d_model,
            output_dim=self.num_classes,
            dropout=self.dropout
        )   

        # self.classification_output_layer = nn.Linear(self.d_model, self.num_classes)
        
        self.training_losses = []

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Forward Pass"):  # 🔥 Profile Forward Pass
            x = self.input_projection(x)
            for encoder in self.encoder_blocks:
                x = encoder(x, mask)
                
            x = self.pooling(x, mask)
            self.nan_logger.info(f"---------Pooling-----------")
            self.nan_logger.info(f"x hasn't nan: {not torch.isnan(x).any()}")
            
            # x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.classification_output_layer(x)
        self.nan_logger.info(f"---------Classification Head-----------")
        self.nan_logger.info(f"logits hasn't nan: {not torch.isnan(logits).any()}")
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=20, verbose=True
        # )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }
        
    # def on_before_optimizer_step(self, optimizer):
    #     nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    # the training step!
    def training_step(self, batch, batch_idx):
        x, y, mask = batch["features"], batch["target"], batch["mask"] # y is the class index
        
        # 🚨 Skip batch if NaNs detected
        if torch.isnan(x).any() or torch.isnan(mask).any():
            print(f"Skipping batch {batch_idx} due to NaNs")
            return None 

        logits = self(x, mask)
        loss = F.cross_entropy(logits, torch.argmax(y, dim=-1))

        print(f"Batch {batch_idx}: train_loss={loss.item():.4f}")
        self.train_logger.info(f"Epoch {self.current_epoch}: train_loss={loss.item():.4f}")
        if self.profiler:
            self.profiler.step()
        self.training_losses.append(loss.detach())  # 🚨 Use .detach() to prevent storing the computation graph
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        
        # for gradient norm logging
        # self._gradient_logging(loss)
        
        # for learning rate logging
        self._learning_rate_logging()
        
        return loss
        
    # def _gradient_logging(self, loss):
    #     self.manual_backward(loss, retain_graph=False)  # 🚀 Call backward inside
    #     gradient_norms = [p.grad.norm(2).item() for p in self.parameters() if p.grad is not None]

    #     mean_gradient_norm = sum(gradient_norms) / len(gradient_norms)
    #     median_gradient_norm = torch.median(torch.tensor(gradient_norms))  # Convert to tensor
    #     max_gradient_norm = max(gradient_norms)

    #     self.log("mean_gradient_norm", mean_gradient_norm, prog_bar=True, on_step=True)
    #     self.log("median_gradient_norm", median_gradient_norm, prog_bar=True, on_step=True)
    #     self.log("max_gradient_norm", max_gradient_norm, prog_bar=True, on_step=True)

    
    def _learning_rate_logging(self):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True)
        
    def validation_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["target"]
        mask = batch["mask"]

        logits = self(x, mask)
        loss = F.cross_entropy(logits, torch.argmax(y, dim=-1))

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == torch.argmax(y, dim=-1)).float().mean()  # Compare indices

        self.train_logger.info(f"Epoch {self.current_epoch}: val_loss={loss.item():.4f}, val_acc={acc.item() * 100:.2f}%")
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        if self.profiler:
            self.profiler.step()
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["features"]
        y = batch["target"]
        mask = batch["mask"]
        
        logits = self(x, mask)  # Get raw logit s
        loss = F.cross_entropy(logits, torch.argmax(y, dim=-1))
        
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities for evaluation
        preds = torch.argmax(probs, dim=1)
        acc = (preds == torch.argmax(y, dim=-1)).float().mean()  # Compare indices

        self.train_logger.info(f"Epoch {self.current_epoch}: test_loss={loss.item():.4f}, test_acc={acc.item() * 100:.2f}%")
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        
        self.nan_logger.info(f"#################### Training epoch {self.current_epoch} ####################")
        
        self.train_logger.info(f"#################### Training epoch {self.current_epoch} ####################")
        self.train_logger.info(f"Current Learning Rate: {current_lr:.6e}")        
        print(f" learning rate{self.current_epoch}: {current_lr:.6e}")

    def predict_step(self, batch, batch_idx):
        """
        Handles inference on new data.
        - Expects a batch with 'features' and optional 'mask'.
        - Returns raw logits and optionally softmax probabilities.
        """
        x, mask = batch["features"], batch.get("mask", None)

        with torch.no_grad():  # 🚀 Disable gradient tracking for inference
            logits = self(x, mask)  # Forward pass

        probs = F.softmax(logits, dim=-1)  # Convert to probabilities

        return {"logits": logits, "probs": probs}


    def on_train_epoch_end(self):
        if self.training_losses:
            median_loss = torch.median(torch.stack(self.training_losses))
            self.train_logger.info(f"Epoch {self.current_epoch}: EPOCH_AVG_TRAIN_LOSS={median_loss.item():.4f}")
            self.log("median_train_loss", median_loss, prog_bar=True, on_epoch=True)
            self.training_losses.clear()