import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import time

from pytorch_lightning import LightningModule

from .EncoderBlock import EncoderBlock
from .BuildingBlocks.Pooling import Pooling
from .BuildingBlocks.OutputProjection import OutputProjection
from Enum.AttentionType import AttentionType
from Enum.PositionalEncodingType import PositionalEncodingType
from Enum.LossType import LossType

import psutil
import os

class FlavourClassificationTransformerEncoder(LightningModule):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_f: int, 
                 num_layers: int, 
                 d_input: int,
                 num_classes: int, 
                 n_output_layers: int,
                 seq_len: int,  
                 loss_type: LossType,
                 attention_type: AttentionType,
                 positional_encoding_type: PositionalEncodingType,
                 dropout: float = 0.01,
                 lr: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.num_layers = num_layers
        self.d_input = d_input
        self.n_output_layers = n_output_layers

        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.loss_type = loss_type
        print(f"The model was told that Loss type is {self.loss_type.description}")        
        self.attention_type = attention_type  
        print(f"The model was told that Attention type is {self.attention_type.description}")
        self.positional_encoding_type = positional_encoding_type
        print(f"The model was told that Positional Encoding type is {self.positional_encoding_type.name}")
        
        self.seq_len = seq_len

        # Input projection layer
        self.input_projection = nn.Linear(self.d_input, self.d_model)
        if self.positional_encoding_type == PositionalEncodingType.ABSOLUTE: 
            self.position_embedding = nn.Embedding(self.seq_len, self.d_model)

        # Stacked encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(
                d_model=self.d_model, 
                n_heads=self.n_heads, 
                d_f=self.d_f, 
                attention_type=self.attention_type, 
                positional_encoding_type=self.positional_encoding_type, 
                dropout=self.dropout, 
                layer_idx=i) for i in range(self.num_layers)]
        )

        self.pooling = Pooling(pooling_type="mean")

        self.classification_output_layer = OutputProjection(d_model=self.d_model, 
                                                  d_f=self.d_f,
                                                  num_classes=self.num_classes, 
                                                  num_layers = self.n_output_layers,
                                                  dropout=self.dropout)
        
    def forward(self, x, target=None, mask=None, event_length=None):
        batch_size, seq_len, input_dim = x.size()
        
        x = self.input_projection(x).to(x.device)
        # x shape: (batch_size, seq_len, d_model)
        # Learned Absolute Positional Encoding
        if self.positional_encoding_type == PositionalEncodingType.ABSOLUTE: 
            # shape: (batch_size, seq_len, d_model)        
            pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_emb
            
        for encoder in self.encoder_blocks:
            x = encoder(x, event_length=event_length)
        
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, -1) < event_length.unsqueeze(1)
        x = x.masked_fill(~mask.unsqueeze(-1), 0) # shape (batch_size, seq_len, d_model)
        # mask shape: (batch_size, seq_len)
        
        x = self.pooling(x, mask)
        # x shape: (batch_size, d_model)
        
        model_output = self.classification_output_layer(x)
        # output shape: (batch_size, num_classes)
        # squeezed output model_output.squeeze() shape:
        
        loss = self.compute_loss(model_output.squeeze(), target.squeeze())

        if torch.isnan(x).any():
            print("Feature stats:", x.min().item(), x.max().item())
            print("⚠️ NaN detected in Transformer Encoder output!")
            raise ValueError("NaN detected before classification layer!")

        return loss, model_output

    def compute_loss(self, output, target):
        loss = None
        if self.loss_type == LossType.CROSSENTROPY:
            # Expect target to be integer class indices
            if target.dim() == 2:  # convert one-hot to class indices
                target = torch.argmax(target, dim=-1)
            if target.dtype != torch.long:
                target = target.long()
            loss = F.cross_entropy(output, target)
        elif self.loss_type == LossType.MSE:
            loss = F.mse_loss(output, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return loss

    def _calculate_accuracy(self, model_output, target):
        """Calculate accuracy given model probabilities and true labels."""
        predicted_labels = torch.argmax(model_output, dim=1).cpu()
        true_labels = torch.argmax(target, dim=1).cpu()
        accuracy = torch.eq(predicted_labels, true_labels).float().mean()
        return accuracy, predicted_labels, true_labels

    def training_step(self, batch, batch_idx):
        assert len(batch) == 3, f"[Training]Batch {batch_idx} has unexpected length: {len(batch)}"
        x, target, event_length = batch
        loss, model_output = self(x, target=target, event_length=event_length)
        accuracy, predicted_labels, true_labels = self._calculate_accuracy(model_output, target)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)
        if self.num_classes == 3:
            target_labels = torch.argmax(target, dim=1)
            tau_indices = (target_labels == 2)
            tau_model_outputs = model_output[tau_indices, 2].detach().cpu()
            self.train_tau_model_outputs.extend(tau_model_outputs.tolist())


        # ✅ Periodic logging for detailed monitoring
        period = max(1, len(self.trainer.train_dataloader) // 3)
        if batch_idx % period == 0:
            print(f"\n[Epoch {self.current_epoch} | Batch {batch_idx}]")
            print(f"Train Loss: {loss.item():.4f} | Train Accuracy: {accuracy.item():.4f} | LR: {current_lr:.6e}")

            # Display predictions for debugging
            softmax_model_output = F.softmax(model_output, dim=1)
            num_samples = min(6, x.size(0))  # Show at most 6 samples

            print("\nmodel_output    \t\t softmax(model_output) \t\t prediction \t target")
            for i in range(num_samples):
                pred_one_hot = [1 if j == predicted_labels[i].item() else 0 for j in range(self.num_classes)]
                true_one_hot = target[i].to(torch.int32).tolist()

                model_output_str = " ".join([f"{score.item():.4f}" for score in model_output[i]])
                softmax_str = " ".join([f"{score.item():.4f}" for score in softmax_model_output[i]])

                print(f"{model_output_str}\t{softmax_str}\t{pred_one_hot}\t{true_one_hot}")

            # ✅ Store predictions if tracking history
            if self.training_predictions is not None and self.training_targets is not None:
                self.training_predictions.extend(predicted_labels.tolist())
                self.training_targets.extend(true_labels.tolist())

        return loss

    def validation_step(self, batch, batch_idx):
        assert len(batch) == 3, f"[Validation]Batch {batch_idx} has unexpected length: {len(batch)}"
        x, target, event_length = batch
        loss, model_output = self(x, target=target, event_length=event_length)
        
        accuracy, predicted_labels, true_labels = self._calculate_accuracy(model_output, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        if self.num_classes == 3:
            target_labels = torch.argmax(target, dim=1)
            tau_indices = (target_labels == 2)
            tau_model_outputs = model_output[tau_indices, 2].detach().cpu()
            self.val_tau_model_outputs.extend(tau_model_outputs.tolist())

                
        period = max(1, len(self.trainer.val_dataloaders) // 3)
        if batch_idx % period == 0:
            print(f"\nValidation: Epoch {self.current_epoch}, Batch {batch_idx}:")
            print(f"Validation Loss: {loss.item():.4f} | Validation Accuracy: {accuracy.item():.4f}")
            softmax_model_output = F.softmax(model_output, dim=1)
            
            how_many = 6
            print("\nmodel_output    \t\t softmax(model_output) \t\t prediction \t target")
            for i in range(min(how_many, x.size(0))):
                pred_one_hot = [1 if j == predicted_labels[i].item() else 0 for j in range(self.num_classes)]
                true_one_hot = target[i].to(torch.int32).tolist()
                
                # Convert model_output to a string with all class scores
                model_output_str = " ".join([f"{score.item():.4f}" for score in model_output[i]])
                softmax_str = " ".join([f"{score.item():.4f}" for score in softmax_model_output[i]])
                print(f"{model_output_str} \t {softmax_str} \t {pred_one_hot} \t {true_one_hot}")

            if self.validation_predictions is not None and self.validation_targets is not None:
                self.validation_predictions.extend(predicted_labels.tolist())
                self.validation_targets.extend(true_labels.tolist())
        
        return loss

    def predict_step(self, batch, batch_idx):
        x, target, event_length = batch
        _, model_outputs = self(x, target=target, event_length=event_length)  # <- fix here
        preds = torch.argmax(model_outputs, dim=-1)

        return {
            "target": target.cpu(),  # fixed key name for consistency
            "pred_class": preds.cpu().numpy(),
            "model_outputs": model_outputs.cpu().numpy(),
        }

    def test_step(self, batch, batch_idx):
        x, target, event_length, analysis = batch
        loss, model_output = self(x, target=target, event_length=event_length)

        period = 5000
        if batch_idx  % period == 0:
            accuracy, predicted_labels, true_labels = self._calculate_accuracy(model_output, target)
            print(f"\nTest: Epoch {self.current_epoch}, Batch {batch_idx}:")
            print(f"test_loss_{str(period)}={loss.item():.4f}, test_accuracy_{str(period)}={accuracy.item():.4f}")
            self.log(f"test_loss_{str(period)}", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"test_accuracy_{str(period)}", accuracy, prog_bar=True, on_step=True, on_epoch=True)

            softmax_model_output = F.softmax(model_output, dim=1)
            how_many = 6
            print("\nmodel_output    \t softmax(model_output) \t prediction \t target")
            for i in range(min(how_many, x.size(0))):
                pred_one_hot = [1 if j == predicted_labels[i].item() else 0 for j in range(self.num_classes)]
                true_one_hot = target[i].to(torch.int32).tolist()
                
                # Convert model_output to a string with all class scores
                model_output_str = " ".join([f"{score.item():.4f}" for score in model_output[i]])
                softmax_str = " ".join([f"{score.item():.4f}" for score in softmax_model_output[i]])
                
                print(f"{model_output_str} \t {softmax_str} \t {pred_one_hot} \t {true_one_hot}")

    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        self.training_predictions = []
        self.training_targets = []
        # self.training_conf_matrix = self.get_confusion_matrix()
        if self.num_classes == 3:
            self.train_tau_model_outputs = []
        
    def on_validation_epoch_start(self):
        self.val_start_time = time.time()
        self.validation_predictions = []
        self.validation_targets = []
        # self.validation_conf_matrix = self.get_confusion_matrix()
        self.nu_tau_model_outputs = []
        if self.num_classes == 3:
            self.val_tau_model_outputs = []
        
    def on_test_epoch_start(self):
        self.test_accuracies = []
        # self.test_conf_matrix = self.get_confusion_matrix()
        self.test_predictions = []
        self.test_targets = []

    def get_confusion_matrix(self):
        task_type = "binary" if self.num_classes == 2 else "multiclass"
        return torchmetrics.ConfusionMatrix(task=task_type, num_classes=self.num_classes)

    def on_train_epoch_end(self):
        # Log confusion matrix and other metrics
        self.log_epoch_end_metrics(stage="train")
        if self.num_classes == 3 and hasattr(self, "train_tau_model_outputs"):
            self._log_tau_metrics(self.train_tau_model_outputs, "train")
            del self.train_tau_model_outputs

    def on_validation_epoch_end(self):
        self.log_epoch_end_metrics(stage="val")
        
        if self.num_classes == 3 and hasattr(self, "val_tau_model_outputs"):
            self._log_tau_metrics(self.val_tau_model_outputs, "val")
            del self.val_tau_model_outputs

    def on_test_epoch_end(self):
        mean_loss = self.trainer.callback_metrics.get("test_loss", None)
        mean_accuracy = self.trainer.callback_metrics.get("test_accuracy", None)

        if mean_loss is not None:
            self.log("mean_test_loss_epoch", mean_loss, prog_bar=True, on_epoch=True)
        if mean_accuracy is not None:
            self.log("mean_test_accuracy_epoch", mean_accuracy, prog_bar=True, on_epoch=True)

        self.log_epoch_end_metrics(stage="test")

    def log_epoch_end_metrics(self, stage="train"):
        """Log confusion matrix and statistics at the end of each epoch."""
        if stage == "train":
            predictions = self.training_predictions
            targets = self.training_targets
            elapsed_time = time.time() - self.train_start_time
            minute = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"\nTraining duration(Epoch {self.current_epoch}): {minute}m {seconds:.2f}s")
            self.log("train_duration", elapsed_time, prog_bar=True, on_epoch=True)
        elif stage == "val":
            predictions = self.validation_predictions
            targets = self.validation_targets
            elapsed_time = time.time() - self.val_start_time
            minute = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"\nValidation duration(Epoch {self.current_epoch}): {minute}m {seconds:.2f}s")
            self.log("val_duration", elapsed_time, prog_bar=True, on_epoch=True)
        elif stage == "test":
            predictions = getattr(self, "test_predictions", [])
            targets = getattr(self, "test_targets", [])
            elapsed_time = time.time() - self.test_start_time
            minute = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"\nTest duration(Epoch {self.current_epoch}): {minute}m {seconds:.2f}s")
        
        # Log confusion matrix and related statistics
        # if len(predictions) > 0 and len(targets) > 0:
        #     self.log_confusion_matrix(predictions, targets, stage=stage)
            
        # 🚽 flush 
        if stage == "train":
            self.training_predictions.clear()
            self.training_targets.clear()
            del self.train_start_time
            # del self.training_conf_matrix
        elif stage == "val":
            self.validation_predictions.clear()
            self.validation_targets.clear()
            del self.val_start_time
            # del self.validation_conf_matrix
        elif stage == "test":
            self.test_predictions.clear()
            self.test_targets.clear()
            del self.test_start_time
            # del self.test_conf_matrix
            
    def _log_tau_metrics(self, tau_model_outputs, prefix):
        if len(tau_model_outputs) > 0:
            model_outputs = torch.tensor(tau_model_outputs)
            threshold = 0.85
            frac_above = (model_outputs > threshold).float().mean().item()
            median = model_outputs.median().item()
            self.log(f"{prefix}_tau_output_085_tau", frac_above, prog_bar=True, on_epoch=True)
            self.log(f"{prefix}_tau_output_median_tau", median, prog_bar=True, on_epoch=True)


    def log_confusion_matrix(self, predictions, targets, stage="train"):
        """Logs the confusion matrix with global normalisation and .3f formatting."""
        device = self.device
        
        # Select the correct confusion matrix
        if stage == "train":
            self.training_conf_matrix = self.training_conf_matrix.to(device)
            conf_matrix = self.training_conf_matrix(
                torch.tensor(predictions, device=device),
                torch.tensor(targets, device=device)
            )
        elif stage == "val":
            self.validation_conf_matrix = self.validation_conf_matrix.to(device)
            conf_matrix = self.validation_conf_matrix(
                torch.tensor(predictions, device=device),
                torch.tensor(targets, device=device)
            )
        elif stage == "test":
            self.test_conf_matrix = self.test_conf_matrix.to(device)
            conf_matrix = self.test_conf_matrix(
                torch.tensor(predictions, device=device),
                torch.tensor(targets, device=device)
            )

        conf_matrix = conf_matrix.float()
        total_samples = conf_matrix.sum()
        conf_matrix_norm = conf_matrix / total_samples  # Global normalisation
        trace_norm = torch.trace(conf_matrix_norm)
        self.log(f"{stage}_trace_norm_epoch", trace_norm, prog_bar=True, on_epoch=True)

        conf_matrix_norm_np = conf_matrix_norm.cpu().numpy()
        formatted_matrix = "\n".join(["\t".join([f"{val:.3f}" for val in row]) for row in conf_matrix_norm_np])

        print(f"\n{stage.capitalize()} Normalised Confusion Matrix")
        print(formatted_matrix)


    def _get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"CPU Memory Usage (RSS): {memory_info.rss / 1024 ** 2:.2f} MB")
        print(f"CPU Memory Usage (VMS): {memory_info.vms / 1024 ** 2:.2f} MB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def set_optimiser(self, optimiser_and_scheduler):
        """Save the optimizer and scheduler with interval & frequency."""
        self.custom_optimizers = [optimiser_and_scheduler["optimizer"]]
        self.custom_schedulers = [optimiser_and_scheduler["lr_scheduler"]["scheduler"]]
        self.custom_interval = optimiser_and_scheduler["lr_scheduler"]["interval"]  # ✅ Store interval
        self.custom_frequency = optimiser_and_scheduler["lr_scheduler"]["frequency"]  # ✅ Store frequency

    def configure_optimizers(self):
        """PyTorch Lightning calls this function to get the optimizer & scheduler."""
        if hasattr(self, "custom_optimizers") and self.custom_optimizers:
            optimizer = self.custom_optimizers[0]
            scheduler = self.custom_schedulers[0]

            config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,  # ✅ Use the stored scheduler
                    "interval": self.custom_interval,  # ✅ Use stored interval
                    "frequency": self.custom_frequency  # ✅ Use stored frequency
                }
            }
            
            return config

        else:
            print("⚠️ Warning: Using default optimizer (AdamW) as none was set.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer