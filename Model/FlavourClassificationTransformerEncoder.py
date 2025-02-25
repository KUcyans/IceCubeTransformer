import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from pytorch_lightning import LightningModule

from .EncoderBlock import EncoderBlock
from .BuildingBlocks.Pooling import Pooling
from .BuildingBlocks.OutputProjection import OutputProjection

class FlavourClassificationTransformerEncoder(LightningModule):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_f: int, 
                 num_layers: int, 
                 d_input: int,
                 num_classes: int, 
                 seq_len: int,  
                 attention_type: str,  
                 dropout: float = 0.1, 
                 train_logger=None,
                 profiler=None,
                 optimiser=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.num_layers = num_layers
        self.d_input = d_input

        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_type = attention_type  
        self.seq_len = seq_len

        # Input projection layer
        self.input_projection = nn.Linear(self.d_input, self.d_model)
        self.position_embedding = nn.Embedding(self.seq_len, self.d_model)

        self.train_logger = train_logger
        self.profiler = profiler
        self.optimiser = optimiser

        # Stacked encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(
                d_model=self.d_model, 
                n_heads=self.n_heads, 
                d_f=self.d_f, 
                attention_type=self.attention_type,  
                dropout=self.dropout, 
                layer_idx=i) for i in range(self.num_layers)]
        )

        self.pooling = Pooling(pooling_type="mean")

        # Classification head
        self.classification_output_layer = nn.Sequential(
                nn.Linear(self.d_model, self.num_classes),
            )

        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.test_accuracies = []
        self.training_predictions = []
        self.training_targets = []
        self.mini_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        
    def forward(self, x, target=None, mask=None, event_length=None):
        batch_size, seq_len, input_dim = x.size()

        x = self.input_projection(x).to(x.device)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1) # shape: (batch_size, seq_len, d_model)
        
        x = x + pos_emb

        for encoder in self.encoder_blocks:
            x = encoder(x, event_length = event_length)
        
        batch_size, seq_len, d_model = x.size()
        
        row_indices = torch.arange(seq_len).view(1, -1, 1)  # Shape: (1, seq_dim, 1)
        row_indices = row_indices.expand(batch_size, -1, d_model)

        row_indices = row_indices.to(x.device)
        
        mask = row_indices < event_length.view(-1, 1, 1).to(x.device) # Shape: (batch_size, seq_dim, emb_dim)
        
        x = x.masked_fill(mask==0, 0)
        x = self.pooling(x, mask)
        
        logit = self.classification_output_layer(x)
        net_target = torch.argmax(target, dim=1).unsqueeze(-1).float()  # Convert one-hot to scalar (0, 1, or 2)
        loss = F.mse_loss(logit, net_target)
        self.training_losses.append(loss)

        return loss, logit

    def _calculate_accuracy(self, prob, target):
        predicted_labels = torch.argmax(prob, dim=1)
        true_labels = torch.argmax(target, dim=1)
        accuracy = torch.eq(predicted_labels, true_labels).float().mean()
        return accuracy, predicted_labels, true_labels

    
    def training_step(self, batch, batch_idx):
        x, target, event_length = batch
        loss, logit = self(x, target=target, event_length=event_length)

        accuracy, predicted_labels, true_labels = self._calculate_accuracy(logit, target)
        self.training_accuracies.append(accuracy)

        # Store predictions for sparse monitoring
        if batch_idx % 1000 == 0:
            self.training_predictions.extend(predicted_labels.cpu().tolist())
            self.training_targets.extend(true_labels.cpu().tolist())

            # Print sample predictions
            how_many = 5
            print(f"\nEpoch {self.current_epoch}, Batch {batch_idx}:")
            print(f"train_loss_step={loss.item():.4f}, train_accuracy_step={accuracy.item():.4f}")
            print(f"Predicted (One-hot): {F.one_hot(predicted_labels[:how_many], num_classes=self.num_classes).tolist()}")
            print(f"True (One-hot)     : {target[:how_many].to(torch.int32).tolist()}")
            
            self.train_logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: train_loss_step={loss.item():.4f}, train_accuracy_step={accuracy.item():.4f}")
            self.log("train_accuracy", accuracy, prog_bar=True, on_step=True)

            if len(self.training_predictions) > 0:
                device = self.device
                self.mini_conf_matrix = self.mini_conf_matrix.to(device)
                conf_matrix = self.mini_conf_matrix(
                    torch.tensor(self.training_predictions, device=device),
                    torch.tensor(self.training_targets, device=device)
                )
                diagonal_sum = torch.trace(conf_matrix)
                total_element = torch.sum(conf_matrix)
                diagonal_ratio = diagonal_sum / total_element

                self.log("train_diagonal_sum", diagonal_sum.float(), prog_bar=True, on_step=True)
                self.log("train_diagonal_ratio", diagonal_ratio.float(), prog_bar=True, on_step=True)

                print(f"\nEpoch {self.current_epoch}, Batch {batch_idx}: Mini Confusion Matrix\n")
                print(conf_matrix.cpu().numpy())

            # Clear predictions and targets after logging
            self.training_predictions.clear()
            self.training_targets.clear()

        return loss


    def validation_step(self, batch, batch_idx):
        x, target, event_length = batch
        loss, logit = self(x, target=target, event_length=event_length)

        accuracy, predicted_labels, true_labels = self._calculate_accuracy(logit, target)
        self.validation_accuracies.append(accuracy)
        self.validation_losses.append(loss)

        # Store predictions for sparse monitoring
        if batch_idx % 1000 == 0:
            self.training_predictions.extend(predicted_labels.cpu().tolist())
            self.training_targets.extend(true_labels.cpu().tolist())

            # Print sample predictions
            self.log_dict({"val_loss_step": loss.item(), "val_accuracy_step": accuracy.item()}, prog_bar=True, on_step=True)
            self.train_logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: val_loss_step={loss.item():.4f}, val_accuracy_step={accuracy.item():.4f}")
            self.log("val_accuracy", accuracy, prog_bar=True, on_step=True)

            if len(self.training_predictions) > 0:
                device = self.device
                self.mini_conf_matrix = self.mini_conf_matrix.to(device)
                conf_matrix = self.mini_conf_matrix(
                    torch.tensor(self.training_predictions, device=device),
                    torch.tensor(self.training_targets, device=device)
                )
                diagonal_sum = torch.trace(conf_matrix)
                total_element = torch.sum(conf_matrix)
                diagonal_ratio = diagonal_sum / total_element

                self.log("val_diagonal_sum", diagonal_sum.float(), prog_bar=True, on_step=True)
                self.log("val_diagonal_ratio", diagonal_ratio.float(), prog_bar=True, on_step=True)

                print(f"\nEpoch {self.current_epoch}, Batch {batch_idx}: Validation Mini Confusion Matrix\n")
                print(conf_matrix.cpu().numpy())

            # Clear predictions and targets after logging
            self.training_predictions.clear()
            self.training_targets.clear()

        return loss



    def predict_step(self, batch, batch_idx):
        x, target, event_length = batch
        with torch.no_grad():
            loss, logit = self(x, event_length=event_length)
            probs = F.softmax(logit, dim=1)
        return {"logits": logit, "probs": probs, "target": target}


    def test_step(self, batch, batch_idx):
        x, target, event_length = batch
        loss, prob = self(x, target=target, event_length=event_length)

        accuracy, predicted_labels, true_labels = self._calculate_accuracy(logit, target)
        self.test_accuracies.append(accuracy)
        how_many = 5
        if batch_idx % 1000 == 0:
            self.log_dict({"test_loss_step": loss.item(), "test_accuracy_step": accuracy.item()}, prog_bar=True, on_step=True)
            self.train_logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: test_loss_step={loss.item():.4f}, test_accuracy_step={accuracy.item():.4f}")
            self.train_logger.info(f"Predicted (One-hot): {F.one_hot(predicted_labels[:how_many], num_classes=self.num_classes).tolist()}")
            self.train_logger.info(f"True (One-hot)     : {target[:how_many].to(torch.int32).tolist()}")

        return loss

    def on_train_epoch_start(self):
        self.train_logger.info(f"#################### Training epoch {self.current_epoch} ####################")

    def on_train_epoch_end(self):
        """Compute median and mean training loss and accuracy at the end of each epoch."""
        if self.training_losses:
            median_loss = torch.median(torch.stack(self.training_losses))
            mean_loss = torch.mean(torch.stack(self.training_losses))

            self.log("median_train_loss", median_loss, prog_bar=True, on_epoch=True)
            self.log("mean_train_loss", mean_loss, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: MEDIAN_TRAIN_LOSS={median_loss.item():.4f}, MEAN_TRAIN_LOSS={mean_loss.item():.4f}")
            self.training_losses.clear()

        if self.training_accuracies:
            median_accuracy = torch.median(torch.stack(self.training_accuracies))
            mean_accuracy = torch.mean(torch.stack(self.training_accuracies))

            self.log("median_train_accuracy", median_accuracy, prog_bar=True, on_epoch=True)
            self.log("mean_train_accuracy", mean_accuracy, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: MEDIAN_TRAIN_ACCURACY={median_accuracy.item():.4f}, MEAN_TRAIN_ACCURACY={mean_accuracy.item():.4f}")
            self.training_accuracies.clear()

    def on_validation_epoch_end(self):
        """Compute and log median and mean validation loss and accuracy at the end of each epoch."""
        if self.validation_losses:
            median_val_loss = torch.median(torch.stack(self.validation_losses))
            mean_val_loss = torch.mean(torch.stack(self.validation_losses))

            self.log("median_val_loss_epoch", median_val_loss, prog_bar=True, on_epoch=True)
            self.log("mean_val_loss_epoch", mean_val_loss, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: MEDIAN_VAL_LOSS={median_val_loss.item():.4f}, MEAN_VAL_LOSS={mean_val_loss.item():.4f}")
            self.validation_losses.clear()

        if self.validation_accuracies:
            median_accuracy = torch.median(torch.stack(self.validation_accuracies))
            mean_accuracy = torch.mean(torch.stack(self.validation_accuracies))

            self.log("median_val_accuracy_epoch", median_accuracy, prog_bar=True, on_epoch=True)
            self.log("mean_val_accuracy_epoch", mean_accuracy, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: MEDIAN_VAL_ACCURACY={median_accuracy.item():.4f}, MEAN_VAL_ACCURACY={mean_accuracy.item():.4f}")
            self.validation_accuracies.clear()

    def on_test_epoch_end(self):
        """Compute and log median and mean test accuracy and confusion matrix statistics at the end of each epoch."""
        if self.test_accuracies:
            median_accuracy = torch.median(torch.stack(self.test_accuracies))
            mean_accuracy = torch.mean(torch.stack(self.test_accuracies))

            self.log("median_test_accuracy_epoch", median_accuracy, prog_bar=True, on_epoch=True)
            self.log("mean_test_accuracy_epoch", mean_accuracy, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: MEDIAN_TEST_ACCURACY={median_accuracy.item():.4f}, MEAN_TEST_ACCURACY={mean_accuracy.item():.4f}")

            self.test_accuracies.clear()

        # Compute and log confusion matrix statistics
        if len(self.training_predictions) > 0 and len(self.training_targets) > 0:
            device = self.device
            self.mini_conf_matrix.to(device)

            conf_matrix = self.mini_conf_matrix(
                torch.tensor(self.training_predictions, device=device),
                torch.tensor(self.training_targets, device=device)
            )
            diagonal_sum = torch.trace(conf_matrix).float()
            total_element = torch.sum(conf_matrix).float()
            diagonal_ratio = diagonal_sum / total_element

            self.log("test_diagonal_sum_epoch", diagonal_sum, prog_bar=True, on_epoch=True)
            self.log("test_diagonal_ratio_epoch", diagonal_ratio, prog_bar=True, on_epoch=True)

            self.train_logger.info(f"Epoch {self.current_epoch}: TEST_CONF_MATRIX_DIAGONAL_SUM={diagonal_sum.item():.2f}, TEST_CONF_MATRIX_DIAGONAL_RATIO={diagonal_ratio.item():.4f}")
            self.train_logger.info(f"\nEpoch {self.current_epoch}: Test Confusion Matrix\n{conf_matrix.cpu().numpy()}")

            # ✅ Clear predictions after logging
            self.training_predictions.clear()
            self.training_targets.clear()


    def set_optimiser(self, optimiser):
        self.optimiser = optimiser

    def configure_optimizers(self):
        """Return the optimizer or use a default if not set."""
        if self.optimiser is not None:
            return self.optimiser
        else:
            return torch.optim.AdamW(self.parameters(), lr=1e-3)
