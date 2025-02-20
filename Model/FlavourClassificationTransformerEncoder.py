import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 seq_len: int,  # ✅ Added sequence length
                 attention_type: str,  
                 dropout: float = 0.1, 
                 train_logger=None,
                 profiler=None):
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
        self.position_embedding = nn.Embedding(self.seq_len, self.d_model)  # ✅ Added Positional Embeddings

        self.train_logger = train_logger
        self.profiler = profiler

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
        # self.classification_output_layer = OutputProjection(
        #     input_dim=self.d_model,  
        #     hidden_dim=self.d_model,
        #     output_dim=self.num_classes,
        #     dropout=self.dropout
        # )
        self.classification_output_layer = nn.Linear(self.d_model, self.num_classes)

        self.training_losses = []
        
    def forward(self, x, target=None, mask = None, event_lengths=None):
        batch_size, seq_dim_x, n_features = x.size()
        
        x = self.input_projection(x)  # ✅ Input embedding
        pos_emb = self.position_embedding(torch.arange(seq_dim_x, device=x.device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb  # ✅ Add positional embedding
        
        # ✅ Fix Masking Using `event_lengths`
        mask = None
        if event_lengths is not None and mask is None:
            mask = torch.arange(seq_dim_x, device=x.device).unsqueeze(0).expand(batch_size, -1)
            mask = mask < event_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, seq_dim_x, -1)  # (batch_size, seq_dim, seq_dim)

        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
            
        if mask is not None:
            mask = mask[:, 0, :]  # (batch_size, seq_dim) 

        x = self.pooling(x, mask)
        
        logits = self.classification_output_layer(x)
        
        loss = None
        if target is not None:
            net_target = target[:, :self.num_classes]
            loss = F.mse_loss(input = logits, 
                              target = net_target)
            self.training_losses.append(loss)

        return loss, logits

    def training_step(self, batch, batch_idx):
        x, target, mask, event_lengths = batch
        loss, prob = self(x, target=target, mask=mask, event_lengths=event_lengths)  # ✅ Now both `mask` & `event_lengths` are passed
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: train_loss={loss.item():.4f}")
            self.log("train_loss", loss, prog_bar=True, on_step=True)
            self.log("learning rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True)
            self.train_logger.info(f"Epoch {self.current_epoch}: train_loss={loss.item():.4f}")

        if batch_idx % 1000 == 0:
            print(f"\nPeek at predictions and targets at batch {batch_idx}")
            prob_np = prob[:15].detach().cpu().numpy()
            target_np = target[:15].detach().cpu().numpy()
            print(f"{'Prediction':<30} {'Target':<30}")
            for p, t in zip(prob_np, target_np):
                print(f"[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] [{int(t[0])},{int(t[1])},{int(t[2])}]")

        if self.profiler:
            self.profiler.step()

        return loss

    def predict_step(self, batch, batch_idx):
        x, _, mask, _ = batch
        with torch.no_grad():
            _, probs = self(x, mask=mask)
        return {"probs": probs}

    def validation_step(self, batch, batch_idx):
        x, target, mask, event_lengths = batch
        _, logits = self(x, mask=mask, event_lengths=event_lengths)
        net_target = target[:, :self.num_classes]
        loss = F.mse_loss(logits, net_target)
        self.log("val_loss", loss)
        self.train_logger.info(f"Epoch {self.current_epoch}: val_loss={loss.item():.4f}")
        return loss

    def test_step(self, batch, batch_idx):
        x, target, mask, event_lengths = batch
        _, logits = self(x, mask=mask, event_lengths=event_lengths)
        net_target = target[:, :self.num_classes]
        loss = F.mse_loss(logits, net_target)
        self.log("test_loss", loss)
        self.train_logger.info(f"Epoch {self.current_epoch}: test_loss={loss.item():.4f}")
        return loss
    
    def on_train_epoch_start(self):
        self.train_logger.info(f"#################### Training epoch {self.current_epoch} ####################")

    def on_train_epoch_end(self):
        if self.training_losses:
            median_loss = torch.median(torch.stack(self.training_losses))
            self.train_logger.info(f"Epoch {self.current_epoch}: EPOCH_AVG_TRAIN_LOSS={median_loss.item():.4f}")
            self.log("median_train_loss", median_loss, prog_bar=True, on_epoch=True)
            self.training_losses.clear()
            
    def configure_optimizers(self):
        """Pass the optimizer from the DataModule or return a default optimizer."""
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule.optimizer:
            return self.trainer.datamodule.optimizer
        else:
            return torch.optim.AdamW(self.parameters(), lr=1e-3)
