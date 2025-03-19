import pytorch_lightning as pl
import os
class LocalMinimumCheckpoint(pl.callbacks.Callback):
    def __init__(self, checkpoint_dir, monitor="mean_val_loss_epoch"):
        super().__init__()
        self.monitor = monitor
        self.prev_loss = float("inf")
        self.local_minima = []
        self.checkpoint_dir = checkpoint_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        current_loss = trainer.callback_metrics.get(self.monitor, None)

        if current_loss is None:
            return  # No validation loss found
        
        # Check if it's a local minimum
        if len(self.local_minima) == 0 or (current_loss < self.prev_loss and all(current_loss < lm[1] for lm in self.local_minima)):
            self.local_minima.append((current_epoch, current_loss.item()))
            
            # Save the model at this local minimum
            ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch:03d}_val_loss_{current_loss:.4f}.ckpt")
            trainer.save_checkpoint(ckpt_path)

        self.prev_loss = current_loss