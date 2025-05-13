from pytorch_lightning.callbacks import Callback
import os
from typing import Tuple

class MidEpochCheckpoint(Callback):
    def __init__(self, dirpath: str, 
                 max_epochs: int,
                 window: Tuple[int, int] = None,
                 save_interval: int = 2,
                 filename: str = "{epoch:02d}-mid.ckpt"):
        """
        Saves model checkpoints at intervals during a middle epoch window.

        Args:
            dirpath (str): Directory to save checkpoint files.
            max_epochs (int): Total number of epochs planned for training.
            window (Tuple[int, int], optional): Epoch window (start, end) to restrict saving.
                                                If None, defaults to middle third of training.
            save_interval (int): Save every N-th epoch in the window.
            filename (str): Format string for filename. Must include `{epoch}` or similar.
        """
        self.dirpath = dirpath
        self.max_epochs = max_epochs
        self.save_interval = save_interval
        self.window = window or (max_epochs // 3, 2 * max_epochs // 3)
        self.filename_template = filename

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.window[0] <= epoch <= self.window[1]:
            if (epoch - self.window[0]) % self.save_interval == 0:
                filename = self.filename_template.format(epoch=epoch)
                path = os.path.join(self.dirpath, filename)
                if not os.path.exists(path):  # avoid overwriting
                    trainer.save_checkpoint(path)
