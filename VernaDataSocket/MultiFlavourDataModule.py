import torch
from .MultiFlavourDataModule import MultiFlavourDataModule
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from ..Enum.Flavour import Flavour
from ..Enum.EnergyRange import EnergyRange

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
        pass