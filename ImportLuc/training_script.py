print("Importing libraries")
import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-cluster'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")

import torch
import pandas as pd
import numpy as np

#from graphnet.data.dataloader import DataLoader
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from dataset_multifile_flavours import PMTfiedDatasetPyArrow
from dataloader import custom_collate_fn
from model import regression_Transformer
from model import LitModel

import json
with open('config.json') as f:
    config = json.load(f)

#==================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    device = torch.device('cuda:0')

print("All is imported")

checkpoint_path = None

# WandB configuration logging
project_name = f"[20250220] Existing Flavour Classification"
wandb_logger = WandbLogger(project=project_name, log_model=True)
wandb_logger.log_hyperparams(config)
tensorboard_logger = TensorBoardLogger(save_dir='lightning_logs', name='')

# train_path_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22011/truth_1.parquet"]
# train_path_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22014/truth_1.parquet"]
# train_path_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22017/truth_1.parquet"]

train_path_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22011/truth_1.parquet"]
train_path_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22014/truth_1.parquet"]
train_path_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22017/truth_1.parquet"]

# val_path_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22011/truth_2.parquet"]
# val_path_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22014/truth_2.parquet"]
# val_path_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22017/truth_2.parquet"]

val_path_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22011/truth_2.parquet"]
val_path_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22014/truth_2.parquet"]
val_path_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22017/truth_2.parquet"]

#==================================================================================================
train_set_total = PMTfiedDatasetPyArrow(
    truth_paths_1=train_path_1,
    truth_paths_2=train_path_2,
    truth_paths_3=train_path_3,
    sample_weights=[1,1,1],
    selection=None,
    )

val_set_total = PMTfiedDatasetPyArrow(
    truth_paths_1=val_path_1,
    truth_paths_2=val_path_2,
    truth_paths_3=val_path_3,
    sample_weights=[1,1,1],
    selection=None,
    )


# Print the number of events in the training and validation sets
print(f"Number of events in the training set: {len(train_set_total)}")
print(f"Number of events in the validation set: {len(val_set_total)}")

#==================================================================================================
# Define the data loaders

train_dataloader = DataLoader(
    dataset=train_set_total, 
    collate_fn = custom_collate_fn, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'],
    persistent_workers=True,
    pin_memory=True,
    )

val_dataloader = DataLoader(
    dataset=val_set_total, 
    collate_fn = custom_collate_fn,
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'],
    persistent_workers=True,
    pin_memory=True,
    )

print(f"Number of batches in the training set: {len(train_dataloader)}")
print(f"Number of batches in the validation set: {len(val_dataloader)}")
#==================================================================================================
# Define the model

model = regression_Transformer(
    embedding_dim = config['embedding_dim'],
    n_layers = config['n_layers'],
    n_heads = config['n_heads'],
    input_dim = train_set_total[0].x.size(1),
    seq_dim = config['seq_dim'],
    dropout = config['dropout'],
    output_dim = config['output_dim'],
).to(device)

optimizer = torch.optim.AdamW(
                model.parameters(),
                lr = config['lr'],
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
            
scheduler = {
    'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #max_lr=self.opt_pars['max_lr'],
        max_lr=config['lr'],
        #total_steps=3197*300,
        epochs=config['n_epochs'],
        steps_per_epoch=len(train_dataloader),
        #pct_start=self.opt_pars['pct_start'],
        pct_start=0.3,
        #final_div_factor=self.opt_pars['final_div_factor'],
        final_div_factor=1e4,
        anneal_strategy='cos'),
    'interval': 'step',
    'frequency': 1,
}

optimizer = [optimizer], [scheduler]

#optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

lit_model = LitModel(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    batch_size=config['batch_size'],
)

# lit_model = BiGRUModel(
#     optimizer, 
#     input_size = train_set_total[0].x.size(1),
# )

#==================================================================================================
# Define the trainer
 
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=config['patience'],
        verbose=True,
    ),
    ModelCheckpoint(
        # save the best and latest model
        dirpath=tensorboard_logger.log_dir,
        filename='transformer-{epoch:02d}',
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min',
    ),
    TQDMProgressBar(),
            ]

trainer = Trainer(
    accelerator= 'gpu',
    devices = [0],
    max_epochs=config['n_epochs'],
    log_every_n_steps=1,
    logger=[wandb_logger, tensorboard_logger],
    callbacks=callbacks,
    #limit_train_batches=1, # Useful for testing the model on single batch
    #limit_val_batches=1,
)
#==================================================================================================
# Train the model

if checkpoint_path is not None:
    trainer.fit(lit_model, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
else:
    trainer.fit(lit_model, train_dataloader, val_dataloader)