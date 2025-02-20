import torch
import pandas as pd
import numpy as np
import os

#from graphnet.data.dataloader import DataLoader
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import TQDMProgressBar

from pytorch_lightning import Trainer

from dataset_multifile_flavours import PMTfiedDatasetPyArrow
from dataloader import custom_collate_fn
from model import regression_Transformer
from model import LitModel

import time

import json
with open('config.json') as f:
    config = json.load(f)

#==================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    device = torch.device('cuda:0')

inference_dir_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22011/truth_3.parquet"]
inference_dir_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22014/truth_3.parquet"]
inference_dir_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22017/truth_3.parquet"]
#inference_dir.append("/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22011/truth_4.parquet")
#inference_dir.append("/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/22012/truth_4.parquet")

inference_dataset = PMTfiedDatasetPyArrow(
    truth_paths_1 = inference_dir_1,
    truth_paths_2 = inference_dir_2,
    truth_paths_3 = inference_dir_3,
    sample_weights=[1,1,1],
    selection=None,
    )

inference_dataloader = DataLoader(
    dataset = inference_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=4, 
    collate_fn=custom_collate_fn,
    persistent_workers=True,
    )

# Define the model
model = regression_Transformer(
    embedding_dim = config['embedding_dim'],
    n_layers = config['n_layers'],
    n_heads = config['n_heads'],
    input_dim = inference_dataset[0].x.size(1),
    seq_dim = config['seq_dim'],
    dropout = config['dropout'],
    output_dim = config['output_dim'],
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

lit_model = LitModel(
    model,
    optimizer,
    None,
    inference_dataloader,
    batch_size=config['batch_size'],
)
callbacks = TQDMProgressBar()

trainer = Trainer(
    #accelerator='cpu',
    accelerator='gpu',
    devices = [0],
    log_every_n_steps=1,
    callbacks=callbacks,
    logger = None,
)

current_date = time.strftime("%Y-%m-%d")
current_time = time.strftime("%H:%M:%S")
ckpt_dir = f"/groups/icecube/cyan/factory/IceCubeTransformer/ImportLuc/checkpoints/version_1"

ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

# if "last.ckpt" in ckpt_files:
#     ckpt_file = "last.ckpt"  # Use the last saved checkpoint
# else:
#     ckpt_file = sorted(ckpt_files)[-1] 

# ckpt_file = "last.ckpt"
ckpt_file = "transformer-epoch=27.ckpt"

ckpt_path = os.path.join(ckpt_dir, ckpt_file)

# Manually load checkpoint with weights_only=False
checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
lit_model.load_state_dict(checkpoint['state_dict'])

# Predict without passing ckpt_path
predictions = trainer.predict(
    model = lit_model,
    dataloaders = inference_dataloader,
)


print('Start predicting')
predictions = trainer.predict(
    model = lit_model,
    dataloaders = inference_dataloader,
    ckpt_path = ckpt_path,
    )

print('Predictions done')

print('Start storing the predictions')
# Store predictions as class indices
# Store predictions as both one-hot encoded vectors and class indices
pred_classes = []
target_classes = []
pred_one_hot = []
target_one_hot = []

for i in range(len(predictions)):
    y_pred = predictions[i]['y_pred']
    target = predictions[i]['target']
    
    if i == 0:
        print('y_pred', y_pred)  # Debugging

    # Convert logits to class indices
    pred_class = torch.argmax(y_pred, dim=-1)  # Shape: (batch_size,)
    target_class = torch.argmax(target, dim=-1)  # Shape: (batch_size,)

    # Convert class indices to one-hot encoding
    pred_one_hot_vec = torch.nn.functional.one_hot(pred_class, num_classes=3).tolist()
    target_one_hot_vec = torch.nn.functional.one_hot(target_class, num_classes=3).tolist()

    pred_classes.append(pred_class)
    target_classes.append(target_class)
    pred_one_hot.extend(pred_one_hot_vec)
    target_one_hot.extend(target_one_hot_vec)

# Convert lists to tensors
pred_classes = torch.cat(pred_classes, dim=0)
target_classes = torch.cat(target_classes, dim=0)

print('Predictions shape:', pred_classes.shape)  # Should be (num_samples,)
print('Targets shape:', target_classes.shape)  # Should be (num_samples,)


# Save predictions in a DataFrame
df = pd.DataFrame({
    "pred_one_hot_pid": pred_one_hot,
    "target_one_hot_pid": target_one_hot,
    "pred_class": pred_classes.numpy(),
    "target_class": target_classes.numpy(),
})


prediction_dir = "/groups/icecube/cyan/factory/IceCubeTransformer/ImportLuc/Predictions/"
prediction_file = f"{current_date}_{current_time}_TRUTH_3_VAL_epoch50_lasts.csv"
df.to_csv(prediction_dir + prediction_file, index=False)

print('Predictions stored')
