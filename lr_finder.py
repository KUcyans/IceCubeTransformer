import time
import json
import os
import torch
import logging
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
from VernaDataSocket.MultiFlavourDataModule import MultiFlavourDataModule
from Enum.EnergyRange import EnergyRange

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax

def lock_and_load(config):
    """Set CUDA device based on config['gpu'] if available, else use CPU."""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    available_devices = list(range(torch.cuda.device_count()))
    print(f"Available CUDA devices: {available_devices}")

    if torch.cuda.is_available() and len(config.get('gpu', [])) > 0:
        selected_gpu = int(config['gpu'][0])

        if selected_gpu in available_devices:
            print("🔥 LOCK AND LOAD! GPU ENGAGED! 🔥")
            device = torch.device(f"cuda:{selected_gpu}")  # ✅ Use the correct index
            torch.cuda.set_device(selected_gpu)  # ✅ Explicitly set device
            torch.set_float32_matmul_precision('highest')
            print(f"Using GPU: {selected_gpu} (cuda:{selected_gpu})")
        else:
            print(f"⚠️ Warning: GPU {selected_gpu} is not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    print(f"Selected device: {device}")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script with Timestamped Logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    return parser.parse_args()


def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

def log_training_parameters(config: dict):
    """Log training parameters for debugging and reproducibility."""
    def flatten_dict(d, parent_key='', sep=' -> '):
        """Recursively flatten nested dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    config_flattened = flatten_dict(config)

    message = """\n
    | Parameter       | Value               |
    |-----------------|---------------------|
    """ + "".join([f"| {k:<30} | {str(v):<20} |\n" for k, v in config_flattened.items()])

    print("Starting training with the following parameters:")
    print(message)

def setup_directories(base_dir: str, current_date: str):
    """Create and return directories for logs and checkpoints with a timestamped subdirectory."""
    
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "best_lr_dir": os.path.join(base_dir, "lr_find", current_date),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def build_model(config: dict, 
                device: torch.device,):
    """Build and return the model."""
    model = FlavourClassificationTransformerEncoder(
        d_model=config['embedding_dim'],
        n_heads=config['n_heads'],
        d_f=config['embedding_dim'] * 4,
        num_layers=config['n_layers'],
        d_input= config['d_input'],
        num_classes=config['output_dim'],
        seq_len=config['event_length'],
        attention_type=config['attention'],
        dropout=config['dropout'],
    )
    return model.to(device)


def build_data_module(config: dict, er: EnergyRange, root_dir: str):
    """Build and return the datamodule."""
    datamodule = MultiFlavourDataModule(
        root_dir=root_dir,
        er=er,
        N_events_nu_e=config['N_events_nu_e'],
        N_events_nu_mu=config['N_events_nu_mu'],
        N_events_nu_tau=config['N_events_nu_tau'],
        event_length=config['event_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        frac_train=config['frac_train'],
        frac_val=config['frac_val'],
        frac_test=config['frac_test'],
    )
    datamodule.setup(stage="fit")
    return datamodule

def read_config(config_file: str):
    with open(config_file, 'r') as f:
        original_config = json.load(f)    
    config = copy.deepcopy(original_config) 
    return config
    
def save_lr_plot(lr_finder, config, save_dir):
    setMplParam()
    fig, ax = plt.subplots(figsize=(13, 9))
    lrs = np.array(lr_finder.results["lr"])
    losses = np.array(lr_finder.results["loss"])
    ax.plot(lrs, losses, label = "Loss", color = getColour(0), linewidth = 2)
    best_lr = lr_finder.suggestion()
    best_idx = (np.abs(lrs - best_lr)).argmin()
    loss_at_best_lr = losses[best_idx]
    ax.scatter(best_lr, loss_at_best_lr, s=75, c=getColour(2), label = f"Best LR: {best_lr:.4e}")
    ax.set_ylim(0.0, 1.0)  # Set y-axis range
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.legend()
    d = {"best_lr" : f"{lr_finder.suggestion():.4e}",
         "loss at best_lr" : f"{loss_at_best_lr:.4f}",
         }
    add_text_to_ax(0.45, 0.85, nice_string_output(d), ax, fontsize=12)

    title = f"batch{config['batch_size']}_dim{config['embedding_dim']}_layers{config['n_layers']}_seq{config['event_length']}_heads{config['n_heads']}"
    ax.set_title(f"Learning Rate Finder: {title}", fontsize=20)
    
    fig_path = os.path.join(save_dir, f"{title}.pdf")
    fig.savefig(fig_path, format="pdf", dpi=300)
    print(f"📁 Best LR plot saved at: {fig_path}")

    
def run_best_lr(config_file: str, training_dir: str, data_root_dir: str, er: EnergyRange):
    """Find the best learning rate and save the plot."""
    args = parse_args()
    current_date = args.date

    config = read_config(config_file)
    log_training_parameters(config)
    
    # ✅ Setup directories
    dirs = setup_directories(base_dir=training_dir, 
                             current_date=current_date)

    # ✅ Secure GPU/CPU!
    device = lock_and_load(config)

    # ✅ Build DataModule
    datamodule = build_data_module(config, er, data_root_dir)

    # ✅ Build Model
    model = build_model(config, device)

    # ✅ Run LR Finder
    trainer = pl.Trainer(devices=config['gpu'], accelerator="gpu" if torch.cuda.is_available() else "cpu")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)

    # ✅ Get and Save Best LR
    best_lr = lr_finder.suggestion()
    print(f"🚀 Suggested Learning Rate: {best_lr:.7e}")

    # ✅ Save LR Finder Plot
    save_lr_plot(lr_finder, config, dirs["best_lr_dir"])


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(this_dir, "config")
    config_file = "config.json"
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_Contained"
    # config_file = "config_35.json"
    # data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered_second_round/Snowstorm/CC_CRclean_Contained"
    start_time = time.time()
    # er = EnergyRange.ER_10_TEV_1_PEV
    er = EnergyRange.ER_1_PEV_100_PEV
    run_best_lr(config_file=os.path.join(config_dir, config_file),
                    training_dir=this_dir,
                    data_root_dir=data_root_dir,
                    er=er)
    end_time = time.time()
    print(f"Learning Rate Finder took {end_time - start_time:.2f} seconds.")

