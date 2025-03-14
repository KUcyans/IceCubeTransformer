import time
import json
import os
import torch
import logging
import argparse

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
from VernaDataSocket.MultiFlavourDataModule import MultiFlavourDataModule
from Enum.EnergyRange import EnergyRange

import sys
sys.stdout.reconfigure(encoding='utf-8')

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
        "best_lr_dir": os.path.join(base_dir, "best_lr", current_date),
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

    
def run_best_lr(config_dir: str, config_file: str, training_dir: str, data_root_dir: str, er: EnergyRange):
    """Find the best learning rate and save the plot."""
    args = parse_args()
    current_date = args.date

    # ✅ Load Configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
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
    fig = lr_finder.plot(suggest=True)
    fig_title = f"batch{config['batch_size']}_dim{config['embedding_dim']}_layers{config['n_layers']}_seq{config['event_length']}_heads{config['n_heads']}"
    fig.suptitle(fig_title)
    fig_path = os.path.join(dirs["best_lr_dir"], f"{fig_title}.pdf")
    fig.savefig(fig_path, format="pdf", dpi=300)
    print(f"📁 Best LR plot saved at: {fig_path}")


if __name__ == "__main__":
    training_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(training_dir, "config")
    config_file = "config.json"
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_Contained"
    start_time = time.time()
    er = EnergyRange.ER_10_TEV_1_PEV
    run_best_lr(config_dir=config_dir,
                    config_file=os.path.join(config_dir, config_file),
                    training_dir=training_dir,
                    data_root_dir=data_root_dir,
                    er=er)
    end_time = time.time()
    print(f"Learning Rate Finder took {end_time - start_time:.2f} seconds.")

