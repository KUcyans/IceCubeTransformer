import os
from datetime import datetime
import time
import logging
import argparse

import wandb

import torch
import torch.profiler
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from DataSocket_Jan.DatasetMultiFlavourShard_Micro import DatasetMultiFlavourShard_Micro
from DataSocket_Jan.DatasetMonoFlavourShard import DatasetMonoFlavourShard
from DataSocket_Jan.DatasetMultiFlavourShard import DatasetMultiFlavourShard
from DataSocket_Jan.DatasetMultiFlavourPart import DatasetMultiFlavourPart
from DataSocket_Jan.EnergyRange import EnergyRange
from DataSocket_Jan.MaxNDOMFinder import MaxNDOMFinder
from DataSocket_Jan.PMTfiedDataModule import PMTfiedDataModule
from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder 

# ----------------------------------------------
# Logger Setup
# ----------------------------------------------
def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script with Timestamped Logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    return parser.parse_args()

# ----------------------------------------------
# Directory Setup
# ----------------------------------------------
def setup_directories(base_dir: str, current_date: str, current_time: str):
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "torch_profile_dir": os.path.join(base_dir, "torch_profile", current_date),
        "checkpoint_dir": os.path.join(base_dir, "checkpoints", current_date),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return {
        **paths,
        "train_log_file": os.path.join(paths["log_dir"], f"{current_time}_training.log"),
        "nan_log_file": os.path.join(paths["log_dir"], f"{current_time}_nan_checks.log"),
    }
    
# ----------------------------------------------
# Callbacks Setup
# ----------------------------------------------
def setup_callbacks(checkpoint_dir: str, current_time: str):
    return [
        ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=f"{current_time}_transformer-epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",  # Minimise loss
    ),
    EarlyStopping(monitor="val_acc", patience=100000, verbose=True, mode="min"),
    ]

# ----------------------------------------------
# WandB Setup
# ----------------------------------------------
def init_wandb(project_name: str, model: nn.Module, max_epochs: int):
    wandb.init(
        project=project_name,
        config={
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "d_f": model.d_f,
            "num_layers": model.num_layers,
            "d_input": model.d_input,
            "num_classes": model.num_classes,
            "dropout": model.dropout,
            "learning_rate": model.learning_rate,
            "epochs": max_epochs,
            "attention": "Scaled Dot-Product",
        },
    )
    
# ----------------------------------------------
# Torch Profiler Setup
# ----------------------------------------------
def setup_profiler(profiler_dir: str, current_time: str):
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(profiler_dir, f"profiler_{current_time}")),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

def log_training_parameters(logger: logging.Logger, config: dict):
    message = (
        "| Parameter       | Value               |\n"
        "|-----------------|---------------------|\n"
        f"| attention       | {config['attention']:<15}|\n"
        f"| d_model         | {config['d_model']:<15}|\n"
        f"| n_heads         | {config['n_heads']:<15}|\n"
        f"| d_f             | {config['d_f']:<15}|\n"
        f"| num_layers      | {config['num_layers']:<15}|\n"
        f"| d_input         | {config['d_input']:<15}|\n"
        f"| num_classes     | {config['num_classes']:<15}|\n"
        f"| dropout         | {config['dropout']:<15}|\n"
        f"| learning_rate   | {config['learning_rate']:<15}|\n"
        f"| epochs          | {config['epochs']:<15}|\n"
        f"| batch_size      | {config['batch_size']:<15}|\n"
    )
    logger.info("Starting training with the following parameters:")
    logger.info(message)  # Log to file
    return message


# ----------------------------------------------
# Trainer Setup
# ----------------------------------------------
def create_trainer(max_epochs: int, checkpoint_dir: str, callbacks: list, wandb_logger: WandbLogger, gpu_no: int):
    return Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[gpu_no],
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
        profiler="advanced",
        precision="bf16" if torch.cuda.is_available() else "16",
        # amp_backend="native",
    )

def secure_the_area():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected.")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.set_float32_matmul_precision('high')

# ----------------------------------------------
# Data Setup
# ----------------------------------------------
def prepare_data(root_dir: str, batch_size: int):
    secure_the_area()
    maxNDOMFinder_PeV_1_1 = MaxNDOMFinder(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        part=1,
        shard=1,
        )
    # ----------------------------------------------
    # dataset with shard 1
    # ----------------------------------------------
    ds_PeV_1_1 = DatasetMultiFlavourShard(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        part=1,
        shard=1,
        max_n_doms=maxNDOMFinder_PeV_1_1(),
        verbosity=1,
        )
    
    dm_PeV_1_1 = PMTfiedDataModule(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        dataset = ds_PeV_1_1,
        batch_size=batch_size,
        num_workers=8,
        verbosity=1,
        )
    
    # ----------------------------------------------
    # dataset with 10 events
    # ----------------------------------------------
    ds_PeV_1_1_first10 = DatasetMultiFlavourShard_Micro(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        part=1,
        shard=1,
        max_n_doms=maxNDOMFinder_PeV_1_1(),
        verbosity=1,
        first_n_events=10,
        )
    
    dm_PeV_1_1_first10 = PMTfiedDataModule(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        dataset=ds_PeV_1_1_first10,
        batch_size=batch_size,
        num_workers=8,
        verbosity=1,
        )
    
    # ----------------------------------------------
    # dataset with part 1
    # ----------------------------------------------
    # maxNDOMFinder_PeV_1 = MaxNDOMFinder(
    #     root_dir=root_dir,
    #     energy_band=EnergyRange.ER_1_PEV_100_PEV,
    #     part=1,
    #     )
    # ds_PeV_1 = DatasetMultiFlavourPart(
    #     root_dir=root_dir,
    #     energy_band=EnergyRange.ER_1_PEV_100_PEV,
    #     part=1,
    #     max_n_doms=maxNDOMFinder_PeV_1(),
    #     verbosity=1,
    #     )
    # dm_PeV_1 = PMTfiedDataModule(
    #     root_dir=root_dir,
    #     energy_band=EnergyRange.ER_1_PEV_100_PEV,
    #     dataset=ds_PeV_1,
    #     batch_size= batch_size,
    #     num_workers=8,
    #     verbosity=1,
    #     )
    
    return dm_PeV_1_1_first10

# ----------------------------------------------
# Model Setup
# ----------------------------------------------
def build_model(config: dict, nan_logger: logging.Logger, train_logger: logging.Logger, profiler=None):
    return FlavourClassificationTransformerEncoder(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_f=config["d_f"],
        num_layers=config["num_layers"],
        d_input=config["d_input"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
        nan_logger=nan_logger,
        train_logger=train_logger,
        profiler=profiler,
    )
    
def validate_config(config: dict):
    required_keys = {
        "d_model": int, "n_heads": int, "d_f": int, "num_layers": int,
        "d_input": int, "num_classes": int, "dropout": float,
        "learning_rate": float, "epochs": int, "attention": str, "batch_size": int
    }
    
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing key in config: {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Incorrect type for {key}: expected {expected_type}, got {type(config[key])}")

    print("Config validation passed.")

# ----------------------------------------------
# Training Execution
# ----------------------------------------------
def run_training(base_dir: str, model_config: dict, datamodule: PMTfiedDataModule):
    args = parse_args()
    current_date, current_time = args.date, args.time
    # current_date, current_time = datetime.now().strftime("%Y%m%d"), datetime.now().strftime("%H%M%S")
    
    dirs = setup_directories(base_dir, current_date, current_time)

    train_logger = setup_logger("training", dirs["train_log_file"])
    nan_logger = setup_logger("nan_checks", dirs["nan_log_file"])

    validate_config(model_config)
    
    profiler = setup_profiler(dirs["torch_profile_dir"], current_time)
    model = build_model(model_config, nan_logger, train_logger, profiler=None)

    # Initialise WandB
    project_name = f"[{current_date}] Flavour Classification"
    init_wandb(project_name, model, model_config["epochs"])
    wandb_logger = WandbLogger(project=project_name)

    # Create Trainer
    params = log_training_parameters(train_logger, model_config)
    trainer = create_trainer(model_config["epochs"], dirs["checkpoint_dir"], setup_callbacks(dirs["checkpoint_dir"], current_time), wandb_logger, model_config["gpu_no"])

    # Torch Profiler
    with profiler:
        trainer.fit(model, datamodule=datamodule)

    # Save model
    torch.save(model.state_dict(), os.path.join(dirs["checkpoint_dir"], f"{current_time}_transformer_final.pth"))
    train_logger.info("Model saved.")

    # Finish WandB session
    wandb.finish()
    print(params)

def execute():
    secure_the_area()
    root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/"
    base_dir = "/groups/icecube/cyan/factory/IceCubeTransformer/"
    config ={
        "d_model": 128,
        "n_heads": 8,
        "d_f": 128,
        "num_layers": 3,
        "d_input": 32,
        "num_classes": 3,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "epochs": 50,
        "attention": "Scaled Dot-Product",
        "batch_size": 16,
        "gpu_no": 0, # 0 or 1
    }
    datamodule = prepare_data(root_dir, config["batch_size"])
    run_training(base_dir, config, datamodule)
    
def main():
    t_start = time.time()
    execute()
    t_end = time.time()
    elapsed_time = t_end - t_start
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"######## Execution time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} ########")

if __name__ == "__main__":
    main()
    # use this spell: 
    # nohup python3.9 TrainingDebuggingYard.py --date $(date +"%Y%m%d") --time $(date +"%H%M%S") > logs/$(date +"%Y%m%d")/$(date +"%H%M%S")_execution.log 2>&1 &