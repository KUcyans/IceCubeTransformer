import os
from datetime import datetime
import time
import logging

import wandb

import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from DataSocket.DatasetMonoFlavourShard import DatasetMonoFlavourShard
from DataSocket.DatasetMultiFlavourShard import DatasetMultiFlavourShard
from DataSocket.DatasetMultiFlavourPart import DatasetMultiFlavourPart
from DataSocket.EnergyRange import EnergyRange
from DataSocket.MaxNDOMFinder import MaxNDOMFinder
from DataSocket.PMTfiedDataModule import PMTfiedDataModule
from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder 
    
def build_model(config: dict, nan_logger: logging.Logger, train_logger: logging.Logger):
    model = FlavourClassificationTransformerEncoder(
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
    )
    return model

def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

def setup_directories(base_dir: str, current_date: str, current_time: str):
    log_dir = os.path.join(base_dir, "logs", current_date)
    checkpoint_dir = os.path.join(base_dir, "checkpoints", current_date)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "train_log_file": os.path.join(log_dir, f"{current_time}_training.log"),
        "nan_log_file": os.path.join(log_dir, f"{current_time}_nan_checks.log")
    }
    
def setup_callbacks(checkpoint_dir: str, current_time: str):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=checkpoint_dir,
        filename=f"{current_time}_transformer-epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=10,
        verbose=True,
        mode="min",
    )

    return [checkpoint_callback, early_stopping_callback]

def init_wandb(project_name: str, model_class: nn.Module, max_epochs: int):
    wandb.init(
        project=project_name,
        config={
            "d_model": model_class.d_model,
            "n_heads": model_class.n_heads,
            "d_f": model_class.d_f,
            "num_layers": model_class.num_layers,
            "d_input": model_class.d_input,
            "num_classes": model_class.num_classes,
            "dropout": model_class.dropout,
            "learning_rate": model_class.learning_rate,
            "epochs": max_epochs,
            "attention": "Scaled Dot-Product",
        },
    )
    
def log_training_parameters(logger: logging.Logger, model_class: nn.Module, max_epochs: int):
    message = (
        "| Parameter       | Value               |\n"
        "|-----------------|---------------------|\n"
        f"| attention       | Scaled Dot-Product |\n"
        f"| d_model         | {model_class.d_model:<15}|\n"
        f"| n_heads         | {model_class.n_heads:<15}|\n"
        f"| d_f             | {model_class.d_f:<15}|\n"
        f"| num_layers      | {model_class.num_layers:<15}|\n"
        f"| d_input         | {model_class.d_input:<15}|\n"
        f"| num_classes     | {model_class.num_classes:<15}|\n"
        f"| dropout         | {model_class.dropout:<15}|\n"
        f"| learning_rate   | {model_class.learning_rate:<15}|\n"
        f"| epochs          | {max_epochs:<15}|\n"
    )
    logger.info("Starting training with the following parameters:")
    logger.info(message)  # Log to file
    return message
    
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
    
def runTrainingAndTesting(base_dir: str, model_config: dict, datamodule: PMTfiedDataModule):
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M%S")
    
    dirs = setup_directories(base_dir, current_date, current_time)
    
    train_logger = setup_logger("training", dirs["train_log_file"])
    nan_logger = setup_logger("nan_checks", dirs["nan_log_file"])
    
    max_epochs = model_config["epochs"]
    model_class = build_model(model_config, nan_logger, train_logger)
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=dirs["log_dir"],
        name=f"{current_time}",
    )
    callbacks = setup_callbacks(dirs["checkpoint_dir"], current_time)
    
    # Initialize WandB
    project_name = f"[{current_date}_{current_time}]Neutrino Flavour Classification"
    init_wandb(project_name, model_class, max_epochs)
    
    # Log training parameters
    params = log_training_parameters(train_logger, model_class, max_epochs)

    # Set up Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[1], 
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=tb_logger,
    )

    trainer.fit(model_class, datamodule=datamodule)
    
    train_logger.info("Training completed.")
    train_logger.info("Saving model...")
    torch.save(model_class.state_dict(), os.path.join(dirs["checkpoint_dir"], f"{current_time}_transformer_final.pth"))
    train_logger.info("Model saved.")
    
    wandb.finish()
    # test
    trainer.test(model_class, datamodule=datamodule)
    print(params)
    
def execute():
    secure_the_area()
    root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/"
    config ={
        "d_model": 128,
        "n_heads": 8,
        "d_f": 256,
        "num_layers": 3,
        "d_input": 32,
        "num_classes": 3,
        "dropout": 0.1,
        "learning_rate": 1e-5,
        "epochs": 100,
        "attention": "Scaled Dot-Product",
    }
    maxNDOMFinder_PeV_1_1 = MaxNDOMFinder(
        root_dir=root_dir,
        energy_band=EnergyRange.ER_1_PEV_100_PEV,
        part=1,
        shard=1,
        )
    
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
        batch_size=128,
        num_workers=8,
        verbosity=1,
        )
    runTrainingAndTesting(root_dir, config, dm_PeV_1_1)

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