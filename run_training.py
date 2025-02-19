import time
import json
import os
import wandb 
import torch
import logging
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar

from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
from SnowyDataSocket.MultiPartDataModule import MultiPartDataModule


def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


def init_wandb(config: dict, project_name: str):
    """Initialize Weights & Biases logging."""
    wandb.init(project=project_name, config=config)


def log_training_parameters(config: dict, training_logger: logging.Logger):
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

    message = """| Parameter       | Value               |
    |-----------------|---------------------|
    """ + "".join([f"| {k:<30} | {str(v):<20} |\n" for k, v in config_flattened.items()])

    training_logger.info("Starting training with the following parameters:")
    training_logger.info(message)


def build_model(config: dict, 
                train_logger: logging.Logger, 
                nan_logger: logging.Logger, 
                device: torch.device):
    """Build and return the model."""
    model = FlavourClassificationTransformerEncoder(
        d_model=config['embedding_dim'],
        n_heads=config['n_heads'],
        d_f=config['embedding_dim'],
        num_layers=config['n_layers'],
        d_input=config['d_input'],
        num_classes=config['output_dim'],
        seq_len=config['event_length'],
        attention_type=config['attention'],
        dropout=config['dropout'],
        nan_logger=nan_logger,
        train_logger=train_logger,
    )
    return model.to(device)


def build_data_module(config: dict, 
                      root_dir:str, 
                      optimizer: torch.optim.Optimizer):
    """Build and return the datamodule."""
    datamodule = MultiPartDataModule(
        root_dir=root_dir,
        subdirectory_parts_train=config['train_data'],
        subdirectory_parts_val=config['validate_data'],
        event_length=config['event_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sample_weights_train=config.get('sample_weights'),
        sample_weights_val=config.get('sample_weights'),
        optimizer=optimizer
    )
    return datamodule


def build_optimiser(config: dict, model: torch.nn.Module, train_dataloader_length: int):
    """Build and return the optimizer and learning rate scheduler."""
    optimizer_config = config['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr'],
        betas=tuple(optimizer_config['betas']),
        eps=optimizer_config['eps'],
        weight_decay=optimizer_config['weight_decay'],
        amsgrad=optimizer_config['amsgrad']
    )

    scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer_config['lr'],
            epochs=config['n_epochs'],
            steps_per_epoch=train_dataloader_length,
            pct_start=optimizer_config['pct_start'],
            final_div_factor=optimizer_config['final_div_factor'],
            anneal_strategy=optimizer_config['anneal_strategy']
        ),
        'interval': optimizer_config['interval'],
        'frequency': optimizer_config['frequency'],
    }

    return [optimizer], [scheduler]


def build_callbacks(config: dict, callback_dir: str):
    """Build and return training callbacks."""
    callbacks = [
        EarlyStopping(monitor='val_loss', 
                      patience=config['patience'], 
                      verbose=True),
        ModelCheckpoint(dirpath=callback_dir,
                        filename="{epoch:02d}_{val_loss:.2f}",  # ✅ Fixed Syntax
                        save_top_k=1, 
                        save_last=True, 
                        monitor='val_loss', 
                        mode='min'),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar()  # ✅ Added missing comma
    ]
    return callbacks


def secure_the_area(config):
    """Configure GPU if available, else use CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_float32_matmul_precision('high')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config['gpu']))
    else:
        device = torch.device('cpu')
    return device


def setup_directories(base_dir: str, current_date: str, current_time: str):
    """Create and return directories for logs and checkpoints."""
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


def run_training(config_file: str, training_dir: str, data_root_dir: str):
    """Main function to run the training pipeline."""
    current_date, current_time = time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S")
    
    # ✅ Load Configuration
    project_name = f"[{current_date}] Flavour Classification"
    with open(config_file, 'r') as f:
        config = json.load(f)

    # ✅ Setup directories and loggers
    dirs = setup_directories(training_dir, current_date, current_time)
    train_logger = setup_logger("training", dirs["train_log_file"])
    nan_logger = setup_logger("nan_checks", dirs["nan_log_file"])
    
    # ✅ Secure GPU/CPU
    device = secure_the_area(config)

    # ✅ Build Model
    model = build_model(config=config, 
                        train_logger=train_logger, 
                        nan_logger=nan_logger, 
                        device=device)

    # ✅ Build DataModule (without optimizer first)
    datamodule = build_data_module(config=config, 
                                   root_dir=data_root_dir, 
                                   optimizer=None)
    datamodule.setup(stage='fit')  # ✅ Prepare datasets

    # ✅ Build Optimizer (after DataModule setup to get train_dataloader_length)
    train_dataloader_length = len(datamodule.train_dataloader())
    optimizer = build_optimiser(config=config, 
                                model=model, 
                                train_dataloader_length=train_dataloader_length)

    # ✅ Assign optimizer to DataModule
    datamodule.optimizer = optimizer

    # ✅ Build Callbacks
    callbacks = build_callbacks(config=config, callback_dir=dirs["checkpoint_dir"])

    # ✅ Initialize WandB Logger
    init_wandb(config=config, project_name=project_name)
    wandb_logger = WandbLogger(project=project_name, config=config)

    # ✅ Log Training Parameters
    log_training_parameters(config=config, training_logger=train_logger)

    # ✅ Initialize Trainer and Fit
    trainer = pl.Trainer(
        max_epochs=config['n_epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=device.type,
        devices=len(config['gpu']) if device.type == 'cuda' else 1
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config_dir = "/groups/icecube/cyan/factory/IceCubeTransformer/config/"
    config_file = "config_default.json"
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/"
    training_dir = os.path.dirname(os.path.realpath(__file__))
    start_time = time.time()
    run_training(config_file=os.path.join(config_dir, config_file),
                 training_dir=training_dir,
                 data_root_dir=data_root_dir)
    end_time = time.time()
    print(f"Training completed in {time.strftime('%d:%H:%M:%S', time.gmtime(end_time - start_time))}")

