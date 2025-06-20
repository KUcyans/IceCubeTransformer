import time
import json
import os
import wandb 
import torch
import logging
import argparse
import copy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.tuner.tuning import Tuner

from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
# from TrainingUtils.LocalMinimumCheckpoint import LocalMinimumCheckpoint
from TrainingUtils.MidEpochCheckPoint import MidEpochCheckpoint
# from TrainingUtils.EquinoxDecayingAsymmetricSinusoidal import EquinoxDecayingAsymmetricSinusoidal
# from TrainingUtils.KatsuraCosineAnnealingWarmupRestarts import CosineAnnealingWarmupRestarts
# from Enum.LrDecayMode import LrDecayMode
from VernaDataSocket.MultiFlavourDataModule import MultiFlavourDataModule
from Enum.EnergyRange import EnergyRange
from Enum.Flavour import Flavour
from Enum.ClassificationMode import ClassificationMode
from Enum.AttentionType import AttentionType
from Enum.PositionalEncodingType import PositionalEncodingType
from Enum.LossType import LossType

import sys
sys.stdout.reconfigure(encoding='utf-8')

def lock_and_load(config):
    """Set CUDA device based on config['gpu'] if available, else use CPU."""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    available_devices = list(range(torch.cuda.device_count()))
    print(f"Available CUDA devices: {available_devices}")

    if torch.cuda.is_available() and len(config.get('gpu', [])) > 0:
        # selected_gpu = int(config['gpu'][0])
        requested_gpus = config.get('gpu', [])
        selected_gpu = int(requested_gpus[0]) if requested_gpus else 0

        if selected_gpu in available_devices:
            torch.cuda.empty_cache()
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


def build_callbacks(config: dict, callback_dir: str):
    """Build and return training callbacks."""
    # One saves best model by validation loss
    checkpoint_loss = ModelCheckpoint(
        dirpath=callback_dir,
        monitor="val_loss",
        mode="min",
        save_last=True, 
        save_top_k=4,
        filename="{epoch}-{val_loss:.3f}"
    )
    
    checkpoint_mid = MidEpochCheckpoint(
        dirpath=callback_dir,
        max_epochs=config['n_epochs'],
        window=(17, 35),
        save_interval=3,
        filename="{epoch}-mid.ckpt"
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', 
                      patience=config['patience'], 
                      verbose=True),
        checkpoint_loss,
        checkpoint_mid,
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=1000),
    ]
    
    if LossType.from_string(config['loss']) == LossType.TAUPURITYMSE:
        checkpoint_tau = ModelCheckpoint(
            dirpath=callback_dir,
            monitor="val_tau_purity",
            mode="max",
            save_last=False, 
            save_top_k=2,
            filename="{epoch}-{val_tau_purity:.3f}"
            )
        callbacks.append(checkpoint_tau)
    return callbacks

def setup_directories(base_dir: str, config_dir:str, current_date: str, current_time: str):
    """Create and return directories for logs and checkpoints with a timestamped subdirectory."""
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "checkpoint_dir": os.path.join(base_dir, "checkpoints", current_date, current_time),
        "config_history": os.path.join(config_dir, "history"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def build_model(config: dict, 
                device: torch.device,):
    """Build and return the model."""
    classification_mode = ClassificationMode.from_string(config['classification_mode'])
    num_classes = classification_mode.num_classes
    attention_type = AttentionType.from_string(config['attention'])
    positional_encoding_type = PositionalEncodingType.from_string(config['positional_encoding'])
    loss_type = LossType.from_string(config['loss'])
    model = FlavourClassificationTransformerEncoder(
        d_model=config['embedding_dim'],
        n_heads=config['n_heads'],
        d_f=config['embedding_dim'] * 4,
        num_layers=config['n_layers'],
        d_input=config['d_input'],
        n_output_layers=config['n_output_layers'],
        num_classes=num_classes,
        seq_len=config['event_length'],
        loss_type=loss_type,
        attention_type=attention_type,
        positional_encoding_type=positional_encoding_type,
        dropout=config['dropout'],
    )
    return model.to(device)


def build_data_module(config: dict, er: EnergyRange, root_dir: str, root_dir_corsika: str = None):
    """Build and return the datamodule."""
    classification_mode = ClassificationMode.from_string(config['classification_mode'])
    datamodule = MultiFlavourDataModule(
        root_dir=root_dir,
        er=er,
        N_events_nu_e=config['N_events_nu_e'],
        N_events_nu_mu=config['N_events_nu_mu'],
        N_events_nu_tau=config['N_events_nu_tau'],
        N_events_noise=config['N_events_noise'],
        event_length=config['event_length'],
        inference_event_length=config['inference_event_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        frac_train=config['frac_train'],
        frac_val=config['frac_val'],
        frac_test=config['frac_test'],
        classification_mode=classification_mode,
        root_dir_corsika=root_dir_corsika,
    )
    datamodule.setup(stage="fit")
    return datamodule

    
def build_optimiser_and_scheduler(config: dict, 
                                model: torch.nn.Module, 
                                datamodule: MultiFlavourDataModule):
    """Build and return the optimizer and learning rate scheduler."""
    optimizer_config = config['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr_max'],
        betas=tuple(optimizer_config['betas']),
        eps=optimizer_config['eps'],
        weight_decay=optimizer_config['weight_decay'],
        amsgrad=optimizer_config['amsgrad']
    )
    # steps_per_epoch = max(len(datamodule.train_dataloader()), 1)  # Prevent division by zero
    # total_N_steps = config["n_epochs"] * steps_per_epoch

    total_N_steps = config['n_epochs'] * len(datamodule.train_dataloader())
    
    one_cycle_scheduler_config = config['one_cycle_scheduler']
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=config['n_epochs'],
            total_steps=total_N_steps,
            pct_start=one_cycle_scheduler_config['pct_start'],
            max_lr=one_cycle_scheduler_config['lr_max'],
            div_factor=one_cycle_scheduler_config['div_factor'],
            final_div_factor=one_cycle_scheduler_config['final_div_factor'],
            max_momentum=one_cycle_scheduler_config['max_momentum'],
            base_momentum=one_cycle_scheduler_config['base_momentum'],
            anneal_strategy=one_cycle_scheduler_config['anneal_strategy']
        ),
    }
    
    # equinox_scheduler_config = config['equinox_scheduler']
    # scheduler = {
    #     'scheduler': EquinoxDecayingAsymmetricSinusoidal(
    #         optimizer, 
    #         lr_max=equinox_scheduler_config['lr_max'],
    #         lr_min=equinox_scheduler_config['lr_min'],
    #         total_steps=total_N_steps,
    #         frequency_per_section=equinox_scheduler_config['frequency_per_section'],
    #         n_sections=equinox_scheduler_config['n_sections'],
    #         lr_decay=LrDecayMode.from_str(equinox_scheduler_config['lr_decay']),
    #     ),
    # }
    
    # katsura_scheduler_config = config['katsura_scheduler']
    # scheduler = {
    #     'scheduler': CosineAnnealingWarmupRestarts(
    #         optimizer, 
    #         first_cycle_steps=katsura_scheduler_config['first_cycle_steps'],
    #         cycle_mult=katsura_scheduler_config['cycle_mult'],
    #         max_lr=katsura_scheduler_config['max_lr'],
    #         min_lr=katsura_scheduler_config['min_lr'],
    #         warmup_steps=katsura_scheduler_config['warmup_steps'],
    #         gamma=katsura_scheduler_config['gamma'],
    #     ),
    # }
    return {"optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler['scheduler'],
                "interval": optimizer_config['interval'],
                "frequency": optimizer_config['frequency'],
            }
    }

def read_and_copy_config(config_file: str, config_copy_path: str):
    with open(config_file, 'r') as f:
        original_config = json.load(f)    
    config = copy.deepcopy(original_config) 
    
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=4)
    return config
     

def run_training(config_dir: str, 
                 config_file: str, 
                 training_dir: str, 
                 data_root_dir: str,
                 data_root_dir_corsika: str,
                 er: EnergyRange):
    args = parse_args()
    current_date, current_time = args.date, args.time
    project_name = f"[{current_date}] Flavour Classification"
    
    # ✅ Setup directories and loggers
    dirs = setup_directories(training_dir, config_dir, current_date, current_time)
    
    # ✅ Load Configuration
    config_copy_path = os.path.join(dirs["config_history"], f"{current_date}_{current_time}_config.json")
    config = read_and_copy_config(config_file, config_copy_path)
    
    # ✅ Secure GPU/CPU!
    device = lock_and_load(config)

    # ✅ Build DataModule (without optimizer first)
    datamodule = build_data_module(config=config, 
                                   er=er,
                                   root_dir=data_root_dir,
                                   root_dir_corsika=data_root_dir_corsika)
    # ✅ Build Model
    model = build_model(config=config, 
                        device=device,)

    # ✅ Build Optimizer (after DataModule setup to get train_dataloader_length)
    optimiser_and_scheduler = build_optimiser_and_scheduler(config=config, 
                                model=model, 
                                datamodule=datamodule)

    # ✅ Assign optimizer to DataModule
    model.set_optimiser(optimiser_and_scheduler)

    # ✅ Build Callbacks
    callbacks = build_callbacks(config=config, callback_dir=dirs["checkpoint_dir"])

    # ✅ Initialize WandB Logger
    wandb.init(project=project_name, config=config, name=current_time)
    wandb_logger = WandbLogger(project=project_name, config=config)

    # ✅ Log Training Parameters
    log_training_parameters(config=config)

    # ✅ Initialize Trainer and Fit
    trainer = pl.Trainer(
        max_epochs=config['n_epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config['gpu'],
        log_every_n_steps=100, 
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
        precision="bf16-mixed"
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    training_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(training_dir, "config")
    
    config_file = "config_2203x_multiflavour.json"
    # config_file = "config_2203x_signal_noise.json"
    # config_file = "config_2203x_track_cascade.json"
    # # config_file  = "config_multiflavour_mse.json"
    # config_file  = "config_multiflavour_ce.json"
    
    # data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered_second_round/Snowstorm/CC_CRclean_IntraTravelDistance_250"
    # data_root_dir_corsika = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_second/Corsika"
    
    # 32 features
    # data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_IntraTravelDistance_250"
    # data_root_dir_corsika = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Corsika"
    
    # 35 features with containment condition
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered_second_round/Snowstorm/CC_CRclean_Contained"
    data_root_dir_corsika = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_second/Corsika_Contained"
    
    er = EnergyRange.ER_100_TEV_100_PEV
    
    print(f"data_root_dir: {data_root_dir}")
    print(f"data_root_dir_corsika: {data_root_dir_corsika}")
    print(f"energy range: {er.string}")
    
    start_time = time.time()
    run_training(config_dir=config_dir,
                    config_file=os.path.join(config_dir, config_file),
                    training_dir=training_dir,
                    data_root_dir=data_root_dir,
                    data_root_dir_corsika=data_root_dir_corsika,
                    er=er)
    end_time = time.time()
    print(f"Training completed in {time.strftime('%d:%H:%M:%S', time.gmtime(end_time - start_time))}")

# squeue -w node071,node072,node161,node162 -o "%.18i %.8u %.2t %.10M %.6D %R"