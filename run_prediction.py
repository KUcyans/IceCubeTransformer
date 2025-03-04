import time
import json
import os
import torch
import logging
import argparse
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
from SnowyDataSocket.MultiPartDataModule import MultiPartDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Script with Timestamped Logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    parser.add_argument("--checkpoint_date", type=str, required=True, help="Date of the checkpoint in YYYYMMDD format")
    parser.add_argument("--checkpoint_time", type=str, required=True, help="Time of the checkpoint in HHMMSS format")
    
    return parser.parse_args()

def setup_directories(base_dir: str, current_date: str, current_time: str, checkpoint_date: str, checkpoint_time: str):
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "predict_dir": os.path.join(base_dir, "predictions", current_date, f"model_{checkpoint_date}", current_time),
        "checkpoint_dir": os.path.join(base_dir, "checkpoints", checkpoint_date, checkpoint_time),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return {
        **paths,
        "predict_log_file": os.path.join(paths["log_dir"], f"{current_time}_predict.log"),
    }


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

def setup_logger(name: str, log_filename: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


def build_model(config: dict, 
                device: torch.device,
                ckpt_file: str = None
                ):
    model = FlavourClassificationTransformerEncoder.load_from_checkpoint(
        checkpoint_path=ckpt_file,
        strict = False,
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
    return model


def build_data_module(config: dict, 
                      root_dir:str):
    datamodule = MultiPartDataModule(
        root_dir=root_dir,
        subdirectory_parts_train=config['train_data'],
        subdirectory_parts_val=config['validate_data'],
        subdirectory_parts_test=config['predict_data'],
        event_length=config['event_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sample_weights_train=config.get('sample_weights'),
        sample_weights_val=config.get('sample_weights'),
    )
    datamodule.setup(stage='predict')
    return datamodule

def build_optimiser_and_scheduler(config: dict, model: torch.nn.Module, datamodule: MultiPartDataModule):
    """Build and return the optimizer and learning rate scheduler."""
    optimizer_config = config['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr_max']/optimizer_config['div_factor'],
        betas=tuple(optimizer_config['betas']),
        eps=optimizer_config['eps'],
        weight_decay=optimizer_config['weight_decay'],
        amsgrad=optimizer_config['amsgrad']
    )
    total_N_steps = config['n_epochs'] * len(datamodule.predict_dataloader())
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer_config['lr_max'],
            epochs=config['n_epochs'],
            total_steps=total_N_steps,
            pct_start=optimizer_config['pct_start'],
            div_factor=optimizer_config['div_factor'],
            max_momentum=optimizer_config['max_momentum'],
            base_momentum=optimizer_config['base_momentum'],
            final_div_factor=optimizer_config['final_div_factor'],
            anneal_strategy=optimizer_config['anneal_strategy']
        ),
        'interval': optimizer_config['interval'],
        'frequency': optimizer_config['frequency'],
    }

    return optimizer, scheduler

def build_callbacks():
    callbacks = [
        TQDMProgressBar(refresh_rate=1000),
    ]   
    return callbacks

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

    print(message)


def save_predictions(predictions: torch.Tensor, prediction_dir: str, ckpt_file: str):
    pred_classes = []
    target_classes = []
    pred_one_hot = []
    target_one_hot = []
    for i in range(len(predictions)):
        prob = predictions[i]['probs']  # ✅ Corrected key
        target = predictions[i].get('target', None)  # Optional, if available

        pred_class = torch.argmax(prob, dim=-1)
        target_class = torch.argmax(target, dim=-1)
        
        pred_one_hot_vec = torch.nn.functional.one_hot(pred_class, num_classes=3).tolist()
        target_one_hot_vec = torch.nn.functional.one_hot(target_class, num_classes=3).tolist()

        pred_classes.append(pred_class)
        pred_one_hot.extend(pred_one_hot_vec)
        
        target_classes.append(target_class)
        target_one_hot.extend(target_one_hot_vec)

    pred_classes = torch.cat(pred_classes, dim=0)
    target_classes = torch.cat(target_classes, dim=0)

    print('Predictions shape:', pred_classes.shape)  # Should be (num_samples,)
    print('Targets shape:', target_classes.shape)  # Should be (num_samples,)
    
    df = pd.DataFrame({
        "pred_one_hot_pid": pred_one_hot,
        "target_one_hot_pid": target_one_hot,
        "pred_class": pred_classes.numpy(),
        "target_class": target_classes.numpy(),
    })

    checkpoint_name = os.path.basename(ckpt_file).replace(".ckpt", "")
    csv_name = os.path.join(prediction_dir, f"predictions_{checkpoint_name}.csv")
    df.to_csv(csv_name, index=False)
    print(f"Predictions saved to.. \n{csv_name}")


def run_prediction(config_file: str, 
                   base_dir: str, 
                   data_root_dir: str):
    args = parse_args()
    current_date, current_time = args.date, args.time
    local_checkpoint = args.checkpoint_date
    specific_checkpoint = args.checkpoint_time
    
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    dirs = setup_directories(base_dir = base_dir, 
                             current_date = current_date, 
                             current_time = current_time, 
                             checkpoint_date = local_checkpoint,
                             checkpoint_time = specific_checkpoint)
    # predict_logger = setup_logger("predict", dirs["predict_log_file"])
    
    device = lock_and_load(config)
    
    datamodule = build_data_module(config=config,
                                    root_dir=data_root_dir)
    
    callbacks = build_callbacks()
    log_training_parameters(config)
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config['gpu'],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=None,
    )
    
    specific_checkpoint_dir = dirs["checkpoint_dir"]
    ckpt_files = [f for f in os.listdir(specific_checkpoint_dir) if f.endswith(".ckpt")]

    for ckpt_file in ckpt_files:
        ckpt_file_dir = os.path.join(specific_checkpoint_dir, ckpt_file)

        if not ckpt_file.endswith(".ckpt"):
            continue
        ckpt_file_dir = os.path.join(specific_checkpoint_dir, ckpt_file)

        print(f"\n🔥 Loading model from {ckpt_file}...")
        model = build_model(config=config, device=device, ckpt_file=ckpt_file_dir)
        optimizer, scheduler = build_optimiser_and_scheduler(config=config, model=model, datamodule=datamodule)
        model.set_optimiser(optimizer, scheduler)
        model.to(device)

        print("🚀 Running predictions...")
        predictions = trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())

        print("💾 Saving predictions...")
        save_predictions(predictions, dirs["predict_dir"], ckpt_file)


if __name__ == "__main__":
    config_dir = "/groups/icecube/cyan/factory/IceCubeTransformer/config/"
    config_file = "config_predict.json"
    # data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/"
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_Contained"
    base_dir = os.path.dirname(os.path.realpath(__file__))
    
    start_time = time.time()
    run_prediction(config_file=os.path.join(config_dir, config_file),
                 base_dir=base_dir,
                 data_root_dir=data_root_dir)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
    