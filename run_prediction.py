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
    
    return parser.parse_args()

def setup_directories(base_dir: str, current_date: str, current_time: str, checkpoint_date: str):
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "predict_dir": os.path.join(base_dir, "predictions", current_date, f"model_{checkpoint_date}"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return {
        **paths,
        "predict_log_file": os.path.join(paths["log_dir"], f"{current_time}_predict.log"),
    }

def lock_and_load(config):
    """Set CUDA device based on config['gpu'] if available, else use CPU."""
    print("CUDA_VISIBLE_DEVICES (before):", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    # Set CUDA devices from config
    if torch.cuda.is_available() and len(config.get('gpu', [])) > 0:
        print("🔥 LOCK AND LOAD! GPU ENGAGED! 🔥")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config['gpu']))
        torch.cuda.set_device(int(config['gpu'][0]))
        device = torch.device('cuda')
        torch.set_float32_matmul_precision('highest')
        print(f"Using GPU(s): {config['gpu']}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    print("CUDA_VISIBLE_DEVICES (after):", os.environ.get("CUDA_VISIBLE_DEVICES"))
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
                train_logger: logging.Logger, 
                device: torch.device,
                ckpt_file: str = None
                ):
    model = FlavourClassificationTransformerEncoder.load_from_checkpoint(
        checkpoint_path=ckpt_file,
        strict = False,
        d_model=config['embedding_dim'],
        n_heads=config['n_heads'],
        d_f=config['embedding_dim'],
        num_layers=config['n_layers'],
        d_input= config['d_input'],
        num_classes=config['output_dim'],
        seq_len=config['event_length'],
        attention_type=config['attention'],
        dropout=config['dropout'],
        train_logger=train_logger,
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


def build_callbacks():
    callbacks = [
        TQDMProgressBar(refresh_rate=1000),
    ]   
    return callbacks

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

    message = """\n
    | Parameter       | Value               |
    |-----------------|---------------------|
    """ + "".join([f"| {k:<30} | {str(v):<20} |\n" for k, v in config_flattened.items()])

    training_logger.info("The training parameters:")
    training_logger.info(message)


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

    csv_name = os.path.join(prediction_dir, f"predictions_{os.path.splitext(ckpt_file)[0]}.csv")
    df.to_csv(csv_name, index=False)
    print(f"Predictions saved to {csv_name}.")


def run_prediction(config_file: str, 
                   prediction_dir: str, 
                   checkpoint_dir: str,
                   data_root_dir: str):
    args = parse_args()
    current_date, current_time = args.date, args.time
    specific_checkpoint = args.checkpoint_date
    
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    dirs = setup_directories(prediction_dir, current_date, current_time, specific_checkpoint)
    predict_logger = setup_logger("predict", dirs["predict_log_file"])
    
    device = lock_and_load(config)
    
    datamodule = build_data_module(config=config,
                                    root_dir=data_root_dir)
    
    callbacks = build_callbacks()
    log_training_parameters(config, predict_logger)
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config['gpu'],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=None,
    )
    
    specific_checkpoint_dir = os.path.join(checkpoint_dir, specific_checkpoint)
    ckpt_files = [f for f in os.listdir(specific_checkpoint_dir) if f.endswith(".ckpt")]

    for ckpt_file in ckpt_files:
        ckpt_file_dir = os.path.join(specific_checkpoint_dir, ckpt_file)

        if not ckpt_file.endswith(".ckpt"):
            continue
        ckpt_file_dir = os.path.join(specific_checkpoint_dir, ckpt_file)

        print(f"\n🔥 Loading model from {ckpt_file}...")
        model = build_model(config=config, train_logger=predict_logger, device=device, ckpt_file=ckpt_file_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
        model.set_optimiser(optimizer)
        model.to(device)

        print("🚀 Running predictions...")
        predictions = trainer.predict(model=model, dataloaders=datamodule.predict_dataloader())

        print("💾 Saving predictions...")
        save_predictions(predictions, dirs["predict_dir"], ckpt_file)


if __name__ == "__main__":
    config_dir = "/groups/icecube/cyan/factory/IceCubeTransformer/config/"
    config_file = "config_predict.json"
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/"
    prediction_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(prediction_dir, "checkpoints")
    
    start_time = time.time()
    run_prediction(config_file=os.path.join(config_dir, config_file),
                 prediction_dir=prediction_dir,
                 checkpoint_dir=checkpoint_dir,
                 data_root_dir=data_root_dir)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
    