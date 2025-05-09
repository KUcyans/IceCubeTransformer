import time
import json
import re
import os
import torch
import logging
import argparse
import pandas as pd
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import TQDMProgressBar
from Model.FlavourClassificationTransformerEncoder import FlavourClassificationTransformerEncoder
from VernaDataSocket.MultiFlavourDataModule import MultiFlavourDataModule
from VernaDataSocket.MonoFlavourDataset import MonoFlavourDataset
from Enum.EnergyRange import EnergyRange
from Enum.Flavour import Flavour
from Enum.ClassificationMode import ClassificationMode
from Enum.AttentionType import AttentionType
from Enum.PositionalEncodingType import PositionalEncodingType

import sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Script with Timestamped Logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    parser.add_argument("--checkpoint_date", type=str, required=True, help="Date of the checkpoint in YYYYMMDD format")
    parser.add_argument("--checkpoint_time", type=str, required=True, help="Time of the checkpoint in HHMMSS format")
    
    return parser.parse_args()

def setup_directories(base_dir: str, config_dir: str, current_date: str, current_time: str, checkpoint_date: str, checkpoint_time: str):
    paths = {
        "log_dir": os.path.join(base_dir, "logs", current_date),
        "predict_dir": os.path.join(base_dir, "predictions", current_date, f"model_{checkpoint_date}_{checkpoint_time}", current_time),
        "checkpoint_dir": os.path.join(base_dir, "checkpoints", checkpoint_date, checkpoint_time),
        "config_history": os.path.join(config_dir, "history"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return {
        **paths,
        "predict_log_file": os.path.join(paths["log_dir"], f"{current_time}_predict.log"),
    }

def lock_and_load(config):
    """Set CUDA device based on config['gpu'] if available, else fallback to any available GPU, else CPU."""
    print("torch.cuda.is_available():", torch.cuda.is_available())
    available_devices = list(range(torch.cuda.device_count()))
    print(f"Available CUDA devices: {available_devices}")

    requested_gpus = config.get('gpu', [])

    if torch.cuda.is_available() and available_devices:
        # Try to use one of the requested GPUs if it's available
        usable_gpus = [int(g) for g in requested_gpus if int(g) in available_devices]
        selected_gpu = usable_gpus[0] if usable_gpus else available_devices[0]

        torch.cuda.empty_cache()
        print("🔥 LOCK AND LOAD! GPU ENGAGED! 🔥")
        device = torch.device(f"cuda:{selected_gpu}")
        torch.cuda.set_device(selected_gpu)
        torch.set_float32_matmul_precision('highest')
        print(f"Using GPU: {selected_gpu} (cuda:{selected_gpu})")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available or no GPUs detected. Using CPU.")

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

def load_model_config(dirs, checkpoint_date, checkpoint_time):
    config_file = os.path.join(dirs["config_history"], f"{checkpoint_date}_{checkpoint_time}_config.json")
    print(f"Loading config from {config_file}...")
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)
    print(f" d_input: {config['d_input']}")
    return config


def build_model(config: dict, device: torch.device, ckpt_file: str):
    """Load model from checkpoint."""
    classification_mode = ClassificationMode.from_string(config['classification_mode'])
    num_classes = classification_mode.num_classes
    attention_type = AttentionType.from_string(config['attention'])
    positional_encoding_type = PositionalEncodingType.from_string(config['positional_encoding'])
    model = FlavourClassificationTransformerEncoder.load_from_checkpoint(
        checkpoint_path=ckpt_file,
        strict=False,
        d_model=config['embedding_dim'],
        n_heads=config['n_heads'],
        d_f=config['embedding_dim'] * 4,
        num_layers=config['n_layers'],
        d_input=config['d_input'],
        n_output_layers=config['n_output_layers'],
        num_classes=num_classes,
        seq_len=config['event_length'],
        attention_type=attention_type,
        positional_encoding_type=positional_encoding_type,
        dropout=config['dropout'],
        map_location=device
    )
    return model


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
    datamodule.setup(stage="predict")
    return datamodule

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

def save_predictions(config: dict, predictions: torch.Tensor, prediction_dir: str, ckpt_file: str):
    pred_classes = []
    target_classes = []
    pred_one_hot = []
    target_one_hot = []
    logits_list = []
    probs_list = []
    analysis_list = []

    num_class = ClassificationMode.from_string(config['classification_mode']).num_classes

    for i in range(len(predictions)):
        logit = predictions[i]['logits']
        prob = predictions[i]['probs']
        target = predictions[i].get('target', None)
        analysis = predictions[i].get("analysis", None)

        pred_class = torch.argmax(prob, dim=-1) #
        target_class = torch.argmax(target, dim=-1)

        pred_one_hot_vec = torch.nn.functional.one_hot(pred_class, num_classes=num_class).tolist()
        target_one_hot_vec = torch.nn.functional.one_hot(target_class, num_classes=num_class).tolist()

        pred_classes.append(pred_class)
        pred_one_hot.extend(pred_one_hot_vec)

        target_classes.append(target_class)
        target_one_hot.extend(target_one_hot_vec)

        logits_list.extend(logit.tolist())
        probs_list.extend(prob.tolist())

        analysis_list.extend(analysis.cpu().numpy().tolist())

    pred_classes = torch.cat(pred_classes, dim=0)
    target_classes = torch.cat(target_classes, dim=0)

    print('Predictions shape:', pred_classes.shape)
    print('Targets shape:', target_classes.shape)
    
    df = pd.DataFrame({
        
        "target_class": target_classes.numpy(),
        "pred_class": pred_classes.numpy(),
        "target_one_hot_pid": target_one_hot,
        "pred_one_hot_pid": pred_one_hot,
        "logits": logits_list,
        "probs": probs_list,
    })
    analysis_columns = MonoFlavourDataset.IDENTIFICATION + MonoFlavourDataset.ANALYSIS
    analysis_df = pd.DataFrame(analysis_list, columns=analysis_columns)
    df = pd.concat([df, analysis_df], axis=1)

    epoch, val_value, val_name = parse_checkpoint_name(ckpt_file)

    if val_name == "val_loss":
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}_val_loss_{val_value}.csv")
    elif val_name == "val_tau_lg_085_tau":
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}_tau085_{val_value}.csv")
    else:
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}.csv")
    df.to_csv(csv_name, index=False)
    print(f"Predictions saved to.. \n{csv_name}")

def parse_checkpoint_name(ckpt_file: str):
    """Parse checkpoint filenames and return epoch, metric value, and metric name."""
    ckpt_name = os.path.basename(ckpt_file).replace(".ckpt", "")

    if ckpt_name.startswith("last"):
        return ckpt_name, "last", "last"

    # Match: epoch=46_tau_085=val_tau_lg_085_tau=0.5472
    match = re.match(r"epoch=(\d+).*?([a-zA-Z0-9_]+)=([\d.]+)$", ckpt_name)
    if match:
        return match.group(1), match.group(3), match.group(2)

    match_simple = re.match(r"epoch=(\d+)$", ckpt_name)
    if match_simple:
        return match_simple.group(1), "none", "none"

    match_keep = re.match(r"epoch=(\d+)_keep", ckpt_name)
    if match_keep:
        return match_keep.group(1), "keep", "keep"

    raise ValueError(f"Unrecognised checkpoint filename: {ckpt_name}")

def run_prediction(config_dir: str, 
                base_dir: str, 
                data_root_dir: str, 
                data_root_dir_corsika: str,
                er: EnergyRange):
    args = parse_args()
    current_date, current_time = args.date, args.time
    local_checkpoint = args.checkpoint_date
    specific_checkpoint = args.checkpoint_time
    
    dirs = setup_directories(base_dir = base_dir, 
                             config_dir = config_dir,
                             current_date = current_date, 
                             current_time = current_time, 
                             checkpoint_date = local_checkpoint,
                             checkpoint_time = specific_checkpoint)
    # predict_logger = setup_logger("predict", dirs["predict_log_file"])
    config = load_model_config(dirs, local_checkpoint, specific_checkpoint)
    
    device = lock_and_load(config)
    
    datamodule = build_data_module(config=config,
                                    root_dir=data_root_dir,
                                    root_dir_corsika=data_root_dir_corsika,
                                    er=er)
    
    callbacks = build_callbacks()
    log_training_parameters(config)
    
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[device.index] if device.type == 'cuda' else 1,
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
        model.to(device)

        print("🚀 Running predictions...")
        predictions = trainer.predict(model=model, dataloaders=datamodule.test_dataloader())

        print("💾 Saving predictions...")
        save_predictions(config=config, 
                         predictions=predictions,
                        prediction_dir=dirs["predict_dir"],
                        ckpt_file=ckpt_file_dir)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(base_dir, "config")
    
    data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered_second_round/Snowstorm/CC_CRclean_IntraTravelDistance_250"
    # data_root_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered_second_round/Snowstorm/CC_CRclean_IntraTravelDistance_250m"
    data_root_dir_corsika = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_second/Corsika"
    
    er = EnergyRange.ER_100_TEV_100_PEV
    
    print(f"data_root_dir: {data_root_dir}")
    print(f"data_root_dir_corsika: {data_root_dir_corsika}")
    print(f"energy range: {er.string}")
    
    start_time = time.time()
    run_prediction(config_dir=config_dir,
                 base_dir=base_dir,
                 data_root_dir=data_root_dir,
                data_root_dir_corsika=data_root_dir_corsika,
                 er=er)
    end_time = time.time()
    # elapsed time in HH:MM:SS with full two digits
    print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    