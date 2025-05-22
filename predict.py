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
from Enum.LossType import LossType

from InferenceUtil import plot_all_metrics, extend_extract_metrics_for_all_flavours
import sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Script with Timestamped Logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    parser.add_argument("--checkpoint_date", type=str, required=True, help="Date of the checkpoint in YYYYMMDD format")
    parser.add_argument("--checkpoint_time", type=str, required=True, help="Time of the checkpoint in HHMMSS format")
    # add an optionl argument --runID where it can be empty
    parser.add_argument("--runID", type=str, default="", help="Run ID for the prediction")
    
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
    loss_type = LossType.from_string(config['loss'])
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
        loss_type=loss_type,
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

def build_predictions(config: dict, predictions: list, prediction_dir: str, ckpt_file: str):
    all_preds = {
        "target_class": [],
        "pred_class": [],
        "target_one_hot_pid": [],
        "pred_one_hot_pid": [],
        "model_outputs": [],
    }

    num_class = ClassificationMode.from_string(config['classification_mode']).num_classes
    for i, batch in enumerate(predictions):
        model_outputs = batch['model_outputs']
        targets = batch.get('target', None)

        if not isinstance(model_outputs, torch.Tensor):
            model_outputs = torch.tensor(model_outputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        pred_class = torch.argmax(model_outputs, dim=-1)
        target_class = torch.argmax(targets, dim=-1)

        all_preds["pred_class"].extend(pred_class.tolist())
        all_preds["target_class"].extend(target_class.tolist())

        pred_one_hot = torch.nn.functional.one_hot(pred_class, num_classes=num_class).tolist()
        target_one_hot = torch.nn.functional.one_hot(target_class, num_classes=num_class).tolist()

        all_preds["pred_one_hot_pid"].extend(pred_one_hot)
        all_preds["target_one_hot_pid"].extend(target_one_hot)
        all_preds["model_outputs"].extend(model_outputs.tolist())

    # Construct dataframe
    df = pd.DataFrame({
        "target_class": all_preds["target_class"],
        "pred_class": all_preds["pred_class"],
        "target_one_hot_pid": all_preds["target_one_hot_pid"],
        "pred_one_hot_pid": all_preds["pred_one_hot_pid"],
        "model_outputs": all_preds["model_outputs"],
    })
    if config['loss'] == "mse":
        # model_outputs = torch.tensor(df['model_outputs'].tolist())
        # model_outputs = torch.clamp(model_outputs, min=0) # ensure non-negative
        # probs = model_outputs / model_outputs.sum(dim=-1, keepdim=True) # manual normalisation
        # df['prob'] = probs.tolist()
        model_outputs = torch.tensor(df['model_outputs'].tolist())
        model_outputs = torch.clamp(model_outputs, min=0)

        row_sums = model_outputs.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1  # prevent division by zero

        probs = model_outputs / row_sums
        df['prob'] = probs.tolist()

    elif config['loss'] == "tau":
        # model_outputs = torch.tensor(df['mode
        model_outputs = torch.tensor(df['model_outputs'].tolist())
        model_outputs = torch.clamp(model_outputs, min=0)

        row_sums = model_outputs.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1  # prevent division by zero

        probs = model_outputs / row_sums
        df['prob'] = probs.tolist()

    elif config['loss'] == "ce":
        model_outputs = torch.tensor(df['model_outputs'].tolist()) # logits
        probs = torch.nn.functional.softmax(model_outputs, dim=-1) # softmax
        df['prob'] = probs.tolist()
    return df

def build_analysis_df(test_dataset):
    """Extracts the analysis columns from the unbatched dataset (no collate interference)."""
    analysis_list = []

    for idx in range(len(test_dataset)):
        _, _, analysis = test_dataset[idx]
        analysis_list.append(list(analysis))

    analysis_columns = MonoFlavourDataset.IDENTIFICATION + MonoFlavourDataset.ANALYSIS
    return pd.DataFrame(analysis_list, columns=analysis_columns)


def save_predictions(df_predictions: pd.DataFrame, df_analysis: pd.DataFrame, prediction_dir: str, ckpt_file: str):
    # Determine filename
    epoch, val_value, val_name = parse_checkpoint_name(ckpt_file)
    if val_name == "val_loss":
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}_val_loss_{val_value}.csv")
    elif val_name == "val_tau_lg_085_tau":
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}_tau085_{val_value}.csv")
    else:
        csv_name = os.path.join(prediction_dir, f"predictions_epoch_{epoch}.csv")

    # Combine predictions and analysis
    if len(df_predictions) != len(df_analysis):
        raise ValueError(f"Mismatch: {len(df_predictions)} predictions vs {len(df_analysis)} analysis rows.")

    df_combined = pd.concat([df_predictions, df_analysis], axis=1)

    # Optional: Uniqueness check
    if "event_no" in df_combined.columns:
        df_combined["event_no"] = df_combined["event_no"].astype(int)
        n_unique = df_combined["event_no"].nunique()
        n_total = len(df_combined)
        print(f"🧠 Unique event_no: {n_unique} / {n_total} total rows")

        dupes = df_combined[df_combined["event_no"].duplicated(keep=False)]
        if not dupes.empty:
            print(f"⚠️ {len(dupes)} duplicated rows found based on event_no!")
            print(dupes["event_no"].value_counts().head())
        else:
            print("✅ All event_no values are unique.")

    # Save to CSV
    df_combined.to_csv(csv_name, index=False)
    print(f"✅ Predictions saved to:\n{csv_name}")
    return df_combined
    


def parse_checkpoint_name(ckpt_file: str):
    """Parse checkpoint filenames and return epoch, metric value, and metric name."""
    ckpt_name = os.path.basename(ckpt_file).replace(".ckpt", "")

    if ckpt_name.startswith("last"):
        return ckpt_name, "last", "last"

    match = re.match(r"epoch=(\d+).*?([a-zA-Z0-9_]+)=([\d.]+)$", ckpt_name)
    if match:
        return match.group(1), match.group(3), match.group(2)

    match_simple = re.match(r"epoch=(\d+)$", ckpt_name)
    if match_simple:
        return match_simple.group(1), "none", "none"

    match_keep = re.match(r"epoch=(\d+)_keep", ckpt_name)
    if match_keep:
        return match_keep.group(1), "keep", "keep"

    # Match mid-epoch files like "17-mid" or "20-mid"
    match_mid = re.match(r"(\d+)-mid", ckpt_name)
    if match_mid:
        return match_mid.group(1), "mid", "mid"

    raise ValueError(f"Unrecognised checkpoint filename: {ckpt_name}")

def extract_epoch(ckpt_filename: str) -> int:
    """Extract the integer epoch number from a checkpoint filename."""
    ckpt_name = os.path.basename(ckpt_filename).replace(".ckpt", "")

    if ckpt_name == "last":
        epoch = 49
    elif ckpt_name.startswith("epoch="):
        match = re.match(r"epoch=(\d+)", ckpt_name)
        if match:
            epoch = int(match.group(1))
        else:
            raise ValueError(f"Invalid format after 'epoch=': {ckpt_filename}")
    elif "-mid" in ckpt_name:
        match = re.match(r"(\d+)-mid", ckpt_name)
        if match:
            epoch = int(match.group(1))
        else:
            raise ValueError(f"Invalid '-mid' format: {ckpt_filename}")
    else:
        raise ValueError(f"Unrecognised checkpoint format: {ckpt_filename}")

    return epoch

################################## MAIN FUNCTION ##################################
def run_prediction(config_dir: str, 
                base_dir: str, 
                data_root_dir: str, 
                data_root_dir_corsika: str,
                er: EnergyRange):
    args = parse_args()
    current_date, current_time = args.date, args.time
    local_checkpoint = args.checkpoint_date
    specific_checkpoint = args.checkpoint_time
    
    if args.runID:
        model_id = f"{args.runID}"
    else:
        model_id = f"{args.checkpoint_date}_{args.checkpoint_time}"
    print(f"Model ID: {model_id}")
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
    
    df_analysis = build_analysis_df(datamodule.test_dataloader().dataset)
    
    summary_metrics = []

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
        # returned shape: [batch_size, num_classes]

        print("💾 Saving predictions...")
        df_predictions = build_predictions(config=config,
                                            predictions=predictions,
                                            prediction_dir=dirs["predict_dir"],
                                            ckpt_file=ckpt_file_dir)
        
        df_combined = save_predictions(df_predictions=df_predictions,
                          df_analysis=df_analysis,
                          prediction_dir=dirs["predict_dir"],
                          ckpt_file=ckpt_file_dir)
        # Create plot filename based on checkpoint
        epoch = extract_epoch(ckpt_file)
        plot_dir = os.path.join("Plot_inference", model_id)
        os.makedirs(plot_dir, exist_ok=True)

        pdf_file = os.path.join(plot_dir, f"{epoch}.pdf")
        plot_all_metrics(df_combined, pdf_path=pdf_file, run_id=model_id, epoch=epoch)
        metrics = extend_extract_metrics_for_all_flavours(df_combined, run_id=model_id, epoch=epoch)
        summary_metrics.append(metrics)
        print("✅ Predictions saved successfully.")
    summary_df = pd.DataFrame(summary_metrics)
    summary_path = os.path.join("Plot_inference", model_id, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    print("✅ All predictions completed successfully.")
        
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
    