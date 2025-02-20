import sys
import time
import platform
import torch
import psutil
from datetime import datetime
from tqdm import tqdm
import warnings

sys.path.append('/groups/icecube/cyan/factory/IceCubeTransformer')

from ImportLuc.dataset_multifile_flavours import PMTfiedDatasetPyArrow
from SnowyDataSocket.MultiPartDataset import MultiPartDataset

warnings.filterwarnings("ignore")  # Ignore all warnings

def log_system_info():
    """Log machine information including node name, CPU, memory, and date/time."""
    print("\n=== System Information ===")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Node Name: {platform.node()}")
    print(f"System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=True)} (Logical), {psutil.cpu_count(logical=False)} (Physical)")
    print(f"Total Memory: {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")
    print(f"Python Version: {platform.python_version()}")

    # Check GPU if available
    if torch.cuda.is_available():
        print("\n=== GPU Information ===")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
    else:
        print("\nNo GPU detected. Using CPU only.")


def benchmarker(dataset, name):
    start = time.perf_counter()
    for _ in tqdm(dataset, desc=name, file=sys.stdout, miniters=10_000):
        pass
    end = time.perf_counter()
    total_time = end - start
    average_event_rate = len(dataset) / total_time

    print(f"\n===================== Benchmark Results for {name} =====================")
    print(f"Dataset Length: {len(dataset)}")
    print(f"Total Time Taken: {total_time:.4f} seconds")
    print(f"Average Events per Second: {average_event_rate:.4f}")


def build_dataset_existing():
    """Build Luc's dataset."""
    train_path_1 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22012/truth_1.parquet"]
    train_path_2 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22012/truth_2.parquet"]
    train_path_3 = ["/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/22012/truth_3.parquet"]
    dataset_existing = PMTfiedDatasetPyArrow(
        truth_paths_1=train_path_1,
        truth_paths_2=train_path_2,
        truth_paths_3=train_path_3,
        sample_weights=[1, 1, 1],
        selection=None,
    )
    return dataset_existing


def build_dataset_snowy():
    """Build Snowy dataset."""
    subdirectory_parts_train = {
        "22012": [1,2,3],
    }
    dataset_snowy = MultiPartDataset(
        root_dir="/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/PureNu/",
        subdirectory_parts=subdirectory_parts_train,
        sample_weights=[1, 1, 1],
        selection=None,
    )
    return dataset_snowy


def run_benchmarker():
    """Main function to log system info and benchmark datasets."""
    print("Running log_system_info()")
    log_system_info()

    print("\n\n===================== DATA: first three parts of 22012 =====================")
    print("22012/truth_1.parquet")
    print("22012/truth_2.parquet")
    print("22012/truth_3.parquet")
    
    print("\n\n===================== BENCHMARKING NEWS =====================\n\n")
    
    print("Building Existing dataset")
    dataset_existing = build_dataset_existing()
    print("Benchmarking Existing dataset")
    benchmarker(dataset_existing, "Existing Dataset")

    print("Building Snowy dataset")
    dataset_snowy = build_dataset_snowy()
    print("Benchmarking Snowy dataset")
    benchmarker(dataset_snowy, "Snowy Dataset")


if __name__ == "__main__":
    print("Starting Benchmark Script")
    run_benchmarker()

