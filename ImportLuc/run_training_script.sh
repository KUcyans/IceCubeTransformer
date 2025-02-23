#!/bin/bash
#SBATCH --job-name=train_script_%j
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me

# Define log directory and filename
LOG_DIR="log"
mkdir -p "${LOG_DIR}"  # Ensure the directory exists
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/${timestamp}_execution.log"

# Redirect output and errors to logfile
exec > "${logfile}" 2>&1

# Print job start time
echo "Starting job at $(date)"
echo "Training script"
echo "Checking allocated GPU..."

# GPU information
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate Conda environment
source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer
echo "Conda environment activated: $CONDA_PREFIX"

# Run training script
echo "Starting Python script"
python -u training_script.py

# Print job completion time
echo "Job completed at $(date)"
