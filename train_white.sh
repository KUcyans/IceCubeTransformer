#!/bin/bash

# Define log directory and filenames
LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")
timestamp=$(date +"%H%M%S")

mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_execution.log"

# Redirect output to logfile
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Training script"
echo "Checking allocated GPU..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer

# Ensure CUDA is set up properly in WSL2
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Run the script using an interactive bash shell to ensure Conda is activated
nohup bash -i -c "python -u train_white.py --date \"$datestamp\" --time \"$timestamp\"" > "${logfile}" 2>&1 &

# Capture the PID of the background process
JOB_PID=$!
echo "Job started with PID: $JOB_PID"

echo "Job is running in the background. Log file: ${logfile}"
disown
