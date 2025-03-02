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
source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer

# Run the script with nohup and disown to detach
nohup python -u run_training.py --date "$datestamp" --time "$timestamp" > "${logfile}" 2>&1 &

# Capture the PID of the background process
JOB_PID=$!
echo "Job started with PID: $JOB_PID"

echo "Job is running in the background. Log file: ${logfile}"
disown
