#!/bin/bash
#SBATCH --job-name=train_script_%j
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=15:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_DIR="log"
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/${timestamp}_prediction.log"
exec > /dev/null 2> "${logfile}"

echo "Starting job at $(date)"
echo "training script"
echo "Checking allocated GPU..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python3 -u inference.py

echo "Job completed at $(date)"
