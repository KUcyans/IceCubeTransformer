#!/bin/bash
#SBATCH --job-name=torch_install
#SBATCH --partition=gr10_gpu
#SBATCH --nodelist=node071
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")
timestamp=$(date +"%H%M%S")
mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_torch_install.log"
exec > "${logfile}" 2>&1

echo "[INFO] Starting job at $(date)"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer_debug

export CUDA_HOME=/groups/icecube/cyan/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/groups/icecube/cyan/.local/lib/python3.12/site-packages/pytorch

cd /groups/icecube/cyan/.local/lib/python3.12/site-packages/pytorch

echo "[INFO] Running pip install -e ."
pip install -e .

echo "[INFO] Finished at $(date)"
