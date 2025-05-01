#!/bin/bash
#SBATCH --job-name=predict_script_%j
##SBATCH --partition=gr10_gpu
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
##SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ✅ Variables for logs
LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")
timestamp=$(date +"%H%M%S")

# ✅ Create directories
mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_prediction.log"

# ✅ Redirect stdout and stderr
exec > "${logfile}" 2>&1

echo "Starting prediction job at $(date)"
echo "Prediction script"
echo "Checking allocated GPU..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

CHECKPOINT_DATE="20250430"
CHECKPOINT_TIME="144939"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer
python -u predict.py --date "$datestamp" --time "$timestamp" \
                            --checkpoint_date "$CHECKPOINT_DATE" \
                            --checkpoint_time "$CHECKPOINT_TIME"

echo "Prediction job completed at $(date)"
