#!/bin/bash
#SBATCH --job-name=predict_script_%j
#SBATCH --partition=gr10_gpu
##SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node072
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=10:00:00
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

CHECKPOINT_DATE="20250521"
CHECKPOINT_TIME="122145"
RUN_ID="25"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer
if [ -n "$RUN_ID" ]; then
    python -u predict.py \
        --date "$datestamp" \
        --time "$timestamp" \
        --checkpoint_date "$CHECKPOINT_DATE" \
        --checkpoint_time "$CHECKPOINT_TIME" \
        --runID "$RUN_ID"
else
    python -u predict.py \
        --date "$datestamp" \
        --time "$timestamp" \
        --checkpoint_date "$CHECKPOINT_DATE" \
        --checkpoint_time "$CHECKPOINT_TIME"
fi
echo "Prediction job completed at $(date)"
