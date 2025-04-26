#!/bin/bash
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

CHECKPOINT_DATE="20250426"
CHECKPOINT_TIME="225654"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer

nohup python -u run_prediction.py --date "$datestamp" --time "$timestamp" \
                            --checkpoint_date "$CHECKPOINT_DATE"\
                            --checkpoint_time "$CHECKPOINT_TIME"\
                             > "${logfile}" 2>&1 &

echo "Prediction job completed at $(date)"
disown