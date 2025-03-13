#!/bin/bash
#SBATCH --job-name=train_script_%j
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=120:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ✅ Fix variable assignment (no spaces around '=')
LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")   # ✅ Removed spaces around '='
timestamp=$(date +"%H%M%S")   # ✅ Removed spaces around '='

# ✅ Create directories properly
mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_execution.log"

# ✅ Redirect stdout and stderr
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Training script"
echo "Checking allocated GPU..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer
python -u run_training.py --date "$datestamp" --time "$timestamp"


echo "Job completed at $(date)"

EMAIL_SUBJECT="SLURM Job Notification: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
EMAIL_BODY="Job Details:\n\
- Job Name: ${SLURM_JOB_NAME}\n\
- Job ID: ${SLURM_JOB_ID}\n\
- Partition: ${SLURM_JOB_PARTITION}\n\
- Status: ${SLURM_JOB_STATUS}\n\
- Completed At: $(date)\n\n\
Log Snippet:\n$(tail -n 100 "${logfile}")"

echo -e "${EMAIL_BODY}" | mailx -s "${EMAIL_SUBJECT}" cyan.jo@proton.me
