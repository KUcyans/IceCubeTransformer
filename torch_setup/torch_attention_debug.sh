#!/bin/bash
#SBATCH --job-name=attention_debug%j
#SBATCH --partition=gr10_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node071
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ✅ Fix variable assignment (no spaces around '=')
LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")   # ✅ Removed spaces around '='
timestamp=$(date +"%H%M%S")   # ✅ Removed spaces around '='

# ✅ Create directories properly
mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_attention_debug.log"

# ✅ Redirect stdout and stderr
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Training script"
echo "Checking allocated GPU..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer_debug

# ✅ Set manual torch and CUDA paths
export CUDA_HOME=/groups/icecube/cyan/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/groups/icecube/cyan/.local/lib/python3.12/site-packages/pytorch
find $PYTHONPATH -name "*.so" | grep cuda || echo "No CUDA .so found in path"
echo "Python path: $PYTHONPATH"
echo "Torch backend shared objects:"

# ✅ Run your sanity test
python -u torch_import.py

echo "Job completed at $(date)"