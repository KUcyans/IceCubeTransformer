#!/bin/bash
#SBATCH --job-name=check_node_%j
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=check_node.out
#SBATCH --error=/dev/null

nvidia-smi
nvcc --version
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
