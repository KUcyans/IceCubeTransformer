#!/bin/bash
#SBATCH --job-name=benchmark_script_%j
#SBATCH --partition=icecube_gpu
#SBATCH --ntasks=1
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Create output directory if not exists
mkdir -p benchmark_result

# Format timestamp as YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H:%M:%S")

# Set the log file path
logfile="benchmark_result/[${timestamp}]benchmarking.log"

# Redirect both stdout and stderr to the log file
exec > "${logfile}" 2>&1

echo "=== Starting Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node Name: $(hostname)"
echo "CPU Info: $(lscpu | grep 'Model name')"
echo "Total CPUs: $(nproc)"
echo "Memory Info: $(free -h)"
echo "Date and Time: $(date)"

# Ensure we are in the correct directory
cd /groups/icecube/cyan/factory/IceCubeTransformer

# Run the Python script
python -u benchmarker.py

echo "=== Benchmark Completed ==="


# commandline to run the script
# timestamp=$(date +"%Y%m%d_%H%M%S")
# nohup python -u benchmarker.py > benchmark_result/[${timestamp}]_benchmarking.log 2>&1 &