#!/bin/bash
#SBATCH --job-name=build_torch
#SBATCH --partition=icecube_gpu
#SBATCH --nodelist=node161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_ROOT_DIR="logs"
datestamp=$(date +"%Y%m%d")   # ✅ Removed spaces around '='
timestamp=$(date +"%H%M%S")   # ✅ Removed spaces around '='

# ✅ Create directories properly
mkdir -p "${LOG_ROOT_DIR}/${datestamp}"
logfile="${LOG_ROOT_DIR}/${datestamp}/${timestamp}_torch_build.log"

exec > "${logfile}" 2>&1
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Load environment
source /groups/icecube/cyan/miniconda3/etc/profile.d/conda.sh
conda activate icecube_transformer_debug

# CUDA environment
export CUDA_HOME=/groups/icecube/cyan/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_GENERATOR=Ninja
export CMAKE_MAKE_PROGRAM=$(which ninja)
export USE_CUDA=1
export MAX_JOBS=8
export USE_NUMA=OFF

# Move to the PyTorch build directory
cd /groups/icecube/cyan/.local/lib/python3.12/site-packages/pytorch

# (Re-)generate build config if needed (optional)
# python tools/setup_helpers/generate_code.py
# python setup.py build_ext --cmake-only

# Build it!
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_NUMA=OFF \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
  -DCMAKE_GENERATOR=${CMAKE_GENERATOR} \
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}

cmake --build . --target install -- -j$MAX_JOBS
