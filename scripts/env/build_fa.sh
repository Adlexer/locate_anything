#!/bin/bash
# Build flash-attn from source inside the locate_anything conda env.
# - CUDA_VISIBLE_DEVICES="" keeps torch from touching the WSL GPU driver during setup.py.
# - MAX_JOBS=1 limits nvcc/cicc parallelism: flash-attn spawns many cicc subprocesses
#   (~1GB RSS each) which OOM-killed the WSL VM at higher job counts.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locate_anything
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=1
export NVCC_THREADS=2
export CUDA_VISIBLE_DEVICES=""
echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_avail', torch.cuda.is_available())"
nvcc --version | tail -1
free -g | head -2
echo "=== installing flash-attn (MAX_JOBS=1) ==="
pip install ~/flash_attn-2.8.3.post1.tar.gz --no-build-isolation --no-cache-dir
echo "=== verify ==="
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "BUILD_OK"
