#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locate_anything
echo "=== installing prebuilt flash-attn wheel from Astral cu130 index ==="
pip install "flash-attn==2.8.3+cu.13.0.torch.2.9" --index-url "https://wheels.astral.sh/simple/cu130/" --no-deps --no-cache-dir
echo "=== verify ==="
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "WHEEL_INSTALL_OK"