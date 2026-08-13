#!/usr/bin/env bash
# Create the YOLO-dedicated conda env on WSL (isolated from LocateAnything envs).
# Env name: yolo  (cloned from locate_anything_sft so torch 2.9+cu130 / py3.10 are reused)
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
if conda env list | grep -qE '(^| )yolo( |$)'; then
    echo "[yolo-env] env 'yolo' already exists"
else
    echo "[yolo-env] cloning locate_anything_sft -> yolo"
    conda create --clone locate_anything_sft -n yolo -y
fi
conda activate yolo
echo "[yolo-env] torch check:"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
echo "[yolo-env] installing ultralytics>=8.4.0 ..."
pip install -q "ultralytics>=8.4.0"
python -c "import ultralytics; print('[yolo-env] ultralytics', ultralytics.__version__)"
echo "[yolo-env] done"