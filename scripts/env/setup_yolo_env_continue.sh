#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo || true
echo "[yolo-env] torch check:"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
echo "[yolo-env] installing ultralytics>=8.4.0 ..."
pip install -q "ultralytics>=8.4.0" 2>&1 | tail -5
python -c "import ultralytics, torch; print('[yolo-env] ultralytics', ultralytics.__version__, '| torch', torch.__version__)"
echo "[yolo-env] done"