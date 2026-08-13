#!/usr/bin/env bash
# YOLO26s fine-tune on the corrected detect_v2 dataset (feedback-loop v2 GT).
# Env: WSL conda env `yolo` (ultralytics >= 8.4). Run from anywhere.
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
DATA=/mnt/c/Data/datasets/yolo_detect/data.yaml
PROJ=/home/xu/yolo_runs
NAME="${1:-run_v1}"
"$PY" - "$DATA" "$PROJ" "$NAME" <<'PY'
import sys
from ultralytics import YOLO

data, proj, name = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[yolo-train] data={data} project={proj} name={name}")
model = YOLO("yolo26s.pt")  # COCO-pretrained
model.train(
    data=data,
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    project=proj,
    name=name,
    patience=100,
    cache=True,
    workers=4,
    seed=42,
    exist_ok=True,
)
PY