#!/usr/bin/env bash
# Tuned YOLO26s training on 2K two-wheeler data (tw_run_v3).
# Tuning vs v1/v2: batch 32, lr0 0.005, freeze 10 (backbone), epochs 150, patience 50.
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
DATA=/mnt/c/Data/datasets/yolo_detect_tw2k/data.yaml
PROJ=/home/xu/data/yolo_runs
NAME=tw_run_v3
"$PY" - "$DATA" "$PROJ" "$NAME" <<'PY'
import sys
from ultralytics import YOLO

data, proj, name = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[tw-v3] data={data} project={proj} name={name}")
model = YOLO("yolo26s.pt")
model.train(
    data=data,
    epochs=150,
    imgsz=640,
    batch=32,
    device=0,
    project=proj,
    name=name,
    patience=50,
    lr0=0.005,
    freeze=10,
    cache=True,
    workers=4,
    seed=42,
    exist_ok=True,
)
PY