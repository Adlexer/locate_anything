#!/usr/bin/env bash
# YOLO26s training on human-reviewed two-wheeler GT (tw_run_v2).
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
DATA=/mnt/c/Data/datasets/yolo_detect_tw_v2/data.yaml
PROJ=/home/xu/data/yolo_runs
NAME=tw_run_v2
"$PY" - "$DATA" "$PROJ" "$NAME" <<'PY'
import sys
from ultralytics import YOLO

data, proj, name = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"[tw-v2] data={data} project={proj} name={name}")
model = YOLO("yolo26s.pt")
model.train(
    data=data,
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    project=proj,
    name=name,
    patience=50,
    cache=True,
    workers=4,
    seed=42,
    exist_ok=True,
)
PY