#!/usr/bin/env bash
# Fair val: v1 / v2 / v3 on the 2K dataset human-GT val (12 images).
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
DATA=/mnt/c/Data/datasets/yolo_detect_tw2k/data.yaml
for MODEL in /home/xu/data/yolo_runs/tw_run_v1/weights/best.pt \
             /home/xu/data/yolo_runs/tw_run_v2/weights/best.pt \
             /home/xu/data/yolo_runs/tw_run_v3/weights/best.pt; do
  [ -f "$MODEL" ] || { echo "skip missing $MODEL"; continue; }
  echo "=== val: $MODEL ==="
  "$PY" - "$MODEL" "$DATA" <<'PY'
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
r = m.val(data=sys.argv[2], imgsz=640, batch=16, verbose=False)
print("mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f" % (r.box.map50, r.box.map, r.box.mp, r.box.mr))
PY
done