#!/usr/bin/env bash
# Fair comparison: tw_run_v1 & tw_run_v2 vs human GT (yolo_detect_tw_v2 data.yaml)
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
DATA=/mnt/c/Data/datasets/yolo_detect_tw_v2/data.yaml
for MODEL in /home/xu/data/yolo_runs/tw_run_v1/weights/best.pt /home/xu/data/yolo_runs/tw_run_v2/weights/best.pt; do
  echo "=== val: $MODEL ==="
  "$PY" - "$MODEL" "$DATA" <<'PY'
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
r = m.val(data=sys.argv[2], imgsz=640, batch=16, verbose=True)
print("mAP50=%.4f mAP50-95=%.4f" % (r.box.map50, r.box.map))
PY
done