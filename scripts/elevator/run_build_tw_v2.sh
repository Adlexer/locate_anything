#!/usr/bin/env bash
# Build v2 two-wheeler YOLO dataset from HUMAN-reviewed GT (incl. empty images as negatives).
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/build_tw_dataset.py \
    --data /mnt/c/Data/datasets/elevator_sample_tw_gt \
    --out /mnt/c/Data/datasets/yolo_detect_tw_v2 \
    --val-ratio 0.1 --seed 42