#!/usr/bin/env bash
# YOLO26s baseline on two-wheeler teacher pseudo-labels (pre-review).
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/build_tw_dataset.py \
    --data /mnt/c/Data/datasets/elevator_sample_tw \
    --out /mnt/c/Data/datasets/yolo_detect_tw \
    --val-ratio 0.1 --seed 42