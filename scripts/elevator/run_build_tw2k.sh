#!/usr/bin/env bash
# Assemble + train tuned YOLO26s on the 2K two-wheeler data (tw_run_v3).
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/build_tw2k_dataset.py \
    --new /mnt/c/Data/datasets/elevator_tw2k \
    --reviewed /mnt/c/Data/datasets/elevator_sample_tw_gt \
    --val-list /mnt/c/Data/datasets/yolo_detect_tw_v2/images/val \
    --out /mnt/c/Data/datasets/yolo_detect_tw2k