#!/usr/bin/env bash
# YOLO critic inference + teacher-YOLO cross-check on the elevator sample.
set -euo pipefail
PY=~/miniconda3/envs/yolo/bin/python
SAMPLE=/mnt/c/Data/datasets/elevator_sample
MODEL=/home/xu/data/yolo_runs/run_v1/weights/best.pt
OUT=/mnt/c/Dev/locate_anything/outputs/elevator_crosscheck
"$PY" /mnt/c/Dev/locate_anything/scripts/loop/cross_check.py \
    --data "$SAMPLE" --model "$MODEL" --out "$OUT" \
    --imgsz 640 --conf 0.25 --iou-thr 0.5