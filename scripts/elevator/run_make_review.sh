#!/usr/bin/env bash
# Generate human review artifacts for the elevator sample.
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
SAMPLE=/mnt/c/Data/datasets/elevator_sample
CC=/mnt/c/Dev/locate_anything/outputs/elevator_crosscheck/worksheet.jsonl
OUT=/mnt/c/Dev/locate_anything/outputs/elevator_review
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/make_review_artifacts.py \
    --data "$SAMPLE" --crosscheck "$CC" --out "$OUT" \
    --title "电梯场景样本人工视觉复核（150 图）"