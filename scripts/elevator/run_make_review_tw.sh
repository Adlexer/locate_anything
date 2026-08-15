#!/usr/bin/env bash
# Review artifacts for the TWO-WHEELER subset.
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
SAMPLE=/mnt/c/Data/datasets/elevator_sample_tw
CC=/mnt/c/Dev/locate_anything/outputs/elevator_crosscheck_tw/worksheet.jsonl
OUT=/mnt/c/Dev/locate_anything/outputs/elevator_review_tw
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/make_review_artifacts.py \
    --data "$SAMPLE" --crosscheck "$CC" --out "$OUT" \
    --title "两轮车交叉标注人工复核（电动车/单车，120 图）"