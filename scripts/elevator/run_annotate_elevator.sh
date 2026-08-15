#!/usr/bin/env bash
# Teacher zero-shot annotation on the elevator sample (LocateAnything base model).
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/annotation/annotate_yolo.py \
    --data /mnt/c/Data/datasets/elevator_sample \
    --classes "gas cylinder,electric scooter,bicycle" \
    --model /home/xu/data/models/LocateAnything-3B \
    --out /mnt/c/Dev/locate_anything/outputs/annotation_elevator \
    --mode hybrid --max-new-tokens 2048 --device cuda