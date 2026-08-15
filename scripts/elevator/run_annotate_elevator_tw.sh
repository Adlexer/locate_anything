#!/usr/bin/env bash
# Teacher zero-shot annotation on the TWO-WHEELER subset (electric scooter / bicycle).
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/annotation/annotate_yolo.py \
    --data /mnt/c/Data/datasets/elevator_sample_tw \
    --classes "electric scooter,bicycle" \
    --model /home/xu/data/models/LocateAnything-3B \
    --out /mnt/c/Dev/locate_anything/outputs/annotation_elevator_tw \
    --mode hybrid --max-new-tokens 2048 --device cuda