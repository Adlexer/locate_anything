#!/usr/bin/env bash
# Sample ~2K two-wheeler images (wave2_ebike + ebike_like), excluding previously sampled 150.
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/sample_dataset.py \
    --data /mnt/c/Data/datasets/elevator_yolo_detect \
    --groups "wave2_ebike:第二波数据采集汇总/电瓶车,ebike_like:电瓶车类似物" \
    --per-group 1000 --seed 42 --max-side 1280 \
    --exclude-file /mnt/c/Data/datasets/elevator_sample/exclude.txt \
    --out /mnt/c/Data/datasets/elevator_tw2k