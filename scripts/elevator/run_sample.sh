#!/usr/bin/env bash
# Sample elevator_yolo_detect -> elevator_sample work set (stratified, downscaled).
set -euo pipefail
PY=~/miniconda3/envs/locate_anything_sft/bin/python
"$PY" /mnt/c/Dev/locate_anything/scripts/elevator/sample_dataset.py \
    --data /mnt/c/Data/datasets/elevator_yolo_detect \
    --groups "gas:煤气罐,battery:电瓶,ebike_like:电瓶车类似物,wave2_battery:第二波数据采集汇总/电瓶,wave2_ebike:第二波数据采集汇总/电瓶车" \
    --per-group 30 --seed 42 --max-side 1280 \
    --out /mnt/c/Data/datasets/elevator_sample