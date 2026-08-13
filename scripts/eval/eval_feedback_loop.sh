#!/usr/bin/env bash
# Feedback-loop evaluation: pretrained vs finetuned (run_v2) on corrected GT (detect_v2).
# Usage: bash scripts/eval/eval_feedback_loop.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/miniconda3/envs/locate_anything_sft/bin/python
DATA=/mnt/c/Data/datasets/detect_v2
SPLIT="$REPO/outputs/annotation_detect/lora_data_v2/val.jsonl"
CLASSES="gas cylinder,electric scooter,bicycle"
OUT="$REPO/outputs/annotation_detect"
MODEL_BASE=/home/xu/models/LocateAnything-3B
MODEL_FT=/home/xu/lora_gas/run_v2

echo "[eval-loop] pretrained: $MODEL_BASE"
"$PY" "$REPO/scripts/eval/eval_det.py" \
    --model "$MODEL_BASE" --data "$DATA" --classes "$CLASSES" --split "$SPLIT" \
    --mode slow --out "$OUT/eval_v2_pretrained.json" --report-md "$OUT/eval_v2_pretrained.md"

echo "[eval-loop] finetuned: $MODEL_FT"
"$PY" "$REPO/scripts/eval/eval_det.py" \
    --model "$MODEL_FT" --data "$DATA" --classes "$CLASSES" --split "$SPLIT" \
    --mode slow --out "$OUT/eval_v2_finetuned.json" --report-md "$OUT/eval_v2_finetuned.md"

echo "[eval-loop] done."