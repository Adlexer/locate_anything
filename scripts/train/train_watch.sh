#!/usr/bin/env bash
# =============================================================================
# train_watch.sh - LocateAnything LoRA training monitor
#
# Usage:
#   bash scripts/train/train_watch.sh <output_dir>            # tail last 40 lines
#   bash scripts/train/train_watch.sh <output_dir> --follow   # tail -f (Ctrl-C to quit)
#   bash scripts/train/train_watch.sh <output_dir> --loss     # recent loss lines
#   bash scripts/train/train_watch.sh <output_dir> --gpu      # GPU stats (1s loop)
#   bash scripts/train/train_watch.sh <output_dir> --tensorboard [--port 6006]
#   bash scripts/train/train_watch.sh <output_dir> --checkpoints
# =============================================================================
set -euo pipefail

OUT="${1:?usage: train_watch.sh <output_dir> [--follow|--loss|--gpu|--tensorboard|--checkpoints]}"
shift || true
MODE="tail"
PORT=6006

while [[ $# -gt 0 ]]; do
    case "$1" in
        --follow)      MODE="follow"; shift ;;
        --loss)        MODE="loss"; shift ;;
        --gpu)         MODE="gpu"; shift ;;
        --tensorboard) MODE="tb"; shift ;;
        --checkpoints) MODE="ckpt"; shift ;;
        --port)        PORT="${2:?--port needs a value}"; shift 2 ;;
        *)             echo "unknown option: $1"; exit 1 ;;
    esac
done

[[ -d "$OUT" ]] || { echo "[watch] output dir not found: $OUT"; exit 1; }
LOG="$OUT/training_log.txt"

case "$MODE" in
    tail)
        if [[ -f "$LOG" ]]; then
            echo "== $LOG =="
            tail -n 40 "$LOG"
        else
            echo "[watch] no training_log.txt yet; listing dir:"
            ls -la "$OUT"
        fi
        ;;
    follow)
        if [[ ! -f "$LOG" ]]; then
            echo "[watch] waiting for $LOG ..."
            while [[ ! -f "$LOG" ]]; do sleep 5; done
        fi
        echo "[watch] following $LOG (Ctrl-C to stop)"
        tail -n 40 -f "$LOG"
        ;;
    loss)
        [[ -f "$LOG" ]] || { echo "[watch] no training_log.txt yet"; exit 0; }
        echo "[watch] recent loss lines:"
        grep -E "loss|grad_norm|learning_rate" "$LOG" | tail -n 20
        ;;
    gpu)
        echo "[watch] GPU stats (Ctrl-C to stop)"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader -l 1
        ;;
    tb)
        if ! command -v tensorboard >/dev/null 2>&1; then
            echo "[watch] tensorboard not found; install: pip install tensorboard"
            exit 1
        fi
        echo "[watch] tensorboard --logdir $OUT --port $PORT  (open http://localhost:$PORT)"
        exec tensorboard --logdir "$OUT" --port "$PORT"
        ;;
    ckpt)
        echo "[watch] checkpoints under $OUT:"
        for d in "$OUT"/checkpoint-*; do
            [[ -d "$d" ]] || continue
            step="$(basename "$d")"
            size="$(du -sh "$d" 2>/dev/null | cut -f1)"
            echo "  $step  ($size)"
        done
        if ls "$OUT"/checkpoint-* >/dev/null 2>&1; then
            latest="$(ls -d "$OUT"/checkpoint-* | sort -V | tail -n1)"
            echo "[watch] latest: $latest"
        else
            echo "[watch] no checkpoints yet"
        fi
        ;;
esac
