#!/usr/bin/env bash
# =============================================================================
# train_lora.sh - LocateAnything LoRA training launcher (single-node)
#
# Based on: reports/05_lora_sft_research.md sec.7 (validated on RTX 5060 Ti 16GB)
#   LoRA r=64 + sdpa + seq=2048 + grad checkpoint + DeepSpeed ZeRO-1(torch_adam)
#
# Examples:
#   bash scripts/train/train_lora.sh \
#       --meta /home/xu/lora_smoke/recipe.json \
#       --output /home/xu/lora_smoke/work_dirs/out \
#       --steps 5000 --save-steps 200
#
# Env overrides: MODEL_PATH / RANK / MAX_SEQ / MAX_STEPS / LR / WARMUP
#   SAVE_STEPS / GRAD_ACC / ATTN / GPUS / MASTER_PORT / WANDB / CONDA_ENV
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMBODIED="$REPO/Eagle/Embodied"
DS_CONFIG="$REPO/scripts/train/ds_z1_torchadam.json"

# ---------------- defaults ----------------
CONDA_ENV="${CONDA_ENV:-locate_anything_sft}"
MODEL_PATH="${MODEL_PATH:-/home/xu/models/LocateAnything-3B}"
RANK="${RANK:-64}"                   # LLM LoRA rank (0 = off)
BACKBONE_RANK="${BACKBONE_RANK:-0}"  # vision LoRA rank (0 = off)
FREEZE_MLP="${FREEZE_MLP:-False}"    # True = freeze MLP connector
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-2e-5}"
WARMUP="${WARMUP:-200}"
MAX_SEQ="${MAX_SEQ:-2048}"           # 16GB single-GPU ceiling ~2048 (measured)
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
GRAD_ACC="${GRAD_ACC:-1}"
ATTN="${ATTN:-sdpa}"                 # sdpa for consumer GPUs; magi needs Hopper
BLOCK_SIZE="${BLOCK_SIZE:-6}"        # MTP block size
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
WANDB="${WANDB:-0}"                  # 1 = enable wandb (requires wandb login)

META_PATH=""
OUTPUT_DIR=""
OVERWRITE=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: train_lora.sh --meta <recipe.json> --output <dir> [options]

Required:
  --meta <recipe.json>    data recipe JSON (use absolute annotation/root paths)
  --output <dir>          output dir (auto-resumes when a checkpoint exists)

Optional:
  --steps N               total steps (default: $MAX_STEPS)
  --lr LR                 learning rate (default: $LR)
  --rank R                LLM LoRA rank (default: $RANK)
  --backbone-rank R       vision LoRA rank (default: $BACKBONE_RANK)
  --seq N                 max_seq_length (default: $MAX_SEQ)
  --save-steps N          checkpoint interval (default: $SAVE_STEPS)
  --grad-acc N            gradient accumulation (default: $GRAD_ACC)
  --attn NAME             sdpa|magi (default: $ATTN)
  --freeze-mlp BOOL       True|False (default: $FREEZE_MLP)
  --overwrite             wipe & rebuild output dir (dangerous)
  --gpus N                number of GPUs (default: $GPUS)
  --wandb                 enable wandb
  --extra "...args..."    extra args passed through to the training script

Env overrides: MODEL_PATH RANK MAX_SEQ MAX_STEPS LR WARMUP SAVE_STEPS GRAD_ACC ATTN GPUS MASTER_PORT WANDB CONDA_ENV
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --meta)          META_PATH="${2:?--meta needs a value}"; shift 2 ;;
        --output)        OUTPUT_DIR="${2:?--output needs a value}"; shift 2 ;;
        --steps)         MAX_STEPS="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --rank)          RANK="$2"; shift 2 ;;
        --backbone-rank) BACKBONE_RANK="$2"; shift 2 ;;
        --seq)           MAX_SEQ="$2"; shift 2 ;;
        --save-steps)    SAVE_STEPS="$2"; shift 2 ;;
        --grad-acc)      GRAD_ACC="$2"; shift 2 ;;
        --attn)          ATTN="$2"; shift 2 ;;
        --freeze-mlp)    FREEZE_MLP="$2"; shift 2 ;;
        --overwrite)     OVERWRITE=1; shift ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        --wandb)         WANDB=1; shift ;;
        --extra)         read -r -a EXTRA_ARGS <<< "$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "unknown argument: $1"; usage; exit 1 ;;
    esac
done

[[ -n "$META_PATH" ]]  || { echo "[train] missing --meta"; usage; exit 1; }
[[ -n "$OUTPUT_DIR" ]] || { echo "[train] missing --output"; usage; exit 1; }

# ---------------- preflight ----------------
[[ -d "$MODEL_PATH" ]] || { echo "[train] model dir not found: $MODEL_PATH"; exit 1; }
[[ -f "$META_PATH" ]]  || { echo "[train] recipe not found: $META_PATH"; exit 1; }
[[ -f "$DS_CONFIG" ]]  || { echo "[train] deepspeed config missing: $DS_CONFIG"; exit 1; }
[[ -d "$EMBODIED" ]]   || { echo "[train] Eagle/Embodied not found: $EMBODIED"; exit 1; }

if [[ -d "$OUTPUT_DIR" && "$OVERWRITE" -eq 1 ]]; then
    echo "[train] --overwrite: removing $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# refuse to touch a non-empty dir that has no checkpoint
if [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]] && ! ls "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
    echo "[train] output dir is non-empty and has no checkpoint; refusing to start (use --overwrite): $OUTPUT_DIR"
    exit 1
fi
if ls "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
    echo "[train] checkpoint found; training will resume (streaming state restores bit-wise)"
fi
if [[ -f "$OUTPUT_DIR/done.txt" ]]; then
    echo "[train] WARNING: done.txt exists (previous run marked finished)."
    echo "[train] The training script will EXIT immediately. To extend training: rm $OUTPUT_DIR/done.txt"
fi

# ---------------- python (env python directly, no conda function needed) ----------------
MINICONDA_ROOT="${MINICONDA_ROOT:-$HOME/miniconda3}"
ENV_PY="$MINICONDA_ROOT/envs/$CONDA_ENV/bin/python"
if [[ ! -x "$ENV_PY" ]]; then
    echo "[train] python not found: $ENV_PY (CONDA_ENV=$CONDA_ENV)"
    exit 1
fi
export PATH="$(dirname "$ENV_PY"):$PATH"

REPORT_TO="tensorboard"
if [[ "$WANDB" == "1" ]]; then
    REPORT_TO="wandb,tensorboard"
    export WANDB_PROJECT="${WANDB_PROJECT:-locate-anything-lora}"
fi

# ---------------- build training command ----------------
CMD=(
    LAUNCHER=pytorch "$ENV_PY" -m torch.distributed.run
    --nnodes=1 --nproc_per_node="$GPUS" --master_port="$MASTER_PORT"
    eaglevl/train/locany_finetune_magi_stream.py
    --model_name_or_path "$MODEL_PATH"
    --meta_path "$META_PATH"
    --output_dir "$OUTPUT_DIR"
    --overwrite_output_dir "$([[ $OVERWRITE -eq 1 ]] && echo True || echo False)"
    --max_steps "$MAX_STEPS"
    --block_size "$BLOCK_SIZE"
    --attn_implementation "$ATTN"
    --causal_attn False
    --freeze_llm True
    --freeze_backbone True
    --freeze_mlp "$FREEZE_MLP"
    --use_llm_lora "$RANK"
    --use_backbone_lora "$BACKBONE_RANK"
    --vision_select_layer -1
    --dataloader_num_workers 0
    --bf16 True
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACC"
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --learning_rate "$LR"
    --weight_decay 0.01
    --warmup_steps "$WARMUP"
    --lr_scheduler_type cosine
    --logging_steps 1
    --video_total_pixels 8192
    --sample_log_interval 1
    --packing_buffer_size 8
    --max_seq_length "$MAX_SEQ"
    --max_num_tokens_per_sample "$MAX_SEQ"
    --max_num_tokens "$MAX_SEQ"
    --do_train True
    --grad_checkpoint True
    --group_by_length False
    --deepspeed "$DS_CONFIG"
    --report_to "$REPORT_TO"
    --run_name "$(basename "$OUTPUT_DIR")"
    --use_onelogger False
    --mlp_connector_layers 2
    "${EXTRA_ARGS[@]}"
)

echo "[train] env   : conda=$CONDA_ENV  model=$MODEL_PATH"
echo "[train] LoRA  : llm_rank=$RANK backbone_rank=$BACKBONE_RANK freeze_mlp=$FREEZE_MLP"
echo "[train] data  : $META_PATH"
echo "[train] out   : $OUTPUT_DIR  (steps=$MAX_STEPS lr=$LR seq=$MAX_SEQ attn=$ATTN gpus=$GPUS)"
echo "[train] log   : $OUTPUT_DIR/training_log.txt"
echo "[train] watch : bash scripts/train/train_watch.sh $OUTPUT_DIR --follow"

cd "$EMBODIED"
exec env "${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/training_log.txt"
