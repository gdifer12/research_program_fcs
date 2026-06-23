#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/../../../"
RUN_SCRIPT_PATH="$EXP_DIR/inference/prepare_sample.py"
EVAL_SCRIPT_PATH="$EXP_DIR/inference/run_sample.sbatch"

EXP_NAME="E_1"

CONFIG_PATH="$SCRIPT_DIR/config.py"

SUMMARY_PATH="$EXP_DIR/res/summary_ftl.csv"

MODELS=(
    "base:/home/alaltischenko/proj/tinyllm/fine-tune-2/runs/3654739_3_ft2_F_base_2026-02-26_06-48-53/out/ckpt_best.pt"
    "ftl_F_1"
    "ftl_F_8"
    "ftl_F_19"
    "ftl_F_7"
)
MODELS=$(IFS=','; printf '%s' "${MODELS[*]}")

PROMPTS_PATH="$EXP_DIR/inference/data/tiny-stories-1/eval_prompts.jsonl"

OUT_DIR="$EXP_DIR/inference/runs/$EXP_NAME"

CMD=(
    python
    "$RUN_SCRIPT_PATH"
    --eval-script="$EVAL_SCRIPT_PATH"
    --summary="$SUMMARY_PATH"
    --models="$MODELS"
    --prompts="$PROMPTS_PATH"
    --config="$CONFIG_PATH"
    --out-dir="$OUT_DIR"
)

printf "CMD: "
printf "%q " "${CMD[@]}"
echo

"${CMD[@]}"