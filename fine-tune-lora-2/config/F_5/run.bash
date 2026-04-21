#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="$SCRIPT_DIR/../../template.sbatch"

EXP_GROUP="ftl"
CUR_EXP_NAME="F_5"
EXP_NAME="${EXP_GROUP}_${CUR_EXP_NAME}"

CONFIG_PATH="$SCRIPT_DIR/config.py"
DATASET="tinystoriesInstruct"
 
RUNS_ROOT="$SCRIPT_DIR/../../runs"

export EXP_DESC="Apply LoRA to wte"

EXTRA_SBATCH_ARGS=(--time=08:00:00)

sbatch "${EXTRA_SBATCH_ARGS[@]}" \
  --job-name="nanogpt-${EXP_NAME}" \
  --export=ALL,EXP_NAME="$EXP_NAME",CONFIG_PATH="$CONFIG_PATH",RUNS_ROOT="$RUNS_ROOT",DATASET="$DATASET" \
  "$TEMPLATE_PATH"
