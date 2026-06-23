#!/bin/bash
#SBATCH --job-name=nanogpt-eval
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1 
#SBATCH --output=/home/%u/proj/tinyllm/logs/%x-%A_%a.out
#SBATCH --error=/home/%u/proj/tinyllm/logs/%x-%A_%a.err


set -euo pipefail

#==============================#
#  BLOCK 0: HELPERS / FUNCTIONS#
#==============================#
log_section() {
  echo
  echo "=============================="
  echo "$1"
  echo "=============================="
}

die() { echo "ERROR: $*" >&2; exit 1; }

ensure_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || mkdir -p "$dir"
}

ensure_file() {
  local cpath="$1"
  [[ -f "$cpath" ]] || die "Error: can not find file $cpath"
}

#==============================#
#  BLOCK 1: ARGS               #
#==============================#
PROJECT_DIR="${PROJECT_DIR:-$HOME/proj/tinyllm}"
ensure_dir "$PROJECT_DIR"

REPO_DIR="${REPO_DIR:-$PROJECT_DIR/repos/nanoGPT}"
# [[ -d "$REPO_DIR" ]] || die "REPO_DIR does not exist: $REPO_DIR"
# [[ -f "$REPO_DIR/sample.py" ]] || die "Not a nanoGPT repo: $REPO_DIR"

MODULE_SET="${MODULE_SET:-"nanogpt-4"}"
ENVNAME="${ENVNAME:-/home/$USER/proj/tinyllm/envs/nanogpt-3/bin/activate}"
ensure_file "$ENVNAME"


MODEL_PATH="${MODEL_PATH:?model is required}"
ensure_file "$MODEL_PATH"

PROMPTS_PATH="${PROMPTS_PATH:?prompts are required}"
ensure_file "$PROMPTS_PATH"

JSON_HEADER="${JSON_HEADER:-''}"

OUT_DIR="${OUT_DIR:?output dir is required}"
OUT_FILENAME="${OUT_FILENAME:?output filname is required}"
OUT_FILEPATH="$OUT_DIR/$OUT_FILENAME"

RUN_DIR="$OUT_DIR"
ensure_dir "$RUN_DIR/logs"

#==============================#
#  BLOCK 3: ENVIRONMENT SETUP  #
#==============================#
cd "$REPO_DIR"

module purge
module restore $MODULE_SET
source $ENVNAME # python environment

module describe $MODULE_SET > "$RUN_DIR/logs/modules.txt"
pip list > "$RUN_DIR/logs/pythonenv.txt"

echo "MODULES: $MODULE_SET"
echo "ENVNAME: $ENVNAME"

export TIKTOKEN_CACHE_DIR=/home/alaltischenko/proj/tinyllm/tiktoken_cache

#==============================#
#  BLOCK 5: SAMPLING RUN       #
#==============================#
CMD=(
    python 
    sample.py 
    "$CONFIG_PATH" 
    --model_path="$MODEL_PATH"
    --start="JSONFILE:$PROMPTS_PATH"
    --output_path="$OUT_FILEPATH"
    --jsonify_output=True
    --json_header="$JSON_HEADER"
    --append_mode=False
    --device=cuda 
    --compile=False
)
printf "CMD: "; printf "%q " "${CMD[@]}"; echo
"${CMD[@]}"

