#!/usr/bin/bash

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <experiment_group_name> <experiment_group_short_name> <experiment_name> <dataset>"
  exit 1
fi

GROUP=$1
SHORTNAME=$2
NAME=$3
DATASET=$4

mkdir -p "$GROUP"/{config,runs,res}

WORK_DIR="$GROUP/config/$NAME"
mkdir $WORK_DIR

touch "$WORK_DIR/config.py"
touch "$WORK_DIR/run.bash"

cat > "$WORK_DIR/run.bash" <<EOF
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="\$SCRIPT_DIR/../../template.sbatch"

EXP_GROUP="$SHORTNAME"
CUR_EXP_NAME="$NAME"
EXP_NAME="\${EXP_GROUP}_\${CUR_EXP_NAME}"

CONFIG_PATH="\$SCRIPT_DIR/config.py"
DATASET="$DATASET"
 
RUNS_ROOT="\$SCRIPT_DIR/../../runs"

export EXP_DESC="Baseline for $NAME"

EXTRA_SBATCH_ARGS=(--time=06:00:00)

sbatch "\${EXTRA_SBATCH_ARGS[@]}" \\
  --job-name="nanogpt-\${EXP_NAME}" \\
  --export=ALL,EXP_NAME="\$EXP_NAME",CONFIG_PATH="\$CONFIG_PATH",RUNS_ROOT="\$RUNS_ROOT",DATASET="\$DATASET" \\
  "\$TEMPLATE_PATH"
EOF

chmod +x "$WORK_DIR/run.bash"

cp "template.sbatch" "$GROUP/template.sbatch"
cp "summarize.py" "$GROUP/summarize.py"

VENV_DIR="$GROUP/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install scipy
  deactivate
fi


echo "Experiment group '$GROUP' created."
echo "Experiment '$GROUP/$NAME' created."
echo "Put config to '$WORK_DIR/config.py' and run '$WORK_DIR/run.bash'"
echo "Last will run $GROUP/template.sbatch"
echo "After run you can run $GROUP/summarize.py (do not forget activate '$VENV_DIR' if run --mode filter)"
