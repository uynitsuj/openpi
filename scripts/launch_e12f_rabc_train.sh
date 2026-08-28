#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 CONFIG_NAME EXPERIMENT_NAME [CHECKPOINT_BASE_DIR]" >&2
  echo "HF_LEROBOT_HOME must point at the staged LeRobot datasets." >&2
  exit 2
fi

config_name=$1
experiment_name=$2
repo_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint_base_dir=${3:-${OPENPI_CHECKPOINT_BASE_DIR:-"$repo_root/checkpoints"}}

: "${HF_LEROBOT_HOME:?Set HF_LEROBOT_HOME to the local LeRobot dataset cache}"
export OPENPI_REMAT_POLICY=${OPENPI_REMAT_POLICY:-dots_with_no_batch_dims_saveable}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.93}
export PYTHONUNBUFFERED=1

cd "$repo_root"
exec uv run scripts/train.py "$config_name" \
  --exp-name="$experiment_name" \
  --checkpoint-base-dir="$checkpoint_base_dir" \
  --no-wandb-enabled
