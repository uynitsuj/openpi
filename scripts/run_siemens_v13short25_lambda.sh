#!/usr/bin/env bash
# Stage or train one Siemens v13 shortest-25% crop ablation on a Lambda 8xA100 node.
#
# Staging does not need a base checkpoint and never starts a GPU process:
#   scripts/run_siemens_v13short25_lambda.sh stage cc
#   scripts/run_siemens_v13short25_lambda.sh stage tc
#
# Training deliberately requires an explicit base checkpoint and experiment name:
#   BASE_PARAMS_URI=s3://.../step/params EXP_NAME=... \
#     scripts/run_siemens_v13short25_lambda.sh train cc
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_siemens_v13short25_lambda.sh stage <cc|tc>
  BASE_PARAMS_URI=<.../params> EXP_NAME=<name> run_siemens_v13short25_lambda.sh train <cc|tc>

Environment overrides:
  OPENPI_ROOT        checkout to run (default: directory above this script)
  OPENPI_PYTHON      Python executable (default: /home/ubuntu/karim/openpi/.venv/bin/python)
  DATA_ROOT          LeRobot cache root (default: /home/ubuntu/.cache/huggingface/lerobot)
  CHECKPOINT_ROOT    local checkpoint root (default: /home/ubuntu/karim/checkpoints/openpi)
  S3_DATASET_ROOT    dataset bucket prefix
  S3_CHECKPOINT_ROOT checkpoint bucket prefix
  AWS_PROFILE        AWS SSO profile (default: EngineerAccess-266735817792)
  NUM_WORKERS        data-loader workers (default: 8)
  OPENPI_FSDP_DEVICES model-shard width (default: 4 for Lambda A100)
  RESUME=1           resume an existing experiment instead of refusing to overwrite it
EOF
}

[[ $# -eq 2 ]] || { usage >&2; exit 2; }
mode=$1
variant=$2
[[ "$mode" == "stage" || "$mode" == "train" ]] || { usage >&2; exit 2; }
[[ "$variant" == "cc" || "$variant" == "tc" ]] || { usage >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
openpi_root=${OPENPI_ROOT:-$(cd -- "$script_dir/.." && pwd)}
python_bin=${OPENPI_PYTHON:-/home/ubuntu/karim/openpi/.venv/bin/python}
data_root=${DATA_ROOT:-/home/ubuntu/.cache/huggingface/lerobot}
checkpoint_root=${CHECKPOINT_ROOT:-/home/ubuntu/karim/checkpoints/openpi}
s3_dataset_root=${S3_DATASET_ROOT:-s3://xdof-internal-research/siemens/datasets}
s3_checkpoint_root=${S3_CHECKPOINT_ROOT:-s3://xdof-internal-research/siemens/policy_ckpts}
num_workers=${NUM_WORKERS:-8}
fsdp_devices=${OPENPI_FSDP_DEVICES:-4}
export AWS_PROFILE=${AWS_PROFILE:-EngineerAccess-266735817792}
export AWS_DEFAULT_PROFILE=${AWS_DEFAULT_PROFILE:-$AWS_PROFILE}

repo="siemens_simple_d405_v13short25${variant}"
config="pi05_siemens_simple_d405_v13short25${variant}_bs128"
expected_episodes=1976
expected_resize_mode=center_crop
if [[ "$variant" == "tc" ]]; then
    expected_episodes=1982
    expected_resize_mode=top_center_crop
fi

dataset_dir="$data_root/$repo"
asset_dir="$openpi_root/assets/$config/$repo"

log() { printf '[v13short25-%s %s] %s\n' "$variant" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

stage_dataset() {
    command -v aws >/dev/null || { log "ERROR: aws CLI is not installed"; return 1; }
    aws sts get-caller-identity --output text >/dev/null
    mkdir -p "$dataset_dir" "$asset_dir"
    log "syncing $s3_dataset_root/$repo -> $dataset_dir"
    aws s3 sync "$s3_dataset_root/$repo" "$dataset_dir" \
        --exclude 'norm_stats/*' --only-show-errors
    aws s3 cp \
        "$s3_dataset_root/$repo/norm_stats/$config/norm_stats.json" \
        "$asset_dir/norm_stats.json" --only-show-errors

    "$python_bin" - "$dataset_dir/meta/info.json" "$expected_episodes" "$expected_resize_mode" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
expected_episodes = int(sys.argv[2])
expected_resize_mode = sys.argv[3]
info = json.loads(info_path.read_text())
checks = {
    "total_episodes": (info.get("total_episodes"), expected_episodes),
    "resize_mode": (info.get("resize_mode"), expected_resize_mode),
    "action_source": (info.get("action_source"), "action_mcap"),
    "joint_order": (info.get("joint_order"), "driver"),
    "fps": (info.get("fps"), 30),
}
bad = [f"{key}: got {got!r}, expected {want!r}" for key, (got, want) in checks.items() if got != want]
for key in ("left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"):
    shape = info.get("features", {}).get(key, {}).get("shape")
    if shape != [224, 224, 3]:
        bad.append(f"{key}.shape: got {shape!r}, expected [224, 224, 3]")
if bad:
    raise SystemExit("dataset validation failed:\n  " + "\n  ".join(bad))
print(f"validated {info_path.parent.parent}: {expected_episodes} episodes, {expected_resize_mode}, leader actions")
PY

    [[ -s "$asset_dir/norm_stats.json" ]] || { log "ERROR: norm stats are missing"; return 1; }
    [[ -d "$dataset_dir/data" && -d "$dataset_dir/videos" ]] || {
        log "ERROR: dataset payload is incomplete"
        return 1
    }
    log "STAGE_READY repo=$repo config=$config bytes=$(du -sb "$dataset_dir" | cut -f1)"
}

if [[ "$mode" == "stage" ]]; then
    stage_dataset
    exit 0
fi

base_params_uri=${BASE_PARAMS_URI:-}
exp_name=${EXP_NAME:-}
[[ -n "$base_params_uri" ]] || {
    log "ERROR: BASE_PARAMS_URI must name the selected base checkpoint's params directory"
    exit 2
}
[[ -n "$exp_name" ]] || {
    log "ERROR: EXP_NAME must be explicit so the base lineage is recorded in the run name"
    exit 2
}
[[ "$base_params_uri" == */params ]] || {
    log "ERROR: BASE_PARAMS_URI must end in /params (got: $base_params_uri)"
    exit 2
}

stage_dataset

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
    log "ERROR: refusing to launch because GPU compute processes already exist: ${gpu_pids[*]}"
    exit 1
fi

checkpoint_dir="$checkpoint_root/$config/$exp_name"
train_mode=(--overwrite)
if [[ -e "$checkpoint_dir" ]]; then
    if [[ "${RESUME:-0}" == "1" ]]; then
        train_mode=(--resume)
    else
        log "ERROR: $checkpoint_dir already exists; choose another EXP_NAME or set RESUME=1"
        exit 1
    fi
fi

mkdir -p "$checkpoint_root"
cd "$openpi_root"
export PYTHONPATH="$openpi_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=${WANDB_ENTITY:-karim-el-refai-ucb}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/home/ubuntu/karim/wandb}
# pi0.5's longer sequence needs the stock, memory-minimizing remat policy on A100.
export OPENPI_REMAT_POLICY=${OPENPI_REMAT_POLICY:-nothing_saveable}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.93}
mkdir -p "$WANDB_DIR"

log "TRAIN_START config=$config exp=$exp_name steps=20000 batch=128 fsdp_devices=$fsdp_devices base=$base_params_uri"
exec "$python_bin" scripts/train_siemens_v13short25.py "$variant" \
    --exp-name="$exp_name" \
    "${train_mode[@]}" \
    --num-workers="$num_workers" \
    --fsdp-devices="$fsdp_devices" \
    --checkpoint-base-dir="$checkpoint_root" \
    --s3-checkpoint-path="$s3_checkpoint_root/$config/$exp_name" \
    --base-params-uri="$base_params_uri"
