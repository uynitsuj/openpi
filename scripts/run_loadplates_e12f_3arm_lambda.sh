#!/usr/bin/env bash
# Train the real ABC load-plates E12F three-arm study from the release-aware
# LeRobot v3 conversion, sequentially on one 8xA100 Lambda node. All configs
# use a global batch of 128 and 15,000 steps.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

python_bin=${OPENPI_PYTHON:-"$repo_root/.venv/bin/python"}
dataset_home=${HF_LEROBOT_HOME:-/home/ubuntu/karim/datasets}
dataset_id=abc130k_real_load_plates_lerobot_v1
sidecar_root=${OPENPI_LOAD_PLATES_E12F_SIDECAR_ROOT:-/home/ubuntu/karim/sidecars/real_load_plates_lerobot_v1/e12f_top50}
checkpoint_root=${OPENPI_CHECKPOINT_ROOT:-/home/ubuntu/karim/checkpoints/openpi}
run_tag=${OPENPI_RUN_TAG:-real_loadplates_e12f_$(date -u +%Y%m%d)}
log_root=${OPENPI_LOG_ROOT:-/home/ubuntu/karim/logs/openpi_loadplates_e12f}

all_configs=(
  pi0_load_plates_e12f_vanilla_bc_bs128_15k
  pi0_load_plates_e12f_top50_chunks_bs128_15k
  pi0_load_plates_e12f_top50_episodes_bs128_15k
)
if [[ -n "${OPENPI_LOAD_PLATES_CONFIGS:-}" ]]; then
  IFS=',' read -r -a configs <<< "${OPENPI_LOAD_PLATES_CONFIGS}"
else
  configs=("${all_configs[@]}")
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "OPENPI_LOAD_PLATES_CONFIGS selected no configs" >&2
  exit 2
fi
for cfg in "${configs[@]}"; do
  valid=0
  for allowed in "${all_configs[@]}"; do
    [[ "$cfg" == "$allowed" ]] && valid=1
  done
  if [[ "$valid" -ne 1 ]]; then
    echo "unknown load-plates config: $cfg" >&2
    exit 2
  fi
done

log() {
  echo "[loadplates-e12f $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

if [[ ! -x "$python_bin" ]]; then
  log "ERROR: OpenPI Python not found at $python_bin"
  exit 2
fi
if [[ ! -f "$dataset_home/$dataset_id/meta/info.json" ]]; then
  log "ERROR: dataset missing at $dataset_home/$dataset_id"
  exit 2
fi
for cfg in "${configs[@]}"; do
  case "$cfg" in
    *top50_chunks*) sidecar="$sidecar_root/top50_action_chunks/frame_signals.parquet" ;;
    *top50_episodes*) sidecar="$sidecar_root/top50_episodes/frame_signals.parquet" ;;
    *) continue ;;
  esac
  if [[ ! -f "$sidecar" ]]; then
    log "ERROR: curation sidecar missing at $sidecar"
    exit 2
  fi
done

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -ne 8 ]]; then
  log "ERROR: expected exactly 8 visible GPUs, found $gpu_count"
  exit 2
fi

# The branch's measured bs128 fast path is fsdp=2 on 80 GB GPUs. A100-40GB
# needs the smaller per-device model/optimizer footprint of full-node fsdp=8.
min_gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if [[ -n "${OPENPI_FSDP_DEVICES:-}" ]]; then
  fsdp_devices=$OPENPI_FSDP_DEVICES
elif [[ "$min_gpu_mib" -ge 70000 ]]; then
  fsdp_devices=2
else
  fsdp_devices=8
fi

export HF_LEROBOT_HOME="$dataset_home"
export OPENPI_LOAD_PLATES_E12F_SIDECAR_ROOT="$sidecar_root"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/home/ubuntu/karim/openpi_cache}
export OPENPI_REMAT_POLICY=${OPENPI_REMAT_POLICY:-dots_with_no_batch_dims_saveable}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.93}
export PYTHONUNBUFFERED=1
mkdir -p "$checkpoint_root" "$log_root"

log "GPU inventory"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
log "global_batch=128 fsdp_devices=$fsdp_devices steps=15000 configs=${configs[*]}"

# Fail before any expensive decode if the configs or curated masks drift.
"$python_bin" - "$dataset_home/$dataset_id" "$sidecar_root" "${configs[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from openpi.training import config

dataset = Path(sys.argv[1])
sidecars = Path(sys.argv[2])
selected_configs = set(sys.argv[3:])
info = json.loads((dataset / "meta" / "info.json").read_text())
assert str(info["codebase_version"]).startswith("v3"), info["codebase_version"]
metadata = LeRobotDatasetMetadata(repo_id=dataset.name, root=dataset)
assert metadata.total_episodes == 1335, metadata.total_episodes
episode_files = sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))
assert episode_files, dataset
lengths = np.concatenate(
    [pq.read_table(path, columns=["length"])["length"].to_numpy() for path in episode_files]
).astype(np.int64)
assert len(lengths) == metadata.total_episodes, len(lengths)
total_frames = int(lengths.sum())
assert total_frames == metadata.total_frames, (total_frames, metadata.total_frames)
total_chunks = total_frames  # one padded action-chunk anchor per LeRobot frame

expected = {}
if "pi0_load_plates_e12f_top50_chunks_bs128_15k" in selected_configs:
    expected["top50_action_chunks"] = {"n_kept": math.ceil(total_chunks * 0.5)}
if "pi0_load_plates_e12f_top50_episodes_bs128_15k" in selected_configs:
    expected["top50_episodes"] = {"n_selected_episodes": 668}
for arm, invariants in expected.items():
    path = sidecars / arm / "frame_signals.parquet"
    summary = json.loads((sidecars / arm / "summary.json").read_text())
    assert summary["dataset_layout"] == "lerobot", (arm, summary["dataset_layout"])
    assert summary["n_episodes"] == 1335, (arm, summary["n_episodes"])
    assert summary["n_chunks"] == total_chunks, (arm, summary["n_chunks"], total_chunks)
    for key, value in invariants.items():
        assert summary[key] == value, (arm, key, summary[key])
    n_expected = summary["n_kept"]
    table = pq.read_table(path, columns=["episode_index", "frame_index", "e12f_drop_score"])
    score = table["e12f_drop_score"].to_numpy()
    assert len(score) == total_frames, (arm, len(score), total_frames)
    assert set(map(float, np.unique(score))) <= {0.0, 1.0}, arm
    n_keep = int((score <= 0.5).sum())
    assert n_keep == n_expected, (arm, n_keep, n_expected)
    print(f"{arm}: {n_keep:,}/{len(score):,} anchor chunks kept")

print(
    f"real ABC LeRobot dataset: episodes={metadata.total_episodes:,} "
    f"frames={total_frames:,} padded_chunk_anchors={total_chunks:,}"
)

for name in selected_configs:
    cfg = config.get_config(name)
    assert cfg.batch_size == 128
    assert cfg.num_train_steps == 15000
    assert cfg.model.action_horizon == 30
    assert cfg.data.repo_id == dataset.name
    print(f"{name}: batch={cfg.batch_size} steps={cfg.num_train_steps} fsdp_default={cfg.fsdp_devices}")
PY

if [[ "${OPENPI_VALIDATE_ONLY:-0}" == 1 ]]; then
  log "validation-only mode complete"
  exit 0
fi

# Use one common normalization file for the controlled comparison. The stats
# script observes the full underlying dataset; the external masks affect the
# training sampler, not the state/action normalization reference.
vanilla=pi0_load_plates_e12f_vanilla_bc_bs128_15k
norm_rel="$dataset_id/norm_stats.json"
vanilla_norm="$repo_root/assets/$vanilla/$norm_rel"
if [[ ! -s "$vanilla_norm" ]]; then
  log "computing common norm stats from 50,000 source frames"
  "$python_bin" scripts/compute_norm_stats.py \
    --config-name "$vanilla" --max-frames 50000
fi
for cfg in "${configs[@]}"; do
  [[ "$cfg" == "$vanilla" ]] && continue
  target="$repo_root/assets/$cfg/$norm_rel"
  mkdir -p "$(dirname "$target")"
  cp "$vanilla_norm" "$target"
done

if [[ "${OPENPI_WANDB:-0}" == 1 ]]; then
  wandb_args=()
else
  wandb_args=(--no-wandb-enabled)
fi

for cfg in "${configs[@]}"; do
  exp_name=${run_tag}
  run_dir="$checkpoint_root/$cfg/$exp_name"
  mkdir -p "$run_dir"
  if find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    start_args=(--resume --no-overwrite)
    log "resuming $cfg from $run_dir"
  else
    start_args=(--overwrite)
    log "starting $cfg at $run_dir"
  fi

  s3_args=()
  if [[ -n "${OPENPI_S3_CHECKPOINT_BASE:-}" ]]; then
    s3_args=(--s3-checkpoint-path "$OPENPI_S3_CHECKPOINT_BASE/$cfg/$exp_name")
  fi

  "$python_bin" scripts/train.py "$cfg" \
    --exp-name "$exp_name" \
    --checkpoint-base-dir "$checkpoint_root" \
    --fsdp-devices "$fsdp_devices" \
    "${start_args[@]}" \
    "${wandb_args[@]}" \
    "${s3_args[@]}" \
    2>&1 | tee -a "$log_root/${cfg}_${exp_name}.log"
  log "completed $cfg"
done

log "all selected load-plates arms completed: ${configs[*]}"
