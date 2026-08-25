#!/usr/bin/env bash
# Local fallback queue for the two Siemens pi0.5 runs on this box's 8xH100.
#
# Policy: wait until ALL GPUs are free (never preempts whatever is running),
# then for each arm: skip if its cloud twin (sky managed job) already SUCCEEDED,
# otherwise stage the dataset + norm stats and run the exact cloud recipe
# (bs128/fsdp2/15k). Checkpoints stream per-save to the same siemens
# policy_ckpts prefix under a "<exp>_local" experiment name (no collision with
# the cloud runs). Timings are logged for the run ledger
# (docs/siemens_packing_runs.md).
#
# Start:  nohup bash scripts/siemens_local_train_queue.sh > local_queue_siemens.log 2>&1 &

set -uo pipefail
cd /home/karim/openpi

# Direct S3 (the SZ proxy drops large transfers); keep localhost reachable for sky.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy 2>/dev/null || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
SKY=/home/karim/openpi/.venv/bin/sky
PY=/home/karim/openpi/.venv/bin/python
S3_CKPT_BASE="s3://xdof-internal-research/siemens/policy_ckpts"
KEEP_PERIOD=15000  # only the final checkpoint is kept locally; every save still streams to S3
# Checkpoints go to NFS: a pi0.5 train_state save cycle needs ~75GB transient
# (previous kept + new tmp + commit), which killed a run on the ~60GB-free root
# disk on 2026-08-25. /nfs_old has ~1.2TB free.
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts

log() { echo "[queue $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

gpus_busy() {
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]
}

wait_for_gpus() {
    while gpus_busy; do
        sleep 600
    done
    log "all GPUs free"
}

cloud_succeeded() {
    # If the sky API is unreachable, assume NOT succeeded (prefer redundancy).
    $SKY jobs queue 2>/dev/null | awk -v id="$1" '$1==id' | grep -q SUCCEEDED
}

run_arm() {
    local sky_id="$1" config="$2" exp="$3" repo="$4" s3ds="$5"

    if cloud_succeeded "$sky_id"; then
        log "SKIP $config: cloud job $sky_id already SUCCEEDED"
        return 0
    fi

    log "staging dataset $repo"
    aws s3 sync "$s3ds" "$HOME/.cache/huggingface/lerobot/$repo" --only-show-errors

    # Norm stats: prefer the copy computed by the cloud worker; else compute here.
    local ns_dir="assets/$config/$repo"
    mkdir -p "$ns_dir"
    if [ ! -s "$ns_dir/norm_stats.json" ] || [ "$(stat -c%s "$ns_dir/norm_stats.json")" -lt 200 ]; then
        if [ -s "$HOME/.cache/huggingface/lerobot/$repo/norm_stats/norm_stats.json" ]; then
            cp "$HOME/.cache/huggingface/lerobot/$repo/norm_stats/norm_stats.json" "$ns_dir/norm_stats.json"
            log "norm stats copied from dataset (cloud-computed)"
        else
            log "computing norm stats locally (50k frames)"
            "$PY" scripts/compute_norm_stats.py --config-name "$config" --max-frames 50000 || return 1
            # Share with the cloud twin so both runs of this arm normalize identically.
            aws s3 cp "$ns_dir/norm_stats.json" "$s3ds/norm_stats/norm_stats.json" --only-show-errors \
                && log "norm stats uploaded to $s3ds/norm_stats/ for cloud reuse"
        fi
    fi

    log "waiting for free GPUs before $config"
    wait_for_gpus

    if cloud_succeeded "$sky_id"; then
        log "SKIP $config: cloud job $sky_id SUCCEEDED while waiting"
        return 0
    fi

    export OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
    local start end rc
    start=$(date +%s)
    log "START $config exp=$exp (local 8xH100, bs128/fsdp2/15k)"
    "$PY" scripts/train.py "$config" \
        --exp-name="$exp" \
        --overwrite \
        --no-wandb-enabled \
        --keep-period "$KEEP_PERIOD" \
        --checkpoint_base_dir "$CKPT_BASE_DIR" \
        --s3_checkpoint_path "$S3_CKPT_BASE/$config/$exp"
    rc=$?
    end=$(date +%s)
    log "final checkpoint sync (rc=$rc)"
    aws s3 sync "$CKPT_BASE_DIR/$config/$exp" "$S3_CKPT_BASE/$config/$exp" --only-show-errors \
        --exclude "*orbax-checkpoint-tmp*"
    log "FINISHED $config exp=$exp rc=$rc wall=$(( (end - start) / 60 ))min"
    return "$rc"
}

log "local queue started (pid $$)"

# ABC arm first: its cloud twin (job 6) has no GPUs, while the LeRobot cloud run
# (job 7) is already training — by the time the ABC arm finishes here, job 7
# should be SUCCEEDED and the LeRobot arm below auto-skips.
run_arm 6 pi05_siemens_packing_abcloader_bs128 siemens_packing_pi05_abcloader_20260825_local \
    industrial_packing_abc224 s3://xdof-internal-research/siemens/datasets/industrial_packing_abc224

run_arm 7 pi05_siemens_industrial_packing_bs128 siemens_packing_pi05_lerobot_20260825_local \
    industrial_packing_yam s3://xdof-internal-research/siemens/datasets/industrial_packing_yam

log "local queue done"
