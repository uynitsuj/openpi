#!/usr/bin/env bash
# Sequential v6dj pair: train recent-only (1353 eps) first, then full (4232 eps),
# both 20k from pi05_base. Stall-aware staging (kill sync on 5-min zero progress,
# retry; no hard timeout — the 30-min timeout livelocked on the bad link window).
# A stage whose final 19999 checkpoint exists on NFS is skipped, so re-exec after
# an early failure never retrains a finished stage.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY
source "$(dirname "$0")/aws_turbo_env.sh"

PY=/home/karim/openpi/.venv/bin/python
S3_BASE=s3://xdof-internal-research/siemens/datasets
S3_CKPT_BASE=s3://xdof-internal-research/siemens/policy_ckpts
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts

log() { echo "[v6dj-pair $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

stage_repo() {  # repo cfg — wait for marker, then stall-aware sync until complete
    local repo=$1 cfg=$2 dst="$HOME/.cache/huggingface/lerobot/$1"
    local tries=0
    until aws s3 ls "$S3_BASE/$repo/norm_stats/$cfg/norm_stats.json" >/dev/null 2>&1; do
        tries=$((tries + 1)); [ "$tries" -ge 450 ] && { log "ERROR: $repo marker never appeared"; return 1; }
        sleep 120
    done
    log "DATASET_BUILT $repo"
    while :; do
        aws s3 sync "$S3_BASE/$repo" "$dst" --exclude "norm_stats/*" --only-show-errors &
        local sp=$! prev=-1 cur
        while kill -0 "$sp" 2>/dev/null; do
            cur=$(du -sb "$dst" 2>/dev/null | cut -f1); cur=${cur:-0}
            if [ "$cur" = "$prev" ]; then log "$repo sync stalled 300s; killing for retry"; kill -9 "$sp"; break; fi
            prev=$cur; sleep 300
        done
        wait "$sp" 2>/dev/null && break
        sleep 30
    done
    find "$dst" \( -name "*.mp4.*" -o -name "*.parquet.*" -o -name "*.json.*" \) -delete
    mkdir -p "assets/$cfg/$repo"
    until timeout 300 aws s3 cp "$S3_BASE/$repo/norm_stats/$cfg/norm_stats.json" "assets/$cfg/$repo/norm_stats.json" --only-show-errors; do
        log "$repo norm stats fetch failed; retry 60s"; sleep 60
    done
    log "DATASET_READY $repo"
}

wait_gpus() {
    log "waiting for GPUs (all compute apps gone for 3 consecutive minutes)"
    local free=0
    while [ "$free" -lt 3 ]; do
        if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; then
            free=$((free + 1))
        else
            free=0
        fi
        sleep 60
    done
}

run_stage() {  # repo cfg exp
    local repo=$1 cfg=$2 exp=$3
    if [ -d "$CKPT_BASE_DIR/$cfg/$exp/19999" ]; then
        log "SKIP $exp (19999 already on NFS)"
        return 0
    fi
    stage_repo "$repo" "$cfg" || return 1
    wait_gpus
    log "TRAIN_START $cfg exp=$exp"
    local start end rc
    start=$(date +%s)
    PYTHONUNBUFFERED=1 WANDB_ENTITY=karim-el-refai-ucb \
    OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 \
        "$PY" scripts/train.py "$cfg" \
        --exp-name="$exp" \
        --overwrite \
        --num-workers 16 \
        --keep-period 5000 \
        --checkpoint_base_dir "$CKPT_BASE_DIR" \
        --s3_checkpoint_path "$S3_CKPT_BASE/$cfg/$exp"
    rc=$?
    end=$(date +%s)
    log "TRAIN_EXIT $exp rc=$rc wall=$(( (end - start) / 60 ))min"
    if [ "$rc" -ne 0 ] && [ $(( end - start )) -lt 1800 ]; then
        log "early failure — likely GPU contention; re-entering wait"
        run_stage "$repo" "$cfg" "$exp"
        return $?
    fi
    local tries=0
    until timeout 3600 aws s3 sync "$CKPT_BASE_DIR/$cfg/$exp" "$S3_CKPT_BASE/$cfg/$exp" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
        tries=$((tries + 1)); [ "$tries" -ge 100 ] && { log "WARN: $exp final sync gave up (NFS intact)"; break; }
        sleep 120
    done
    log "STAGE_DONE $exp rc=$rc"
    return "$rc"
}

run_stage siemens_simple_d405_v6dj_recent pi05_siemens_simple_d405_v6dj_recent_bs128 siemens_simple_d405_v6dj_recent_20k_20260904
run_stage siemens_simple_d405_v6dj pi05_siemens_simple_d405_v6dj_bs128 siemens_simple_d405_v6dj_20k_20260904
log "V6DJ_PAIR_DONE"
