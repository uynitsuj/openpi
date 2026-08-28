#!/usr/bin/env bash
# v3 local pipeline (2026-08-29): wait for sky job 15 (abc224_v3 export, +33 eps)
# -> turbo-sync dataset -> train ZED-only v3 -> train combined v3, sequentially
# on the local 8xH100. Same recipe as v2 (pi0.5, bs128/fsdp2/15k, ckpts to NFS,
# S3 streaming non-blocking, final sync with retries).
#
# Start:  nohup bash scripts/run_v3_local_pipeline.sh > v3_pipeline.log 2>&1 &

set -uo pipefail
cd /home/karim/openpi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy 2>/dev/null || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
PY=/home/karim/openpi/.venv/bin/python
SKY=/home/karim/openpi/.venv/bin/sky
REPO=industrial_packing_abc224_v3
S3_DS="s3://xdof-internal-research/siemens/datasets/$REPO"
LOCAL_DS="$HOME/.cache/huggingface/lerobot/$REPO"
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts
TURBO_CFG=/tmp/claude-1010/-home-karim-openpi/a5866512-4371-49a1-ae31-2129c126e6b4/scratchpad/aws_turbo_config

log() { echo "[v3-pipeline $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "0/3 waiting for sky job 15 (v3 export) to SUCCEED"
cycle=0
while true; do
    line=$(timeout 120 "$SKY" jobs queue 2>/dev/null | grep -E "^\s*15\s" | head -1) || true
    st=$(echo "$line" | grep -oE "SUCCEEDED|FAILED|CANCELLED|RUNNING|PENDING|STARTING|RECOVERING" | head -1)
    if [ -z "${st:-}" ]; then
        # queue hides finished jobs after controller autostop; --refresh reveals them
        line=$(timeout 300 "$SKY" jobs queue --refresh 2>/dev/null | grep -E "^\s*15\s" | head -1) || true
        st=$(echo "$line" | grep -oE "SUCCEEDED|FAILED|CANCELLED|RUNNING|PENDING|STARTING|RECOVERING" | head -1)
    fi
    case "${st:-unknown}" in
        SUCCEEDED) log "job 15 SUCCEEDED"; break ;;
        FAILED|CANCELLED) log "ERROR: job 15 ended $st — aborting pipeline"; exit 1 ;;
    esac
    cycle=$((cycle + 1))
    [ $((cycle % 10)) -eq 0 ] && log "still waiting on job 15 (status=${st:-unknown})"
    sleep 120
done

log "1/3 syncing $REPO from S3"
tries=0
until AWS_CONFIG_FILE="$TURBO_CFG" aws s3 sync "$S3_DS" "$LOCAL_DS" --only-show-errors; do
    tries=$((tries + 1))
    [ "$tries" -ge 200 ] && { log "ERROR: dataset sync failed after 200 attempts"; exit 1; }
    log "dataset sync interrupted (attempt $tries); retrying in 60s"
    sleep 60
done
n_vids=$(ls "$LOCAL_DS"/train/*/combined_camera-images-rgb.mp4 2>/dev/null | wc -l)
log "dataset ready: $n_vids train videos"
[ "$n_vids" -lt 1450 ] && { log "ERROR: expected ~1460 train videos, got $n_vids"; exit 1; }

train_arm() {  # $1=config  $2=exp
    local CONFIG=$1 EXP=$2
    local S3_CKPT="s3://xdof-internal-research/siemens/policy_ckpts/$CONFIG/$EXP"
    local NS_FILE="assets/$CONFIG/$REPO/norm_stats.json"
    if [ ! -s "$NS_FILE" ]; then
        log "[$CONFIG] computing norm stats (50k frames)"
        "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG" --max-frames 50000 || return 1
        aws s3 cp "$NS_FILE" "$S3_DS/norm_stats/$CONFIG/norm_stats.json" --only-show-errors \
            && log "[$CONFIG] norm stats uploaded (namespaced)" \
            || log "[$CONFIG] norm stats upload failed (non-fatal; retried at final sync window)"
    else
        log "[$CONFIG] norm stats already present"
    fi
    log "[$CONFIG] training exp=$EXP (8xH100, bs128/fsdp2/15k)"
    mkdir -p "$CKPT_BASE_DIR"
    local start end rc
    start=$(date +%s)
    OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 \
        "$PY" scripts/train.py "$CONFIG" \
        --exp-name="$EXP" \
        --overwrite \
        --no-wandb-enabled \
        --keep-period 15000 \
        --checkpoint_base_dir "$CKPT_BASE_DIR" \
        --s3_checkpoint_path "$S3_CKPT"
    rc=$?
    end=$(date +%s)
    log "[$CONFIG] train rc=$rc wall=$(( (end - start) / 60 ))min; final checkpoint sync"
    local t=0
    until aws s3 sync "$CKPT_BASE_DIR/$CONFIG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
        t=$((t + 1))
        [ "$t" -ge 200 ] && { log "[$CONFIG] ERROR: final sync gave up after 200 attempts"; break; }
        log "[$CONFIG] final sync interrupted (attempt $t); retrying in 60s"
        sleep 60
    done
    log "[$CONFIG] DONE_ARM rc=$rc wall=$(( (end - start) / 60 ))min"
    return "$rc"
}

log "2/3 ZED-only v3 arm"
train_arm pi05_siemens_packing_abcloader_v3_zedonly_bs128 siemens_packing_pi05_zedonly_v3_20260829
rc1=$?
log "3/3 combined v3 arm"
train_arm pi05_siemens_packing_abcloader_v3_bs128 siemens_packing_pi05_combined_v3_20260829
rc2=$?
log "PIPELINE_DONE rc_zedonly=$rc1 rc_combined=$rc2"
exit $(( rc1 | rc2 ))
