#!/usr/bin/env bash
# Trimmed v3 simple-D405 training, gated on the sky conversion (job 24):
# wait for norm-stats marker -> stage dataset + norm stats locally (timeout-guarded,
# retries across auth drops) -> train 15k from pi05_base.
# Training itself has NO hard S3 dependency: checkpoints go to NFS (keep-period 5000),
# the S3 checkpoint stream is non-fatal, and the final sync is bounded best-effort.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY
source "$(dirname "$0")/aws_turbo_env.sh"

PY=/home/karim/openpi/.venv/bin/python
CFG=pi05_siemens_simple_d405_v4_bs128
EXP=siemens_simple_d405_v4_20260903
REPO=siemens_simple_d405_v4
S3_DS=s3://xdof-internal-research/siemens/datasets/$REPO
S3_CKPT=s3://xdof-internal-research/siemens/policy_ckpts/$CFG/$EXP
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts

log() { echo "[v4-run $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

tries=0
until aws s3 ls "$S3_DS/norm_stats/$CFG/norm_stats.json" >/dev/null 2>&1; do
    tries=$((tries + 1)); [ "$tries" -ge 450 ] && { log "ERROR: v3 dataset marker never appeared"; exit 1; }
    sleep 120
done
log "DATASET_BUILT $REPO"

until timeout 1800 aws s3 sync "$S3_DS" "$HOME/.cache/huggingface/lerobot/$REPO" \
        --exclude "norm_stats/*" --only-show-errors; do
    log "dataset sync interrupted/stalled (link or auth); retry 60s"; sleep 60
done
mkdir -p "assets/$CFG/$REPO"
until timeout 300 aws s3 cp "$S3_DS/norm_stats/$CFG/norm_stats.json" "assets/$CFG/$REPO/norm_stats.json" --only-show-errors; do
    log "norm stats fetch interrupted; retry 60s"; sleep 60
done
log "DATASET_READY $REPO"

log "waiting for GPUs to be free (all compute apps gone for 3 consecutive minutes)"
free_checks=0
while [ "$free_checks" -lt 3 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; then
        free_checks=$((free_checks + 1))
    else
        free_checks=0
    fi
    sleep 60
done
log "GPUs free — V4_TRAIN_START $CFG exp=$EXP"

mkdir -p "$CKPT_BASE_DIR"
start=$(date +%s)
PYTHONUNBUFFERED=1 WANDB_ENTITY=karim-el-refai-ucb \
OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 \
    "$PY" scripts/train.py "$CFG" \
    --exp-name="$EXP" \
    --overwrite \
    --num-workers 16 \
    --keep-period 5000 \
    --checkpoint_base_dir "$CKPT_BASE_DIR" \
    --s3_checkpoint_path "$S3_CKPT"
rc=$?
end=$(date +%s)
log "train exit rc=$rc wall=$(( (end - start) / 60 ))min"

# If we got beaten to the GPUs (OOM inside 30 min), go back to waiting.
if [ "$rc" -ne 0 ] && [ $(( end - start )) -lt 1800 ]; then
    log "early failure — likely GPU contention; re-entering wait loop"
    exec "$0"
fi

tries=0
until timeout 3600 aws s3 sync "$CKPT_BASE_DIR/$CFG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
    tries=$((tries + 1)); [ "$tries" -ge 200 ] && { log "WARN: final sync gave up (NFS copy intact)"; break; }
    log "final sync retry $tries"; sleep 120
done
log "V4_RUN_DONE rc=$rc wall=$(( (end - start) / 60 ))min"
