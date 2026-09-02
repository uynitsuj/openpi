#!/usr/bin/env bash
# Relaunch of the trimmed-v3 training after the 09:47Z NCCL OOM (box was taken by
# another user's 8-GPU run mid-staging). Dataset + norm stats are already staged
# locally — this script only gates on REAL GPU occupancy (nvidia-smi compute apps,
# not a pgrep pattern) and trains. No hard S3 dependency: NFS checkpoints,
# non-fatal S3 stream, bounded final sync.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY

PY=/home/karim/openpi/.venv/bin/python
CFG=pi05_siemens_simple_d405_v3_bs128
EXP=siemens_simple_d405_v3_20260902
REPO=siemens_simple_d405_v3
S3_CKPT=s3://xdof-internal-research/siemens/policy_ckpts/$CFG/$EXP
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts

log() { echo "[v3-run2 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

[ -f "$HOME/.cache/huggingface/lerobot/$REPO/meta/info.json" ] || { log "ERROR: dataset not staged"; exit 1; }
[ -f "assets/$CFG/$REPO/norm_stats.json" ] || { log "ERROR: norm stats missing"; exit 1; }

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
log "GPUs free — V3_TRAIN_START $CFG exp=$EXP"

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

# If we got beaten to the GPUs again (OOM inside 30 min), go back to waiting.
if [ "$rc" -ne 0 ] && [ $(( end - start )) -lt 1800 ]; then
    log "early failure — likely GPU contention; re-entering wait loop"
    exec "$0"
fi

tries=0
until timeout 3600 aws s3 sync "$CKPT_BASE_DIR/$CFG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
    tries=$((tries + 1)); [ "$tries" -ge 200 ] && { log "WARN: final sync gave up (NFS copy intact)"; break; }
    log "final sync retry $tries"; sleep 120
done
log "V3_RUN_DONE rc=$rc wall=$(( (end - start) / 60 ))min"
