#!/usr/bin/env bash
# Local 8xH100 training for the siemens "simple" D405 job.
# Dataset + norm stats already staged (sky job 17). 10-episode val split via
# val_frac; live val every 1k steps; wandb ENABLED (siemens-industrial-packing).
# Checkpoints: NFS is the durable copy (--keep-period 5000); S3 streams best-effort.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY

PY=/home/karim/openpi/.venv/bin/python
CONFIG=pi05_siemens_simple_d405_bs128
EXP=siemens_simple_d405_pi05_20260901
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts
S3_CKPT=s3://xdof-internal-research/siemens/policy_ckpts/$CONFIG/$EXP

log() { echo "[simple-d405 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

pgrep -f "[t]rain.py pi05" >/dev/null && { log "ERROR: another trainer is running"; exit 1; }
[ -s "assets/$CONFIG/siemens_simple_d405/norm_stats.json" ] || { log "ERROR: norm stats missing"; exit 1; }

log "training $CONFIG exp=$EXP (bs128/fsdp2/15k, val every 1k on 10 held-out eps, wandb on)"
mkdir -p "$CKPT_BASE_DIR"
start=$(date +%s)
PYTHONUNBUFFERED=1 WANDB_ENTITY=karim-el-refai-ucb \
OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 \
    "$PY" scripts/train.py "$CONFIG" \
    --exp-name="$EXP" \
    --overwrite \
    --num-workers 16 \
    --keep-period 5000 \
    --checkpoint_base_dir "$CKPT_BASE_DIR" \
    --s3_checkpoint_path "$S3_CKPT"
rc=$?
end=$(date +%s)

log "final checkpoint sync (train rc=$rc, wall=$(( (end - start) / 60 ))min)"
tries=0
until aws s3 sync "$CKPT_BASE_DIR/$CONFIG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
    tries=$((tries + 1))
    [ "$tries" -ge 200 ] && { log "ERROR: final sync failed after 200 attempts"; break; }
    log "final sync interrupted (attempt $tries; SSO may need refresh); retrying in 120s"
    sleep 120
done
[ "$tries" -lt 200 ] && log "final sync complete"
log "SIMPLE_D405_TRAIN_DONE rc=$rc wall=$(( (end - start) / 60 ))min"
