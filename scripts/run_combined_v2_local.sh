#!/usr/bin/env bash
# Local (8xH100) run of the combined v2 policy — fallback for cloud job 12 (5h of failed GPU provisioning).
# Chain: full abc v2 dataset sync (with retries; the SZ<->S3 link is flaky) ->
# per-config norm stats (all stations) -> train
# pi05_siemens_packing_abcloader_v2_zedonly_bs128 (15k, bs128) -> final S3 sync.
# Checkpoints go to NFS (root disk cannot hold a pi0.5 save cycle) and every
# save streams to S3 via --s3_checkpoint_path.
#
# Start:  nohup bash scripts/run_zedonly_v2_local.sh > zedonly_v2_local.log 2>&1 &

set -uo pipefail
cd /home/karim/openpi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy 2>/dev/null || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
PY=/home/karim/openpi/.venv/bin/python
CONFIG=pi05_siemens_packing_abcloader_v2_bs128
EXP=siemens_packing_pi05_combined_v2_20260826_local
REPO=industrial_packing_abc224_v2
S3_DS="s3://xdof-internal-research/siemens/datasets/$REPO"
S3_CKPT="s3://xdof-internal-research/siemens/policy_ckpts/$CONFIG/$EXP"
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts

log() { echo "[combined-local $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# The zedonly chain already pulled and verified the full dataset; skip S3 when it's
# complete locally so an expired SSO token can't block training.
n_vids_pre=$(ls "$HOME/.cache/huggingface/lerobot/$REPO"/train/*/combined_camera-images-rgb.mp4 2>/dev/null | wc -l)
if [ "$n_vids_pre" -ge 1400 ]; then
    log "1/4 dataset already complete locally ($n_vids_pre videos); skipping S3 sync"
else
    log "1/4 syncing full abc v2 dataset (incl. videos) with retries"
    tries=0
    # 100-way concurrency: the SZ<->S3 link is slow per-stream (~30KB/s) but parallelizes ~3x.
    TURBO_CFG=/tmp/claude-1010/-home-karim-openpi/a5866512-4371-49a1-ae31-2129c126e6b4/scratchpad/aws_turbo_config
    until AWS_CONFIG_FILE="$TURBO_CFG" aws s3 sync "$S3_DS" "$HOME/.cache/huggingface/lerobot/$REPO" --only-show-errors; do
        tries=$((tries + 1))
        [ "$tries" -ge 100 ] && { log "ERROR: dataset sync failed after 100 attempts"; exit 1; }
        log "dataset sync interrupted (attempt $tries); retrying in 60s"
        sleep 60
    done
fi
n_train=$(ls "$HOME/.cache/huggingface/lerobot/$REPO/train" | wc -l)
n_vids=$(ls "$HOME/.cache/huggingface/lerobot/$REPO"/train/*/combined_camera-images-rgb.mp4 2>/dev/null | wc -l)
log "dataset ready: $n_train train episodes, $n_vids videos"
[ "$n_vids" -lt 1400 ] && { log "ERROR: expected ~1427 videos, got $n_vids"; exit 1; }

NS_DIR="assets/$CONFIG/$REPO"
NS_FILE="$NS_DIR/norm_stats.json"
if [ ! -s "$NS_FILE" ] || [ "$(stat -c%s "$NS_FILE")" -lt 200 ]; then
    log "2/4 computing norm stats (50k frames, all stations)"
    "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG" --max-frames 50000 || exit 1
    # namespaced upload so a future cloud run of this config reuses identical stats
    aws s3 cp "$NS_FILE" "$S3_DS/norm_stats/$CONFIG/norm_stats.json" --only-show-errors \
        && log "norm stats uploaded (namespaced)"
else
    log "2/4 norm stats already present"
fi

log "3/4 training $CONFIG exp=$EXP (8xH100, bs128/fsdp2/15k)"
mkdir -p "$CKPT_BASE_DIR"
export OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
start=$(date +%s)
"$PY" scripts/train.py "$CONFIG" \
    --exp-name="$EXP" \
    --overwrite \
    --no-wandb-enabled \
    --keep-period 15000 \
    --checkpoint_base_dir "$CKPT_BASE_DIR" \
    --s3_checkpoint_path "$S3_CKPT"
rc=$?
end=$(date +%s)

log "4/4 final checkpoint sync (train rc=$rc, wall=$(( (end - start) / 60 ))min)"
tries=0
until aws s3 sync "$CKPT_BASE_DIR/$CONFIG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
    tries=$((tries + 1))
    [ "$tries" -ge 100 ] && { log "ERROR: final sync failed after 100 attempts"; break; }
    log "final sync interrupted (attempt $tries); retrying in 60s"
    sleep 60
done
log "DONE rc=$rc wall=$(( (end - start) / 60 ))min"
exit "$rc"
