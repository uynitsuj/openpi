#!/usr/bin/env bash
# v3cc (center-crop) pipeline: wait for AWS auth -> launch the cloud export (if not
# already submitted) -> wait for it to SUCCEED and for the val sweep to finish
# (GPU 0 in use until then) -> sync dataset -> train combined v3cc with the live
# val pass -> final S3 sync. Fully self-driving across SSO outages.
#
# Start:  nohup bash scripts/run_v3cc_local.sh > v3cc_local.log 2>&1 &

set -uo pipefail
cd /home/karim/openpi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy 2>/dev/null || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
PY=/home/karim/openpi/.venv/bin/python
SKY=/home/karim/openpi/.venv/bin/sky
SC=/tmp/claude-1010/-home-karim-openpi/a5866512-4371-49a1-ae31-2129c126e6b4/scratchpad
CONFIG=pi05_siemens_packing_abcloader_v3cc_bs128
EXP=siemens_packing_pi05_combined_v3cc_20260829
REPO=industrial_packing_abc224_v3cc
JOBNAME=siemens-abc-export-v3cc
S3_DS="s3://xdof-internal-research/siemens/datasets/$REPO"
S3_CKPT="s3://xdof-internal-research/siemens/policy_ckpts/$CONFIG/$EXP"
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts
LOCAL_DS="$HOME/.cache/huggingface/lerobot/$REPO"

log() { echo "[v3cc $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "0/5 waiting for AWS auth"
until aws sts get-caller-identity >/dev/null 2>&1; do sleep 120; done
log "auth OK"

job_status() {
    timeout 300 "$SKY" jobs queue 2>/dev/null | grep -E "\s$JOBNAME\s" | head -1 \
        | grep -oE "SUCCEEDED|FAILED|CANCELLED|RUNNING|PENDING|STARTING|RECOVERING" | head -1
}

st=$(job_status || true)
if [ -z "${st:-}" ]; then
    log "1/5 launching cloud export $JOBNAME"
    timeout 900 "$SKY" jobs launch sky/convert_siemens_abc_layout_v3cc.yaml -n "$JOBNAME" --yes -d \
        > "$SC/v3cc_launch.log" 2>&1 || log "launch cmd exited nonzero (may still have submitted; will poll)"
else
    log "1/5 export already in queue (status=$st)"
fi

log "waiting for $JOBNAME to SUCCEED"
cycle=0
while true; do
    st=$(job_status || true)
    case "${st:-unknown}" in
        SUCCEEDED) log "export SUCCEEDED"; break ;;
        FAILED|CANCELLED) log "ERROR: export ended $st"; exit 1 ;;
    esac
    cycle=$((cycle + 1))
    [ $((cycle % 10)) -eq 0 ] && log "still waiting on export (status=${st:-unknown})"
    sleep 180
done

log "2/5 waiting for val sweep to release GPU 0"
until grep -q "VAL_EVALS_DONE" "$SC/run_val_evals.log" 2>/dev/null; do
    pgrep -f "[r]un_val_evals2.sh" >/dev/null || { log "val sweep not running (assuming done)"; break; }
    sleep 120
done

log "3/5 syncing $REPO"
TURBO_CFG=$SC/aws_turbo_config
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

NS_FILE="assets/$CONFIG/$REPO/norm_stats.json"
if [ ! -s "$NS_FILE" ]; then
    log "4/5 computing norm stats (50k frames)"
    "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG" --max-frames 50000 || exit 1
    aws s3 cp "$NS_FILE" "$S3_DS/norm_stats/$CONFIG/norm_stats.json" --only-show-errors \
        && log "norm stats uploaded (namespaced)" || log "norm stats upload failed (non-fatal)"
fi

log "5/5 training $CONFIG exp=$EXP (8xH100, bs128/fsdp2/15k, live val every 1k)"
mkdir -p "$CKPT_BASE_DIR"
start=$(date +%s)
OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 \
    "$PY" scripts/train.py "$CONFIG" \
    --exp-name="$EXP" \
    --overwrite \
    --no-wandb-enabled \
    --keep-period 5000  \
    --checkpoint_base_dir "$CKPT_BASE_DIR" \
    --s3_checkpoint_path "$S3_CKPT"
rc=$?
end=$(date +%s)

log "final checkpoint sync (train rc=$rc, wall=$(( (end - start) / 60 ))min)"
tries=0
until aws s3 sync "$CKPT_BASE_DIR/$CONFIG/$EXP" "$S3_CKPT" --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
    tries=$((tries + 1))
    [ "$tries" -ge 200 ] && { log "ERROR: final sync failed after 200 attempts"; break; }
    log "final sync interrupted (attempt $tries); retrying in 60s"
    sleep 60
done
log "DONE rc=$rc wall=$(( (end - start) / 60 ))min"
exit "$rc"
