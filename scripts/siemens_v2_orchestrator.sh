#!/usr/bin/env bash
# Session-proof chain for the Siemens v2 datasets -> training launches.
#
# Watches the two v2 conversion jobs (sky managed jobs 10 = LeRobot, 11 = abc layout,
# both built with the user-picked ZED top crop 435,304,1194x896 bottom-anchored).
# When a conversion SUCCEEDS: stage its dataset locally (metadata only; videos stay
# S3-canonical) and launch the corresponding pi0.5 v2 training as a managed job with
# checkpoints under s3://xdof-internal-research/siemens/policy_ckpts/.
#
# Runs detached via nohup so it survives interactive-session drops.
# Start:  nohup bash scripts/siemens_v2_orchestrator.sh > siemens_v2_orchestrator.log 2>&1 &

set -uo pipefail
cd /home/karim/openpi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy 2>/dev/null || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
export PATH="/home/karim/openpi/.venv/bin:$PATH"
PY=/home/karim/openpi/.venv/bin/python
SKY=/home/karim/openpi/.venv/bin/sky
S3_DS_BASE="s3://xdof-internal-research/siemens/datasets"
S3_CKPT_BASE="s3://xdof-internal-research/siemens/policy_ckpts"
DATE_TAG=20260826

log() { echo "[orch $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

job_status() {
    $SKY jobs queue 2>/dev/null | awk -v id="$1" '$1==id' \
        | grep -oE "FAILED_SETUP|FAILED_PRECHECKS|FAILED_NO_RESOURCE|FAILED_CONTROLLER|SUCCEEDED|FAILED|CANCELLED|CANCELLING|RECOVERING|STARTING|RUNNING|PENDING|SUBMITTED" | head -1
}

launch_lerobot_v2() {
    # Dataset staged for future use; per Karim 2026-08-26 the two trained policies are
    # both abc-loader arms (combined vs ZED-only) — no LeRobot-arm training this round.
    log "staging industrial_packing_yam_v2 (data/meta/norm_stats; no training launch)"
    aws s3 sync "$S3_DS_BASE/industrial_packing_yam_v2" "$HOME/.cache/huggingface/lerobot/industrial_packing_yam_v2" \
        --exclude "videos/*" --only-show-errors || return 1
    mkdir -p assets/pi05_siemens_industrial_packing_v2_bs128/industrial_packing_yam_v2
    cp "$HOME/.cache/huggingface/lerobot/industrial_packing_yam_v2/norm_stats/norm_stats.json" \
       assets/pi05_siemens_industrial_packing_v2_bs128/industrial_packing_yam_v2/norm_stats.json 2>/dev/null || true
}

launch_abc_v2() {
    log "staging industrial_packing_abc224_v2 (bins/meta, no videos)"
    aws s3 sync "$S3_DS_BASE/industrial_packing_abc224_v2" "$HOME/.cache/huggingface/lerobot/industrial_packing_abc224_v2" \
        --exclude "*combined_camera-images-rgb.mp4" --only-show-errors || return 1
    # openpi norm stats are computed per-config on the training workers.
    rm -rf assets/pi05_siemens_packing_abcloader_v2_bs128 assets/pi05_siemens_packing_abcloader_v2_zedonly_bs128
    for cfg in pi05_siemens_packing_abcloader_v2_bs128:combined pi05_siemens_packing_abcloader_v2_zedonly_bs128:zedonly; do
        name="${cfg%%:*}"; tag="${cfg##*:}"
        log "launching $tag policy ($name)"
        $PY sky/launch_training.py \
            --config-name "$name" \
            --exp-name "siemens_packing_pi05_${tag}_v2_$DATE_TAG" \
            --s3-checkpoint-base "$S3_CKPT_BASE" \
            --s3-dataset-path "$S3_DS_BASE/industrial_packing_abc224_v2" \
            --service-provider aws \
            --accelerators A100-80GB:8 H100:8 H200:8 \
            --disable-wandb 2>&1 | grep -E "Job submitted|Checkpoints:|ERROR" | sed "s/^/[orch]   /"
    done
}

retry_arm() {
    # retry_arm <fn> <label>: run fn; on failure retry every 10 min for up to 24h.
    # Survives local AWS SSO expiry: once `aws sso login` restores creds, the next
    # retry succeeds. Training checkpoint uploads run from the cloud workers and
    # never depend on this box's credentials.
    local fn="$1" label="$2" tries=0
    until $fn; do
        tries=$((tries + 1))
        if [ "$tries" -ge 144 ]; then log "ERROR: $label failed after 24h of retries"; return 1; fi
        if ! aws sts get-caller-identity >/dev/null 2>&1; then
            log "$label blocked: AWS credentials expired - run aws sso login (retry in 10m, attempt $tries)"
        else
            log "$label failed (retry in 10m, attempt $tries)"
        fi
        sleep 600
    done
}

log "orchestrator started (pid $$): job 10 -> lerobot v2, job 11 -> abc v2"
done10=0; done11=0
while [ "$done10" = 0 ] || [ "$done11" = 0 ]; do
    if [ "$done10" = 0 ]; then
        st=$(job_status 10)
        case "$st" in
            SUCCEEDED) log "job 10 (lerobot v2 convert) SUCCEEDED"; retry_arm launch_lerobot_v2 "lerobot v2 staging" && log "lerobot v2 dataset staged (no training)"; done10=1 ;;
            FAILED*|CANCELLED) log "job 10 terminal without success: $st — NOT launching lerobot training"; done10=1 ;;
        esac
    fi
    if [ "$done11" = 0 ]; then
        st=$(job_status 11)
        case "$st" in
            SUCCEEDED) log "job 11 (abc v2 export) SUCCEEDED"; retry_arm launch_abc_v2 "abc v2 training launches" && log "both v2 policies submitted"; done11=1 ;;
            FAILED*|CANCELLED) log "job 11 terminal without success: $st — NOT launching abc training"; done11=1 ;;
        esac
    fi
    [ "$done10" = 1 ] && [ "$done11" = 1 ] && break
    sleep 300
done
log "orchestrator done"
