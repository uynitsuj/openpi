#!/usr/bin/env bash
# Two-stage curriculum orchestrator v2 (2026-09-01), freshest-data variant:
#   1. sky-launch the industrial_packing_yam_v3 build now (static episode pool);
#      wait for its S3 marker, stage locally.
#   2. wait for the running siemens_simple_d405 training to finish.
#   3. THEN re-query de_prod for the simple job (people are actively collecting)
#      so stage 2 trains on the newest episodes; sky-launch that conversion.
#   4. stage 1: 5k on industrial_packing_yam_v3 (from pi05_base) — runs while the
#      stage-2 dataset converts in the cloud.
#   5. wait for the stage-2 dataset, stage locally, then stage 2: 15k fine-tune
#      from the stage-1 4999 checkpoint.
# Checkpoints: NFS durable (--keep-period 5000); S3 best-effort + retried syncs.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY

PY=/home/karim/openpi/.venv/bin/python
SKY() { no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost" uv run sky "$@"; }
SC=/tmp/claude-1010/-home-karim-openpi/a5866512-4371-49a1-ae31-2129c126e6b4/scratchpad
CKPT_BASE_DIR=/nfs_old/karim/siemens_tmp_ckpts
S3_DS=s3://xdof-internal-research/siemens/datasets
S3_CKPT_BASE=s3://xdof-internal-research/siemens/policy_ckpts
CUR_TRAIN_LOG=$SC/simple_d405_train.log

CFG1=pi05_siemens_packing_yam_v3_bs128
EXP1=siemens_packing_yam_v3_5k_20260901
REPO1=industrial_packing_yam_v3

CFG2=pi05_siemens_simple_d405_v2_ft_bs128
EXP2=siemens_simple_d405_v2_ft_20260901
REPO2=siemens_simple_d405_v2

log() { echo "[2stage $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
marker() { aws s3 ls "$S3_DS/$2/norm_stats/$1/norm_stats.json" >/dev/null 2>&1; }

launch_job() {  # yaml jobname [extra args...]
    local yaml=$1 name=$2; shift 2
    local tries=0
    until SKY jobs launch "$yaml" -n "$name" -d --yes "$@"; do
        tries=$((tries + 1))
        [ "$tries" -ge 300 ] && { log "ERROR: giving up launching $name after 15h"; return 1; }
        log "AUTH_WAIT: sky launch $name failed (likely expired SSO — run 'aws sso login'); retry in 180s"
        sleep 180
    done
    log "LAUNCHED $name"
}

wait_marker() {  # cfg repo
    local tries=0
    until marker "$1" "$2"; do
        tries=$((tries + 1))
        [ "$tries" -ge 450 ] && { log "ERROR: $2 marker never appeared (15h)"; return 1; }
        sleep 120
    done
    log "DATASET_BUILT $2"
}

stage_local() {  # cfg repo
    local cfg=$1 repo=$2
    until aws s3 sync "$S3_DS/$repo" "$HOME/.cache/huggingface/lerobot/$repo" \
            --exclude "norm_stats/*" --only-show-errors; do
        log "AUTH_WAIT: local sync of $repo interrupted; retry 120s"; sleep 120
    done
    mkdir -p "assets/$cfg/$repo"
    until aws s3 cp "$S3_DS/$repo/norm_stats/$cfg/norm_stats.json" "assets/$cfg/$repo/norm_stats.json" --only-show-errors; do
        log "AUTH_WAIT: norm stats fetch for $cfg interrupted; retry 120s"; sleep 120
    done
    log "DATASET_READY $repo (local + norm stats)"
}

run_stage() {  # cfg exp
    local cfg=$1 exp=$2 rc start end tries
    log "STAGE_START $cfg exp=$exp"
    mkdir -p "$CKPT_BASE_DIR"
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
    log "STAGE_TRAIN_EXIT $cfg rc=$rc wall=$(( (end - start) / 60 ))min"
    tries=0
    until aws s3 sync "$CKPT_BASE_DIR/$cfg/$exp" "$S3_CKPT_BASE/$cfg/$exp" \
            --exclude "*orbax-checkpoint-tmp*" --only-show-errors; do
        tries=$((tries + 1)); [ "$tries" -ge 200 ] && { log "WARN: final sync gave up (NFS copy intact)"; break; }
        log "final sync retry $tries"; sleep 120
    done
    return $rc
}

# ── 1. yam_v3 build (static pool — start immediately) ──
marker "$CFG1" "$REPO1" || launch_job sky/convert_siemens_yam_v3.yaml siemens-yam-v3-convert || exit 1
wait_marker "$CFG1" "$REPO1" || exit 1
stage_local "$CFG1" "$REPO1"

# ── 2. wait for the current run ──
until grep -aq "SIMPLE_D405_TRAIN_DONE" "$CUR_TRAIN_LOG" 2>/dev/null; do sleep 120; done
log "current run finished: $(grep -ao 'SIMPLE_D405_TRAIN_DONE.*' "$CUR_TRAIN_LOG" | tail -1)"
while pgrep -f "[t]rain.py pi05" >/dev/null; do log "trainer process still up; waiting"; sleep 60; done

# ── 3. NOW pull the freshest episode list and launch the stage-2 dataset build ──
tries=0
until "$SC/qenv/bin/python" "$SC/query_simple_latest.py" > "$SC/query_latest_run.log" 2>&1; do
    tries=$((tries + 1))
    [ "$tries" -ge 30 ] && { log "ERROR: de_prod query keeps failing"; exit 1; }
    log "de_prod query failed (attempt $tries); retry 120s"; sleep 120
done
cp "$SC/job_episodes_simple_d405_latest.csv" job_episodes_simple_d405_v2.csv
N_PASS=$("$SC/qenv/bin/python" -c "
import pandas as pd
df = pd.read_csv('job_episodes_simple_d405_v2.csv')
print(int((df.duration_s >= 10).sum()))")
log "FRESH_QUERY: $(($(wc -l < job_episodes_simple_d405_v2.csv) - 1)) episodes, $N_PASS pass >=10s"
# Backgrounded: an expired SSO token must never block stage 1 (which needs no AWS).
# wait_marker before stage 2 is the real synchronization point for this build.
launch_job sky/convert_siemens_simple_d405_v2.yaml siemens-simple-d405-v2-convert \
    --env MIN_EPISODES=$((N_PASS - 20)) &

# ── 4. stage 1 trains locally while the stage-2 dataset converts in the cloud ──
run_stage "$CFG1" "$EXP1" || { log "ERROR: stage 1 failed — aborting"; exit 1; }
[ -d "$CKPT_BASE_DIR/$CFG1/$EXP1/4999/params" ] \
    || { log "ERROR: stage-1 checkpoint 4999/params missing on NFS"; exit 1; }
log "STAGE1_DONE checkpoint verified"

# ── 5. stage-2 dataset ready → fine-tune ──
wait_marker "$CFG2" "$REPO2" || exit 1
stage_local "$CFG2" "$REPO2"
run_stage "$CFG2" "$EXP2" || { log "ERROR: stage 2 failed"; exit 1; }
log "TWO_STAGE_DONE"
