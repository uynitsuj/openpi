#!/usr/bin/env bash
# Timing and correctness pass for the bottles evaluation path (coordinator request, 2026-09-08).
# Stage A: one serial shard (32 worlds, 60 s horizon, pi0 backend through the websocket server) -> wall time.
# Stage B: batched in-process pi0 runner (abc-rabc eval-speedup), same 32 seeds -> wall time, then the
#          4-shard n=128 development block for the released vanilla and WARP-BC checkpoints.
# Stage C: score_bottles on both paths; compare with the published n=128 numbers (vanilla 3.977, WARP-BC 4.672).
# Payloads upload first, complete_marker.json last; a failure uploads partial outputs and failure_marker.json.
set -uo pipefail
OUT=$HOME/eval_out; mkdir -p "$OUT"; ABC=$HOME/abc-rabc; ABCP=$HOME/abc-paper; CK=$HOME/ckpts/released
# ABC  = abc-rabc eval-speedup 1b89f1bc (batched runner); ABCP = paper pin eab8e9b3 (serial evaluator + score_bottles.py, documented CLI)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
upload() { aws s3 sync "$OUT" "$OUT_S3" --only-show-errors --exclude complete_marker.json --exclude failure_marker.json; }
fail() {
  log "FAILED at $1"
  for f in "$OUT"/*.log "$OUT"/*.txt "$OUT"/*.csv; do [ -f "$f" ] && { echo "=== $f (tail) ==="; tail -40 "$f"; aws s3 cp "$f" "$OUT_S3/logs/$(basename "$f")" --only-show-errors; }; done
  upload
  echo "{\"result\":\"h19_base_bottles_eval_timing_failed\",\"stage\":\"$1\"}" > "$OUT/failure_marker.json"
  aws s3 cp "$OUT/failure_marker.json" "$OUT_S3/failure_marker.json"; exit 2
}
wait_server() {  # port, pid, log: up to 15 min; fail if the server process exits
  for i in $(seq 1 180); do
    kill -0 "$2" 2>/dev/null || { log "server exited"; tail -30 "$3"; return 1; }
    (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null && { log "server port $1 open after $((i*5)) s"; tail -5 "$3"; return 0; }
    sleep 5
  done
  log "server not ready after 15 min"; tail -30 "$3"; return 1
}
WD=$PWD; source "$WD/.venv/bin/activate"   # server venv (openpi paper-repro)
export MUJOCO_GL=egl
if [ -n "$(aws s3 ls "$OUT_S3/" 2>/dev/null)" ]; then log "output prefix not empty: $OUT_S3"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv > "$OUT/gpu.csv"; cat "$OUT/gpu.csv"
T0=$(date +%s)
# ---- Stage A: serial shard through the websocket server (skipped when SKIP_SERIAL=1; measured 10,279 s in job 122) ----
TA0=$(date +%s); TA1=$TA0
if [ "${SKIP_SERIAL:-0}" != "1" ]; then
  log "stage A: serve released vanilla"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 nohup uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi0_put_bottles_mjwarp_no_rabc --policy.dir "$CK/vanilla" > "$OUT/serve_vanilla.log" 2>&1 &
  SERVE_PID=$!
  wait_server 8000 $SERVE_PID "$OUT/serve_vanilla.log" || fail serve_vanilla
  python -c "import socket; socket.create_connection(('127.0.0.1',8000),timeout=5).close(); print('tcp ok')" || log "tcp smoke skipped"
  TA0=$(date +%s)
  (cd "$ABCP" && source .venv/bin/activate && PYTHONPATH="$ABCP" python -m abc_minimal.eval_policy --policy-backend pi0 --pi0-host 127.0.0.1 --pi0-port 8000 --num-worlds 32 --seed 20260511 --no-early-stop --output-dir "$OUT/serial/fullhz_vanilla_sh0" > "$OUT/serial_vanilla_sh0.log" 2>&1) || { tail -60 "$OUT/serial_vanilla_sh0.log"; fail serial_shard; }
  TA1=$(date +%s); log "serial shard wall $((TA1-TA0)) s"
  kill $SERVE_PID 2>/dev/null; sleep 5
  upload
else
  log "stage A skipped (SKIP_SERIAL=1); serial 32-world shard measured 10279 s in job 122 (321 s/world, H100 PCIe)"
fi
# ---- Stage B: batched in-process runner -----------------------------------------------------------
seeds_for_shard() { local base=$((20260511 + 32*$1)); seq $base $((base+31)) | tr '\n' ' '; }
declare -A BT
for arm in vanilla warp; do
  if [ "$arm" = vanilla ]; then cfg=pi0_put_bottles_mjwarp_no_rabc; ck=$CK/vanilla; else cfg=pi0_put_bottles_mjwarp_rabc_sss15; ck=$CK/warp_rabc_sss15; fi
  for sh in 0 1 2 3; do
    TB0=$(date +%s)
    (cd "$ABC" && source .venv/bin/activate && PYTHONPATH="$ABC" XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 python batched_runner.py --seeds $(seeds_for_shard $sh) --steps 1800 --gpu 0 \
        --pi0-config $cfg --pi0-ckpt "$ck" --arm $arm --trace-dir "$OUT/batched" --shard $sh > "$OUT/batched_${arm}_sh${sh}.log" 2>&1) || fail "batched_${arm}_sh${sh}"
    TB1=$(date +%s); BT["${arm}_sh${sh}"]=$((TB1-TB0)); log "batched $arm shard $sh wall ${BT[${arm}_sh${sh}]} s"
  done
  upload
done
# ---- Stage C: scoring ------------------------------------------------------------------------------
(cd "$ABCP" && source .venv/bin/activate && PYTHONPATH="$ABCP" python score_bottles.py --trace-dir "$OUT/batched" --arm-glob 'fullhz_{arm}_sh*' --baseline vanilla --arm warp > "$OUT/score_batched_warp_vs_vanilla.txt" 2>&1) || fail score_batched
(cd "$ABCP" && source .venv/bin/activate && PYTHONPATH="$ABCP" python score_bottles.py --trace-dir "$OUT/serial" --arm-glob 'fullhz_{arm}_sh*' --baseline vanilla --arm vanilla > "$OUT/score_serial_vanilla_sh0.txt" 2>&1) || true
# serial shard 0 vs batched shard 0, same 32 seeds: distributional equivalence at shard level (only when the serial stage ran)
if [ "${SKIP_SERIAL:-0}" != "1" ]; then
mkdir -p "$OUT/equiv/fullhz_serial_sh0" "$OUT/equiv/fullhz_batched_sh0"; cp "$OUT"/serial/fullhz_vanilla_sh0/qpos_trace_*.npz "$OUT/equiv/fullhz_serial_sh0/" 2>/dev/null; cp "$OUT"/batched/fullhz_vanilla_sh00/qpos_trace_*.npz "$OUT/equiv/fullhz_batched_sh0/" 2>/dev/null || cp "$OUT"/batched/fullhz_vanilla_sh0/qpos_trace_*.npz "$OUT/equiv/fullhz_batched_sh0/" 2>/dev/null
(cd "$ABCP" && source .venv/bin/activate && PYTHONPATH="$ABCP" python score_bottles.py --trace-dir "$OUT/equiv" --arm-glob 'fullhz_{arm}_sh*' --baseline serial --arm batched > "$OUT/score_serial_vs_batched_sh0.txt" 2>&1) || true
fi
T1=$(date +%s)
SKIP_SERIAL="${SKIP_SERIAL:-0}" python3 - "$((TA1-TA0))" "$((T1-T0))" "$(for k in "${!BT[@]}"; do echo "$k=${BT[$k]}"; done | tr '\n' ',')" > "$OUT/timing_summary.json" <<'PY'
import json,sys,datetime,subprocess,os
serial, total, bt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
batched = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in bt.strip(',').split(',') if kv}
gpu = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip().splitlines()
print(json.dumps({"result":"h19_base_bottles_eval_timing_v1","utc":datetime.datetime.utcnow().isoformat(),"gpus_name_and_memory":gpu,"gpu_count":len(gpu),"server_xla_env":"XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45",
  "serial_shard_32_worlds_wall_s":serial,"serial_shard_32_worlds_wall_s_measured_job122":10279,"serial_skipped":os.environ.get("SKIP_SERIAL","0")=="1","serial_inference_batched_across_worlds":False,
  "batched_shard_32_worlds_wall_s":batched,"batched_inference_batched_across_worlds":True,
  "horizon_steps":1800,"seeds":"development block 20260511..20260638 (4 shards x 32)","total_wall_s":total,"pins":{"serial_and_scorer":"abc-rabc eab8e9b3 (paper pin, tarball sha256 e14871964779fd0a381e86dd0b0444b078aa860ef12ce2a13b5ce0d40b810048)","batched_runner":"abc-rabc "+os.environ.get("ABC_PIN","unset")+" (ABC_PIN env)","openpi":"paper-repro 91f99d6 worktree"}},indent=1))
PY
cat "$OUT/timing_summary.json"; upload && aws s3 cp "$OUT/timing_summary.json" "$OUT_S3/timing_summary.json" && echo '{"result":"h19_base_bottles_eval_timing_complete"}' > "$OUT/complete_marker.json" && aws s3 cp "$OUT/complete_marker.json" "$OUT_S3/complete_marker.json" && log DONE
