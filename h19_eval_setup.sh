#!/usr/bin/env bash
# Setup for the H19 bottles evaluation timing job. One step per line so `set -e` stops at the first
# failure (an `a && b && c` chain masks failures of a and b). Exercised end to end locally with
# H19_SETUP_SMOKE=1 (skips apt and the 24 GB checkpoint download) before every launch.
set -eo pipefail
WD=$PWD
SMOKE=${H19_SETUP_SMOKE:-0}
ABC_PIN=${ABC_PIN:-1b89f1bc}
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
export DEBIAN_FRONTEND=noninteractive
if [ "$SMOKE" != "1" ]; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git curl awscli ffmpeg libegl1 libgl1 libglib2.0-0 > /dev/null; fi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
SIMCHECK='import mujoco, mujoco_warp, warp; v=(mujoco.__version__, mujoco_warp.__version__, warp.__version__); print("sim versions", v); assert v==("3.10.0","3.10.0.1","1.14.0"), v'

echo "== (1) server venv: openpi paper-repro from its own lock; no simulator packages; hosts the HF CLI"
uv venv --python 3.11 --clear
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install "huggingface_hub[cli]"
hf version
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
mkdir -p "$HOME/ckpts"
if [ "$SMOKE" = "1" ]; then
  python -m huggingface_hub.commands.huggingface_cli download --help > /dev/null && echo "smoke: checkpoint download skipped"
else
  # 2026-09-15: anonymous downloads of the 24 GB anchor hit HF 429 rate limits (job 250 FAILED_SETUP).
  # Retry with backoff. HF_TOKEN, when present in the job env, is read by the CLI automatically and
  # lifts the anonymous limit; it is passed from the launcher's environment, never written to a file.
  for attempt in 1 2 3 4; do
    if python -m huggingface_hub.commands.huggingface_cli download uynitsuj/paper-sim-policy-checkpoints --local-dir "$HOME/ckpts/released" > /dev/null; then break; fi
    [ "$attempt" = 4 ] && { echo "[ERROR] anchor download failed after 4 attempts"; exit 1; }
    echo "[setup] anchor download attempt $attempt failed; sleeping $((attempt*90)) s"; sleep $((attempt*90))
  done
  ls "$HOME/ckpts/released"
fi
deactivate

echo "== (2) serial evaluator + scorer venv: abc-rabc paper pin eab8e9b3 from ITS lock"
echo "e14871964779fd0a381e86dd0b0444b078aa860ef12ce2a13b5ce0d40b810048  h19_abc_source_eab8e9b3.tar.gz" | sha256sum -c -
rm -rf "$HOME/abc-paper"; mkdir -p "$HOME/abc-paper"
tar -xzf h19_abc_source_eab8e9b3.tar.gz -C "$HOME/abc-paper"
(
  set -eo pipefail
  cd "$HOME/abc-paper"
  uv venv --python 3.11 --clear .venv
  source .venv/bin/activate
  uv sync --locked
  uv pip install -e "$WD/packages/openpi-client"
  python -c "$SIMCHECK"
  python -c "import openpi_client, torch; print('abc-paper torch', torch.__version__)"
)

echo "== (3) batched venv: eval-speedup $ABC_PIN + openpi (in-process pi0); openpi first, simulator trio pinned last"
rm -rf "$HOME/abc-rabc"
git clone --quiet https://github.com/uynitsuj/abc-rabc "$HOME/abc-rabc"
git -C "$HOME/abc-rabc" checkout --quiet "$ABC_PIN"
(
  set -eo pipefail
  cd "$HOME/abc-rabc"
  uv venv --python 3.11 --clear .venv
  source .venv/bin/activate
  uv sync
  # openpi's full locked dependency set, all groups (chex, pytest, ... are imported by openpi modules at import time)
  (cd "$WD" && uv export --frozen --no-hashes --all-groups --no-emit-project --no-emit-workspace -o /tmp/openpi_lock_requirements.txt)
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -r /tmp/openpi_lock_requirements.txt
  GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps -e "$WD" -e "$WD/packages/openpi-client"
  uv pip install --reinstall "mujoco==3.10.0" "mujoco-warp==3.10.0.1" "warp-lang==1.14.0"
  python -c "$SIMCHECK"
  python -c "import openpi, jax; print('jax', jax.__version__)"
  python -c "import openpi.policies.policy_config, openpi.training.config as c; print('openpi policy_config import ok;', len(c._CONFIGS), 'configs')"
  python -c "import mujoco_warp, warp; import abc_minimal.eval_policy as e; print('abc_minimal import ok; PI0_CAMERA_KEY_MAP', sorted(e.PI0_CAMERA_KEY_MAP))"
)
echo "== setup complete"
