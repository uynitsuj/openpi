#!/usr/bin/env bash
# H19 baseline arms: training node setup for the openpi paper-repro worktree.
# One step per line. H19_SETUP_SMOKE=1 runs the same steps locally in a scratch copy (no apt, no uv install).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if [ "${H19_SETUP_SMOKE:-0}" != "1" ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git curl pkg-config awscli ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev > /dev/null
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv venv --python 3.11 --clear
source .venv/bin/activate
export GIT_LFS_SKIP_SMUDGE=1
uv sync
uv pip install -e .
python -c "import openpi.training.config as c; c.get_config('pi0_put_bottles_mjwarp_h19base_sidecar'); c.get_config('pi0_put_bottles_mjwarp_no_rabc'); c.get_config('pi0_put_bottles_mjwarp_rabc_sss15'); print('configs ok')"
python -c "import jax; print('jax', jax.__version__)"
aws --version
echo "SETUP_OK"
