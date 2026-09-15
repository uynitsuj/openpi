# ABC load-plates E12F three-arm training on Lambda

This experiment compares E12F curation against unfiltered behavior cloning on
the 1,335-episode **real** ABC-130k Load Plates train split, converted from the
official release MCAPs to LeRobot v3. The completed runs were prepared from
`karim/industrial-packing-v3` lineage rooted at commit
`8de14b2dfeb706e5a79bc52ce0c805195b99841b`; these configs are carried forward
on that branch and retain its measured batch-128 rematerialization recipe.

## Controlled arms

All arms use pi0 initialized from `gs://openpi-assets/checkpoints/pi0_base/params`,
action horizon 30, global batch 128, cosine decay over 15,000 optimizer steps,
and checkpoints at 5,000, 10,000, and 14,999.

| Config | Training samples |
|---|---|
| `pi0_load_plates_e12f_vanilla_bc_bs128_15k` | Every padded 30-step action-chunk anchor; no reward transform or filter |
| `pi0_load_plates_e12f_top50_chunks_bs128_15k` | Exactly the globally top 50% of padded anchors by E12F final-action reward |
| `pi0_load_plates_e12f_top50_episodes_bs128_15k` | Every padded anchor from the top 668 of 1,335 episodes |

The episode arm is episode-count matched, not frame-count matched. Its final
chunk fraction depends on the lengths of the selected real episodes.

## Why the filtered configs use an anchor sidecar

`icrrt/scripts/curate_top_fraction.py` emits one decision for every action-chunk
start. Its `frame_signals.parquet` contains `velocity` (`1=keep`) and
`e12f_drop_score` (`0=keep`, `1=drop`). The OpenPI configs read
`e12f_drop_score` through `LeRobotScizorSidecarDataConfig` and keep scores at
or below 0.5. This is a generic
`(episode_index, frame_index, score)` anchor join. Anchor lookup is important:
building a 30-frame velocity window and reading its final value would shift the
curation decision by 29 frames.

The vanilla config uses `LeRobotYamDataConfig` and `rabc_enabled=False`, so no
selection transform or zero-weight rejection runs.

All three arms read the same release-aware LeRobot v3 conversion. State and
actions use the established YAM per-arm flipped-joint convention, and actions
come from the commanded/leader action topics rather than being synthesized
from observed state. Serving must apply the corresponding state/action flips.

## Release MCAP conversion

`scripts/yam_data/convert_abc_release_mcap_job.py` is the ABC-release adapter
for `convert_xdof_mcap_job.py`. It reads the bundled `/left-arm-*`,
`/right-arm-*`, `/left-ee-*`, `/right-ee-*`, and camera topics; aligns them to a
30 Hz clock; pads the wrist views; center-crops the top view; writes LeRobot
v2.1; and runs the official local v2.1-to-v3.0 migration. It requires every
state, commanded-action, and camera stream and skips an episode rather than
silently substituting observed state for a missing action.

```bash
cd /home/ubuntu/karim/openpi
.venv/bin/python scripts/yam_data/convert_abc_release_mcap_job.py \
  --input-root /home/ubuntu/karim/datasets/abc130k_real_load_plates_v1/hf_tasks/load_the_plates_into_the_dish_rack/train \
  --output-dir /home/ubuntu/karim/datasets \
  --repo-name abc130k_real_load_plates_lerobot_v1 \
  --task-override "Load the plates into the dish rack" \
  --max-workers 24
```

The converter refuses to overwrite an existing output. After migration it
validates episode totals, frame totals, parquet rows, and LeRobot metadata, and
leaves the v2.1 source at `<repo>_old` until the v3 dataset has been validated
and archived.

## Expected node layout

The queue script defaults to:

```text
/home/ubuntu/karim/datasets/abc130k_real_load_plates_lerobot_v1/
/home/ubuntu/karim/sidecars/real_load_plates_lerobot_v1/e12f_top50/
  top50_action_chunks/frame_signals.parquet
  top50_episodes/frame_signals.parquet
/home/ubuntu/karim/checkpoints/openpi/
/home/ubuntu/karim/logs/openpi_loadplates_e12f/
```

Override these without editing the configs:

```bash
export HF_LEROBOT_HOME=/different/dataset/root
export OPENPI_LOAD_PLATES_E12F_SIDECAR_ROOT=/different/e12f_top50
export OPENPI_CHECKPOINT_ROOT=/different/checkpoints
export OPENPI_LOG_ROOT=/different/logs
export OPENPI_DATA_HOME=/different/openpi_download_cache
```

## Environment

Use a non-Conda Python so torchcodec loads the system FFmpeg libraries:

```bash
cd /path/to/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

Check the exact source and config values:

```bash
git status --short --branch
git rev-parse HEAD
.venv/bin/python - <<'PY'
from openpi.training.config import get_config
for name in (
    "pi0_load_plates_e12f_vanilla_bc_bs128_15k",
    "pi0_load_plates_e12f_top50_chunks_bs128_15k",
    "pi0_load_plates_e12f_top50_episodes_bs128_15k",
):
    c = get_config(name)
    print(name, c.batch_size, c.fsdp_devices, c.num_train_steps)
PY
```

## Launch

The three jobs run sequentially because each uses all eight GPUs:

```bash
cd /path/to/openpi
nohup scripts/run_loadplates_e12f_3arm_lambda.sh \
  > /home/ubuntu/karim/logs/openpi_loadplates_e12f/queue.log 2>&1 &
echo $!
```

The script checks for 1,335 real ABC episodes, derives the exact padded-anchor
count from the LeRobot frame total, verifies exact sidecar retention, verifies
all three configs are batch 128 / 15k, computes one shared set of normalization
statistics from 50,000 source frames, and resumes a non-empty checkpoint
directory rather than overwriting it.

Set `OPENPI_WANDB=1` to enable Weights & Biases. Set
`OPENPI_S3_CHECKPOINT_BASE=s3://bucket/prefix` to stream saved checkpoints to
object storage.

## A100 memory and FSDP

The global batch remains 128 in both cases:

- 8x A100-80GB: the script selects `fsdp_devices=2`, the measured fast recipe
  on this branch (four data-parallel groups, two-way model sharding).
- 8x A100-40GB: it selects `fsdp_devices=8` to reduce per-GPU model, optimizer,
  and EMA state. This is the safer memory layout but should receive a one-step
  smoke test on the actual node before committing to all 15k steps.

Override the choice with `OPENPI_FSDP_DEVICES=2`, `4`, or `8`. The launcher also
exports the required settings before Python imports the model:

```text
OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable
XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
```

Monitor with:

```bash
watch -n 2 nvidia-smi
tail -F /home/ubuntu/karim/logs/openpi_loadplates_e12f/*.log
```
