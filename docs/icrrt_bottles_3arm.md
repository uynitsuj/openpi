# icrrt bottles 3-arm curation study (E12 vs random vs vanilla)

Branch: `icrrt/bottles-3arm`. Tests whether E12-model chunk curation (RABC
final-action gate) beats a retention-matched random gate and vanilla BC on the
real bottles task, using the bs128 speedup recipe (`docs/speedup/`).

## The three arms

All arms: pi0 (action_horizon 30) from `pi0_base`, the same 2,747-episode
LeRobot-v3 bottles pool (no `top_shortest_frac` filter — that differs from the
production `pi0_bottles_warpbc_*` configs on purpose, so retention is the only
variable), batch 128, `fsdp_devices=2`, cosine decay over the full 15k steps,
checkpoints at 5k, 10k, and the final step. The current configs use 15k steps;
older 60k experiments predate this recipe.

| config | repo_id (dataset) | gate |
|---|---|---|
| `pi0_bottles_e12study_vanilla_bs128` | `..._d405_v021_sss45` | none (`rabc_enabled=False`) |
| `pi0_bottles_e12study_e12rabc_thr100_nomax_bs128` | `..._d405_v021_e12rabc` | final-action, thr 1.0 strict, clip_max=inf (kept chunks loss-weighted by raw velocity) |
| `pi0_bottles_e12study_rndmatch_bs128` | `..._d405_v021_rndmatch` | final-action, thr 1.0, clip_max=1.0 (kept chunks weigh exactly 1) |

## Sidecar alternative (no dataset copies)

`LeRobotVelocitySidecarDataConfig` gates RABC from a standalone parquet
(`episode_index, frame_index, velocity` — the shape of icrrt's
`frame_signals.parquet`) instead of a dataset column, so a new RM scoring is a
config swap rather than a 25 GB dataset copy. `LoadVelocitySidecar` builds
each sample's velocity window at sample time (lerobot-identical tail
clamping) and the subset precompute reads the same sidecar; local paths and
s3:// both work. Verified equivalent to the baked-column arm: identical
valid-index arrays (2,192,771 kept) and 800/800 sampled gate decisions/weights
match `chunk_scores.parquet`.

`pi0_bottles_e12study_e12rabc_sidecar_bs128` is the sidecar twin of the
e12rabc arm — same gate, trains on the unmodified `_sss45` dataset with the
sidecar at
`s3://xdof-internal-research/icrrt/curation/bottles_d405_v021_full/e12_zeroshot/frame_signals.parquet`.

## Datasets

openpi autodetects the velocity column from a fixed name list, so both injected
copies *replace* `warp_rm_signed_magnitude` (and `warp_rm_progress`) in the
data parquets; meta/videos/norm_stats are identical to `_sss45`.

- `_e12rabc`: velocity = E12 (`e12_cotrain_mv`) absolute-head progress,
  interpolated from a stride-10 grid, 45-frame box smooth, d/dt x 1395
  (WARP-RM units). **Verified: the openpi final-action gate over this column
  reproduces the e12_zeroshot curation keeps exactly (keep_frac 0.2830642759).**
- `_rndmatch`: velocity = 2.0 x Bernoulli(p=0.2830642758555165), numpy seed 0
  (realized frame density 0.28325) — retention-matched random control.
- Built by `icrrt/scripts/inject_bottles_train_columns.py` from
  `/nfs_us_2/icrrt/curation/bottles_d405_v021_full/e12_zeroshot/frame_signals.parquet`.

Locations (all three: `data/ meta/ videos/ norm_stats/`):

- S3: `s3://xdof-internal-research/lerobot/put_the_plastic_bottles_in_the_bin_d405_v021_{sss45,e12rabc,rndmatch}/`
- NFS: `_sss45` at `/nfs_us_2/karim/warp/datasets/`, injected `data/`+`meta/` at
  `/nfs_us_2/icrrt/datasets_train/` (videos not duplicated on NFS — reuse
  `_sss45`'s or pull from S3).

## Running on a local cluster (no SkyPilot)

Stage each dataset at `$HF_LEROBOT_HOME/<repo_id>` (default
`~/.cache/huggingface/lerobot/`), e.g.:

```bash
aws s3 sync s3://xdof-internal-research/lerobot/put_the_plastic_bottles_in_the_bin_d405_v021_e12rabc \
  ~/.cache/huggingface/lerobot/put_the_plastic_bottles_in_the_bin_d405_v021_e12rabc
```

Real norm stats are committed under `assets/<config>/<repo_id>/norm_stats.json`
(no compute_norm_stats run needed).

Per arm (needs 8 GPUs with 80 GB — bs128/fsdp2 OOMs on fewer; see
`docs/speedup/PROFILE_LOG.md`):

```bash
export OPENPI_REMAT_POLICY=dots_with_no_batch_dims_saveable  # BEFORE python imports (siglip reads it at import time)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
uv run scripts/train.py pi0_bottles_e12study_e12rabc_thr100_nomax_bs128 \
  --exp-name=e12study_e12rabc_local_$(date +%Y%m%d) \
  --checkpoint_base_dir /path/with/space \
  --s3_checkpoint_path s3://xdof-internal-research/icrrt/policy_ckpts/pi0_bottles_e12study_e12rabc_thr100_nomax_bs128/<exp_name> \
  --overwrite
```

`--s3_checkpoint_path` streams every saved step to S3 (final checkpoints land
under `s3://xdof-internal-research/icrrt/policy_ckpts/<config>/<exp>/<step>/`);
drop the flag to keep checkpoints local only. Add `--no-wandb-enabled` if no
wandb key. The RABC subset precompute caches under
`~/.cache/openpi/rabc_valid_indices/` (velocity columns only, no video decode).

Expected keep rates the loader should report (`sample_weight_zero_frac` ~0 in
subset mode since zeros are pre-filtered): e12rabc keeps 28.31% of chunks,
rndmatch 28.33%, vanilla 100%.

## e12f_ctx24 simulation runs

The current simulation recipe uses pi0 with action horizon 30, batch 128,
`fsdp_devices=2`, eight 80 GB GPUs, a 7.5k-step cosine schedule, and checkpoints
at steps 2500, 5000, and 7499. RABC uses the e12f_ctx24 checkpoint geometry
`[400, 160, 60, 30, 15, 5, 0]`, all wrist cameras, strict final-action velocity
`> 1.0`, and `clip_max=1.0` (binary accepted-sample weights).

| config | dataset | arm |
|---|---|---|
| `pi0_sim_load_plates_e12f_ctx24_sidecar_thr100_max1_bs128_7k5` | `sim_load_the_plates_into_the_dish_rack` | RABC |
| `pi0_sim_throw_bottles_e12f_ctx24_sidecar_thr100_max1_bs128_7k5` | `sim_throw_plastic_bottles_in_bin` | RABC |
| `pi0_sim_lego_blocks_sorting_vanilla_bs128_7k5` | `lego_blocks_sorting_20260720_30hz_mjwarp310_lerobot_v3` | vanilla BC |
| `pi0_sim_lego_blocks_sorting_e12f_ctx24_sidecar_thr100_max1_bs128_7k5` | `lego_blocks_sorting_20260720_30hz_mjwarp310_lerobot_v3` | RABC |

The score outputs are archived at
`s3://xdof-internal-research/icrrt/curation/<dataset>/e12f_ctx24/`; the RABC
configs fetch `frame_signals.parquet` from those locations automatically.
Observed keep rates were 36.65% for plates, 44.08% for bottles, and 31.75% for
Lego. The reusable launcher requires `HF_LEROBOT_HOME` and accepts an optional
checkpoint base directory:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot
scripts/launch_e12f_rabc_train.sh <config> <experiment> /path/to/checkpoints
```

## SkyPilot notes (if relaunching from karim's box)

Managed jobs 1-3 were submitted 2026-08-20 (exp names `e12study_*_20260820`,
pending on us-west-2 8-GPU capacity). Gotchas encountered: `rsync -z` spins
forever on that box — the sky api server must be started with
`PATH=/home/karim/.sky/rsync-shim:$PATH`; AWS SSO expiry breaks sky storage ops
(`aws sso login --sso-session karim-sso`); `sky_utils.py` here already exports
`OPENPI_REMAT_POLICY` and fixes the hardcoded workdir.
