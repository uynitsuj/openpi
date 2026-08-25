# Siemens industrial packing — run ledger

**Bottom line (2026-08-25):** both pi0.5 arms trained to completion; final checkpoints at
`s3://xdof-internal-research/siemens/policy_ckpts/<config>/<exp>/14999/params`.
Dataloader verdict: abc's mcap-export loader matches LeRobot throughput exactly (1.7 s/it on
p5.48xlarge, both arms; ~0.85 s/it on the SZ box's 8×H100) — its advantages are the 40× smaller
dataset artifact (2.5 GB vs 100 GB raw / 850 MB LeRobot incl. re-encoded videos), no lerobot
format-version coupling, and commanded-action supervision. Loss curves: LeRobot arm reached
0.0022 @ 14.5k. NOTE the two policies use different action conventions (see table) — serving
must match the arm's convention.

DataEngine job `01a01dc5-bef9-7233-b409-4db2d832ac91` → 728 episodes ("industrial packing",
YAM stations sz_43/sz_48, Aug 20–24 2026). Duration filter ≥10s keeps **716** (the 12 dropped
are exactly the 12 `overall_quality='poor'` episodes). All wall times below are SkyPilot
managed-job durations (us-west-2), TOT = submit→end incl. provisioning, JOB = run script only.

## Dataset builds (2026-08-25) — both SUCCEEDED

| sky job | what | TOT | JOB | output |
|---|---|---|---|---|
| 2 | ABC layout export (`export_abc_layout_job.py`, 48 workers) | 48m 25s | 47m 35s | `s3://xdof-internal-research/siemens/datasets/industrial_packing_abc224` (2.5 GB, 708 train + 8 val) |
| 3 | LeRobot v2.1→v3.0 (`convert_xdof_mcap_job.py`, 24 workers) | 37m 2s | 36m 13s | `s3://xdof-internal-research/siemens/datasets/industrial_packing_yam` (850 MB, 716 eps / 939,326 frames + norm stats) |
| 1 | (failed attempt of job 3's pipeline) | 31m 44s | 30m 40s | FAILED in norm stats: lerobot 0.4.5 v21→v30 migrator double-counts `num_frames` on data-file rollover → corrupt `dataset_from/to_index`. Fixed via single data file (`--data-file-size-in-mb 100000`) + boundary assert. |

## Training runs — pi0.5, bs128, 15k steps, cosine decay, ckpt every 5k

Both to `s3://xdof-internal-research/siemens/policy_ckpts/<config>/<exp>/`, 8×80GB GPUs,
wandb disabled. Two arms differing in dataloader (and its inherited conventions):

| arm | config / exp | data path | conventions |
|---|---|---|---|
| LeRobot | `pi05_siemens_industrial_packing_bs128` / `siemens_packing_pi05_lerobot_20260825` | LeRobotDataset over `industrial_packing_yam` | yam flip joint order, actions = observed state |
| ABC loader | `pi05_siemens_packing_abcloader_bs128` / `siemens_packing_pi05_abcloader_20260825` | `AbcLayoutDataset` (abc-style random access) over `industrial_packing_abc224` | raw joint order, commanded actions |

Attempt log:

- sky job 5 (LeRobot arm): FAILED after JOB 3m 11s — `${WANDB_MODE_ARG}` expansion executed
  `WANDB_MODE=disabled` as a command (exit 127); latent sky_utils bug, only fires with
  `--disable-wandb`. Fixed with `env` prefix in `_build_run_script`.
- sky job 4 (ABC arm): CANCELLED while PENDING (carried the same buggy script).
- relaunched 2026-08-25 ~07:40 UTC+8 as new jobs (IDs TBD) — timings to be filled in on completion:

| sky job | arm | submitted (UTC) | started (RUNNING) | finished | TOT | JOB | result |
|---|---|---|---|---|---|---|---|
| 6 | ABC loader | 2026-08-24 23:39 | 2026-08-25 01:37 | 2026-08-25 09:28 | **9h 49m** (incl. 2h GPU queue) | **~7h 51m** | **SUCCEEDED**. 1.7 s/it — identical to the LeRobot arm (GPU-bound at bs128×8GPU; loader not the bottleneck). Reused shared norm stats. Checkpoints 5000/10000/14999 on S3, 44.7 GB each. |
| 7 | LeRobot | 2026-08-24 23:43 | 2026-08-24 23:52 | 2026-08-25 07:26 | **7h 43m** | **~7h 30m** | **SUCCEEDED**. 1.7 s/it steady; loss 0.008 (step 1.3k) → 0.0022 (step 14.5k). Checkpoints 5000/10000/14999 on S3, 44.7 GB each (params + train_state + assets). |

Local hedge runs (this box, 8×H100-96GB, exp `*_local`): abc arm started 01:19 UTC, spent ~1h
downloading pi05_base from GCS (~3MB/s from SZ; now cached in ~/.cache/openpi), stepped at
**~1.2 it/s (0.85 s/it) — 2× the cloud p5 rate** (abc loader still not the bottleneck), but
**died at the step-10k checkpoint write: root disk full** (a pi0.5 train_state save cycle needs
~75GB transient; box had ~60GB). Step-5k checkpoint is safe on S3 (`.../_local/5000/`, resumable
with `--resume`). Queue script now writes checkpoints to `/nfs_old/karim/siemens_tmp_ckpts`.
Not restarted: cloud job 6 is healthy and would finish at essentially the same time; the repaired
queue (datasets, weights, norm stats all staged) is the standby fallback if a cloud job dies.
