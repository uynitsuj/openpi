# Siemens industrial packing — run ledger

## v2 (2026-08-26): new data + FOV harmonization

The job grew to 1,447 episodes: 1,358 on ZED stations (sz_43/48/50) + 89 on the D405 station
sz_44 (`yam_0_61`). ≥10s filter keeps 1,435. Changes for the v2 datasets:

- **ZED top crop → D405 reference FOV** (78.7°×63.2°, measured on sz_44): window size computed
  per episode from its own intrinsics; placement chosen by Karim in the interactive viewer
  (2026-08-26): **bottom-anchored (bottom margin 0), window center 86.7px right of the principal
  point** — 1194×896 @ (435, 304) on sz_48; the horizontal shift generalizes as an angular offset
  fx·(86.7/729). Side cams NOT cropped — a centered crop cuts off the partner arm / hand-off
  region. Interactive viewer: https://claude.ai/code/artifact/8c694873-0876-4bcc-9d05-fff50c2f165d
- Station-aware top source (`top_camera-images-left_rgb.mp4` on ZED, mono `top_camera-images-rgb.mp4`
  on D405) in both converters.
- Timestamp-unit normalization (`to_ns`): global clock is float seconds, ZED cam ts int64 ns,
  **D405 cam ts float milliseconds** — the old `>1e12 → ns` heuristic would have silently
  produced frozen-frame videos on D405 episodes.

Datasets (built 2026-08-28 with the approved crop): `industrial_packing_yam_v2` (sky job 10,
54m 15s) and `industrial_packing_abc224_v2` (sky job 11, 1h 25m 51s, 5.4GB / 4306 objects) under
`s3://xdof-internal-research/siemens/datasets/`. Karim approved the per-station crops incl. the
sz_43 bin-edge clipping (2026-08-28). The lerobot v2 dataset is an artifact only this round.

**v2 training runs** (all pi0.5 bs128/15k via the abc loader, checkpoints under
`s3://.../siemens/policy_ckpts/<config>/<exp>/`). Cloud jobs 13/14 (ZED-only) were cancelled in
favor of the local box; cloud job 12 (combined) flapped STARTING→PENDING 4× over ~5h of scarce
8-GPU capacity, so the combined run also fell back to the local box (job 12 left queued as a
race — cancel whichever loses):

| run | policy | config / exp | data | result |
|---|---|---|---|---|
| local 8×H100 | ZED-only | `pi05_siemens_packing_abcloader_v2_zedonly_bs128` / `siemens_packing_pi05_zedonly_v2_20260826` | station filter `yam_zed_0_61`, 1346 train eps / 1.62M frames | **SUCCEEDED 2026-08-28**, rc=0, **wall 288 min** (4h48m: 1.2 it/s steady, ckpt saves ~7.5 min each). **All ckpts on S3 & verified byte-exact** (5000/10000 streamed during training; 14999 pushed 00:15–00:19 after Karim's SSO refresh — 42GB in ~4 min: the SZ↔S3 link does ~175MB/s at night with the 100-way config vs ~4MB/s daytime). Local retention: only 14999 kept on NFS (`--keep-period 15000` GC'd 5k/10k). |
| local 8×H100 | combined | `pi05_siemens_packing_abcloader_v2_bs128` / `siemens_packing_pi05_combined_v2_20260826_local` | all 1427 train eps | **SUCCEEDED 2026-08-29**, rc=0, **wall 313 min** (23:24→04:40 local). All 3 ckpts verified full-size on S3 (44.7GB each; 5000/10000 streamed during training, 14999 right after). Dataset-sync step skips S3 when the local copy is complete (expired-auth-proof). |
| sky job 12 | combined | same config / `..._20260826` (no `_local`) | same | CANCELLED 2026-08-29 00:13 local after 6h40m of provisioning flaps (never reached RUNNING) — local run won the race. |

**Post-hoc val losses (2026-08-30)** — model.compute_loss(train=False) on each dataset's 8
held-out val episodes (exporter holds out the *newest* 8 → all-D405 in v2/v3, all-ZED in v1;
ZED-only rows are therefore D405 OOD transfer, not in-dist val). Full numbers in wandb
(karim-el-refai-ucb/siemens-industrial-packing):

| run | @5000 | @10000 | @14999 | read |
|---|---|---|---|---|
| v1 abcloader (in-dist) | 0.0066 | 0.0075 | 0.0090 | steady overfit; best ckpt = 5000 |
| v2 combined (in-dist) | 0.0051 | 0.0050 | 0.0055 | flat — 2× data fixed the overfit |
| v2 ZED-only (OOD) | 0.0120 | 0.0147 | 0.0178 | specializes away from D405 |
| v3 combined (in-dist) | lost | lost | 0.0068 | 5k/10k ckpts lost (SSO expiry killed the S3 stream mid-upload + keep-period GC deleted local) |
| v3 ZED-only (OOD) | 0.0150 | 0.0175 | 0.0203 | as v2; different val eps than v2 (newest-8 moved) |

Lesson: local runs now keep 5k-interval checkpoints on NFS (`--keep-period 5000`) so
intermediates never depend on the S3 stream surviving an auth window. train.py gained an
opt-in live val pass (`val_interval`) — first used by the v3cc center-crop ablation.

**Sampled-action recon MSE (2026-09-01)** — abc-style val metric (`scripts/eval_val_recon.py`):
full `sample_actions` (the serving path, default denoising steps) over the same val-8 splits,
MSE vs GT actions in normalized space, headline = first 14 (real) dims. All 14 surviving
checkpoints; in wandb summaries as `recon_mse_{14,32,arm,gripper}[_d405_ood]_<step>`:

| run | @5000 | @10000 | @14999 | arm / gripper @14999 |
|---|---|---|---|---|
| v1 abcloader (in-dist) | 0.0078 | 0.0067 | 0.0064 | 0.0045 / 0.0178 |
| v2 combined (in-dist) | 0.0065 | 0.0052 | **0.0047** | 0.0041 / 0.0081 |
| v2 ZED-only (OOD) | 0.0165 | 0.0180 | 0.0162 | 0.0126 / 0.0378 |
| v3 combined (in-dist) | lost | lost | 0.0056 | 0.0049 / 0.0099 |
| v3 ZED-only (OOD) | 0.0231 | 0.0216 | 0.0207 | 0.0175 / 0.0396 |
| v3cc center-crop (in-dist) | lost | lost | 0.0055 | 0.0047 / 0.0102 |

Reads: (1) **the metric disagrees with flow val loss on v1** — flow val said overfit after 5k
(0.0066→0.0090) but sampled-action error improves monotonically through 15k, so the final
checkpoint is the right serving choice by the metric that matches serving; (2) gripper error is
2–4× arm error everywhere (worst in v1 and the OOD rows) — the gap the physical-units
decomposition should watch; (3) v2 combined final is the best sampled-action model overall,
with the caveat that v2's val-8 differs from v3's (newest-8 moved); (4) center-crop vs pad
stays a wash (0.0055 vs 0.0056) — still a rollout decision.

**v3cc center-crop ablation (2026-08-30)**: `industrial_packing_abc224_v3cc` (sky job 16, same
episodes + same val 8 as v3, `--resize-mode center_crop`) →
`pi05_siemens_packing_abcloader_v3cc_bs128` / `siemens_packing_pi05_combined_v3cc_20260829`,
local 8×H100, **rc=0 wall 252min**. First run with train.py's live val pass (val_interval=1000):
val 0.0122 @1k → best **0.0068 @8k** → 0.0072 @15k — **parity with padded v3 combined (0.0068)**;
the pad-vs-center-crop choice doesn't move action-prediction val loss; decide on rollouts.
Serving this ckpt requires **center_crop** preprocessing (bair-style), unlike all pad-lineage runs.
Ckpts 5000/10000 lost (the mid-run keep-period patch lost its read race, orbax GC'd them locally
and the S3 streams failed silently); 14999 safe on NFS, pushed to S3 2026-09-01. **SSO gotcha
quantified**: the effective upload window after a token refresh was only ~3.5–5h (absolute session
expiry, not 8h-per-refresh) — never let a checkpoint's only copy depend on an S3 stream.

**v3 (2026-08-29)**: +33 eps (30 D405 sz_44/20260828, 3 ZED on new station sz_04 —
crop formula verified visually, 1224×918 @ (442,282)). Export sky job 15
(`sky/convert_siemens_abc_layout_v3.yaml` → `industrial_packing_abc224_v3`); local pipeline
`scripts/run_v3_local_pipeline.sh` then trains `pi05_siemens_packing_abcloader_v3_zedonly_bs128`
/ `siemens_packing_pi05_zedonly_v3_20260829` and `..._v3_bs128` / `..._combined_v3_20260829`
sequentially. Deployment: `robots_realtime` gained `publish_crop_rect` + `publish_image_key`
(CameraNode) and `configs/yam/yam_bimanual_openpi_policy_sz_zed_siemens.yaml` pins the exact
training crop (sz_48 rect active, sz_43/sz_50 + formula in header; `image_preprocess: pad`).

2026-08-28 also evaluated the abc-rabc `karim/rerender-pipeline` LeRobot converter per Karim's
suggestion: it targets *sim re-render* mcaps (`/left-arm-proprio`, in-mcap h264) — not usable on
DataEngine episodes; its good ideas (direct LeRobot v3.0 write, commanded actions) noted for a
future converter revision. Decision: keep the current pipeline for v2.

Norm stats are per-config (namespaced at `<dataset>/norm_stats/<config>/` on S3) so the ZED-only
arm normalizes over its own subset. Gotchas hit on the way: `sky jobs queue` hides finished jobs
once the controller autostops (orchestrators must poll with that in mind), and the abc dataset
first uploaded to `_v2_v2` (yaml sed double-replace) — server-side moved to the right key.

## v1 (2026-08-25)

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
