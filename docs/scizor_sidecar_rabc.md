# SCIZOR sidecar gating for RA-BC

This document describes the paper-faithful SCIZOR (Zhang et al., 2026) sample
gating mode added to openpi's RA-BC training pipeline. It is written so the
relevant paragraphs can be lifted directly into an academic methods section.

Implementation entrypoints:

| Layer | File | Symbol |
|---|---|---|
| Data config | `src/openpi/training/config.py` | `LeRobotScizorSidecarDataConfig` |
| Sample-time score join | `src/openpi/transforms.py` | `LoadScizorSidecar` |
| Gate decision | `src/openpi/transforms.py` | `ComputeRABCWeights(mode="scizor_anchor")` |
| Subset precompute | `src/openpi/training/data_loader.py` | `precompute_valid_indices` (`scizor_anchor` branch) |
| Registered train configs | `src/openpi/training/config.py` | `pi0_yam_tshirt_scizor_sidecar_{110612, 122320}` |

## Methodology (for the paper)

> **SCIZOR per-frame suboptimality scores.** Following Zhang et al. (2026), we
> train a time-progress classifier `f_θ` over a frozen DINOv2-base visual
> backbone. For each frame pair `(I_t, I_{t+T})` drawn from a training
> trajectory the classifier predicts which of five elapsed-time bins
> `B = {[0, 0.5), [0.5, 1), [1, 2), [2, 5), [5, ∞)}` the gap `T` falls into.
> Per-frame suboptimality scores `V_t ∈ [0, 1]` are computed at inference
> with `T = 2.0 s` (paper default): the bin-prediction gap is normalised to
> `[0, 1]` per the paper's `rank_prob_to_score` rule, distributed across the
> `T · fps` constituent transitions by uniform convolution, temporally
> discounted at `γ = 0.5` to fold in future suboptimality, and mixed with
> the trajectory mean at `α = 0.5` (paper Appendix A.5):
>
> ```
> V_t^final = α · V_t^local + (1 - α) · mean_{t'} V_{t'}^local
> ```
>
> All scores are produced once, offline, by a single pass over the training
> dataset, and persisted to a sidecar parquet keyed by `(episode_index,
> frame_index)`. The training dataset itself is **not** modified — see
> *Sidecar decoupling* below.
>
> **Anchor-frame gating in behaviour cloning.** Each behaviour-cloning sample
> drawn from the dataset is keyed to an anchor frame `t` that defines the
> start of an `H`-frame action chunk (we use `H = 30` at 30 fps, i.e. an
> action chunk of ≈ 1.0 s; the SCIZOR score window `T = 2.0 s` is decoupled
> from `H`). Following SCIZOR's reference implementation — specifically the
> filter applied after chunk construction in the released octo training
> pipeline, `octo/octo/data/dataset.py:612` — the sample is **kept** iff
>
> ```
> V_t^final ≤ ε_s
> ```
>
> and **discarded** otherwise, with `ε_s = 0.58` (the paper-unified
> threshold from Appendix A.1). The gate is anchored on `t` only: it does
> *not* aggregate (min / mean / max) over the `H` action frames, because
> the SCIZOR score itself is already a 2 s-lookahead progress signal at
> frame `t`. Filtered samples contribute zero gradient and zero FLOPs: at
> dataloader construction time we precompute the set of anchor-frame
> indices whose weight is positive and wrap the LeRobot dataset in a
> `torch.utils.data.Subset` so the effective batch size, optimiser step
> count, and gradient-noise scale are all preserved.
>
> **Continuous-weight variant.** We additionally implement (but do not use
> by default) a continuous-weight ablation that replaces the hard gate
> with a per-sample weight `w_t = clip(1 - V_t^final, 0, 1)` applied to
> the per-sample BC loss. This is the only deviation from the original
> paper recipe; it is exposed via the `scizor_weight_mode="continuous"`
> config field.
>
> **Sidecar decoupling.** Unlike the reference SCIZOR codebase, which
> assumes the curation labels are interleaved with the training dataset,
> our implementation reads `V_t^final` from an external parquet file and
> joins per training sample via `(episode_index, frame_index)`. This
> serves two purposes: (i) the underlying LeRobot dataset is never
> mutated, so a single dataset can serve any number of SCIZOR (or
> RORM-style) ablation conditions through a config swap; (ii) all
> per-checkpoint provenance — including the SCIZOR model hash, the
> trajectory-mean mix coefficient `α`, the discount `γ`, and the goal
> horizon `T` — is encoded in the sidecar metadata rather than in the
> training data, which is critical for reproducibility of the cross-paper
> comparison.
>
> **Caching and exact reproducibility.** The set of valid (kept) anchor
> indices is precomputed once per `(repo_id, action_horizon, ε_s,
> weight_mode, sidecar_path, sidecar_mtime, sidecar_size)` tuple and
> persisted to `~/.cache/openpi/rabc_valid_indices/`. Swapping SCIZOR
> checkpoints invalidates the cache automatically. A spot-check pass
> re-decodes a small random sample of kept indices through the live
> dataloader and asserts agreement with the precompute, catching any
> flat-indexing drift before training begins.

## Data flow

```
                          ┌──────────────────────────┐
                          │ SCIZOR_Baseline/         │
                          │  curation/video_encoding/│
                          │  score_lerobot.py        │     (run once, offline)
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ scizor_predictions.      │
                          │ parquet                  │     (sidecar; not touched by training)
                          │  episode_index           │
                          │  frame_index             │
                          │  scizor_score            │  ← α-mixed V_t^final
                          │  scizor_score_local      │
                          │  scizor_score_traj_mean  │
                          └────────────┬─────────────┘
                                       │
                                       │  (mounted by config; not in LeRobot repo)
                                       ▼
LeRobot v2.1 dataset ──► LeRobotDataset[k]  ──►  LoadScizorSidecar  ──►  ComputeRABCWeights
   (videos, state,           {episode_index,        joins by                mode=scizor_anchor
    actions; UNTOUCHED)       frame_index, …}      (ep, frame)             w = 1[V_t ≤ ε_s]
                                                                                  │
                                                                                  ▼
                                                                       sample_weights ∈ {0, 1}
                                                                                  │
                                                                                  ▼
                                                                   precompute_valid_indices
                                                                          (subset filter)
                                                                                  │
                                                                                  ▼
                                                                          training loop
                                                                    (only sample_weights=1)
```

## Usage

### 1. Score the dataset (in the SCIZOR uv env)

```bash
cd /path/to/SCIZOR_Baseline
uv run python curation/video_encoding/score_lerobot.py \
    --data-dir   /path/to/lerobot/dataset \
    --model-path /path/to/scizor_ckpt_dir   \
    --out-dir    /path/to/scizor_outputs/<run_name> \
    --image-key  top_camera-images-rgb \
    --goal-time  2.0
```

Produces `<out-dir>/scizor_predictions.parquet`. Re-run with different
SCIZOR checkpoints into different `--out-dir`s to A/B them.

### 2. Train RA-BC with the sidecar (in the openpi uv env)

```bash
cd /path/to/openpi
uv run scripts/train.py \
    --config-name=pi0_yam_tshirt_scizor_sidecar_110612 \
    --exp-name=tshirt_scizor_110612_seed0 \
    --seed=0
```

To register a new sidecar config (different dataset, different checkpoint,
different ε_s), copy one of the `pi0_yam_tshirt_scizor_sidecar_*` blocks
in `src/openpi/training/config.py` and edit:

- `repo_id`                — the LeRobot dataset (unchanged across SCIZOR runs)
- `scizor_sidecar_path`    — the parquet produced in step 1
- `scizor_eps_s`           — the threshold (paper default 0.58)
- `scizor_weight_mode`     — `"binary"` (paper) or `"continuous"` (ablation)

### 3. Inspect the deletion ratio

The first epoch's logs include:

```
[rabc_precompute][scizor] <repo> ε_s=0.58 mode=binary: 12,345/45,678 kept (27.0%) in 1.2s
```

Use this for the paper's transparency table (per-dataset deletion fraction
at the paper-unified ε_s).

## Verification

1. **Smoke test the join.** Pick three random `(episode, frame)` pairs from
   the sidecar parquet; instantiate `LoadScizorSidecar` and a tiny
   `LeRobotDataset`; assert `data["scizor_score"]` matches the parquet
   value to float-precision.
2. **Precompute equivalence.** For a small `episodes=(0, 1, 2)` subset, run
   `precompute_valid_indices`; independently load the sidecar in a notebook
   and apply `score <= ε_s` per anchor frame; assert the two index lists
   match exactly.
3. **End-to-end dry-run.** `uv run scripts/train.py
   --config-name=pi0_yam_tshirt_scizor_sidecar_110612
   --num-train-steps=10 --batch-size=4 --num-workers=0`. Verify
   `sample_weight_zero_frac` in W&B matches the precompute log line.
4. **Cross-validation against SCIZOR's own viz.** Run SCIZOR's
   `curation/scripts/viz_scizor_episodes.py` at `ε_s=0.58` on the same
   dataset and compare the keep ratio to the precompute log. Should agree
   to within float-precision (both sides apply the same predicate to the
   same sidecar values).
