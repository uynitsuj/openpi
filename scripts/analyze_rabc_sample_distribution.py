"""Compare RABC weighting schemes on a dataset by simulating the per-sample
weights every action chunk would receive under each mode.

For each mode (velocity_only, multiplicative, additive, q_threshold variants),
walks the dataset's parquets, integrates rorm_velocity over an action_horizon
window for every valid chunk start, and reports two metrics per Q-decile:

  retained_pct: % of chunks in this decile with weight > eps
  weight_mass_pct: sum-of-weights in this decile / total sum-of-weights

Usage:
  uv run python3 scripts/analyze_rabc_sample_distribution.py \
      --repo-id tshirt_folding_d405_v010_20260420_gop10 \
      --action-horizon 30
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.utils.constants import HF_LEROBOT_HOME

EPS = 1e-6
DEFAULT_DECILES = 10


def _read_dataset(repo_id: str) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Returns (per-episode rorm_velocity arrays, per-episode mean Q)."""
    root = HF_LEROBOT_HOME / repo_id
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet under {root}/data")

    ep_vel: dict[int, list[np.ndarray]] = {}
    ep_q_sum: dict[int, float] = {}
    ep_q_count: dict[int, int] = {}

    for f in parquet_files:
        schema_cols = set(pq.read_schema(f).names)
        vel_col = "repromo_signed_magnitude" if "repromo_signed_magnitude" in schema_cols else "rorm_velocity"
        q_col = "repromo_quality" if "repromo_quality" in schema_cols else "rorm_q"
        if vel_col not in schema_cols:
            raise KeyError(f"velocity column missing in {f}")
        if q_col not in schema_cols:
            raise KeyError(f"quality column missing in {f}")
        cols = ["episode_index", "frame_index", vel_col, q_col]
        t = pq.read_table(f, columns=cols)
        eps = np.asarray(t["episode_index"]).astype(np.int64)
        frames = np.asarray(t["frame_index"]).astype(np.int64)
        vels = np.asarray(t[vel_col]).astype(np.float32)
        qs = np.asarray(t[q_col]).astype(np.float64)
        # Sort by episode then frame to ensure contiguous order (parquet usually is, but be safe).
        order = np.lexsort((frames, eps))
        eps = eps[order]; frames = frames[order]; vels = vels[order]; qs = qs[order]
        # Group by episode.
        unique_eps, starts = np.unique(eps, return_index=True)
        starts = np.append(starts, len(eps))
        for i, e in enumerate(unique_eps):
            seg = vels[starts[i]:starts[i+1]]
            seg_q = qs[starts[i]:starts[i+1]]
            e = int(e)
            ep_vel.setdefault(e, []).append(seg)
            ep_q_sum[e] = ep_q_sum.get(e, 0.0) + float(seg_q.sum())
            ep_q_count[e] = ep_q_count.get(e, 0) + len(seg_q)

    ep_vel_arr = {e: np.concatenate(parts) for e, parts in ep_vel.items()}
    ep_mean_q = {e: ep_q_sum[e] / ep_q_count[e] for e in ep_q_sum}
    return ep_vel_arr, ep_mean_q


def _rank_norm(ep_mean_q: dict[int, float]) -> dict[int, float]:
    ids = list(ep_mean_q.keys())
    means = np.array([ep_mean_q[i] for i in ids])
    order = np.argsort(means)
    n = len(ids)
    norms = np.empty(n)
    if n == 1:
        norms[order[0]] = 1.0
    else:
        norms[order] = np.linspace(0.0, 1.0, n)
    return {ids[i]: float(norms[i]) for i in range(n)}


def _q_min_max(ep_mean_q: dict[int, float]) -> tuple[float, float]:
    arr = np.array(list(ep_mean_q.values()))
    return float(arr.min()), float(arr.max())


def _per_chunk_mean_velocity(vel: np.ndarray, h: int) -> np.ndarray:
    """For an episode of length L, returns L-h+1 chunk means."""
    if len(vel) < h:
        return np.zeros((0,), dtype=np.float32)
    csum = np.cumsum(vel, dtype=np.float64)
    csum = np.concatenate([[0.0], csum])
    chunk_sums = csum[h:] - csum[:-h]
    return (chunk_sums / h).astype(np.float32)


def _threshold_from_q_norm(
    q_norm: float, low: float, high: float, shape: str, center: float, steepness: float,
) -> float:
    if shape == "linear":
        return low + (high - low) * q_norm
    if shape == "sigmoid":
        x = (q_norm - center) * steepness
        sig = 1.0 / (1.0 + math.exp(-x))
        return high + (low - high) * (1.0 - sig)
    raise ValueError(f"unknown shape {shape!r}")


def _scheme_weight(
    chunk_mean_vel: np.ndarray,
    ep_mean_q: float,
    ep_q_norm_value: float,
    ep_q_norm_rank: float,
    q_min: float,
    q_max: float,
    mode: str,
    shape: str = "linear",
    center: float = 0.5,
    steepness: float = 10.0,
    low: float = 1.0,
    high: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    threshold: float | None = None,
) -> np.ndarray:
    """Per-chunk weight under the given scheme. Mirrors ComputeRABCWeights."""
    n = len(chunk_mean_vel)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if mode == "velocity_only":
        if threshold is not None:
            w = np.where(chunk_mean_vel < threshold, 0.0, np.minimum(chunk_mean_vel, clip_max))
        else:
            w = np.clip(chunk_mean_vel, clip_min, clip_max)
        return w.astype(np.float32)
    if mode == "multiplicative":
        denom = max(q_max - q_min, 1e-8)
        q_norm = float(np.clip((ep_mean_q - q_min) / denom, 0.0, 1.0))
        v_w = np.clip(chunk_mean_vel, clip_min, clip_max)
        return (v_w * q_norm).astype(np.float32)
    if mode == "additive":
        denom = max(q_max - q_min, 1e-8)
        q_norm = float(np.clip((ep_mean_q - q_min) / denom, 0.0, 1.0))
        v_w = np.clip(chunk_mean_vel, clip_min, clip_max)
        return (0.5 * (v_w + q_norm)).astype(np.float32)
    if mode == "q_threshold":
        thr = _threshold_from_q_norm(ep_q_norm_rank, low, high, shape, center, steepness)
        w = np.where(chunk_mean_vel < thr, 0.0, np.minimum(chunk_mean_vel, clip_max))
        return w.astype(np.float32)
    raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--action-horizon", type=int, default=30)
    p.add_argument("--clip-max", type=float, default=1.0)
    p.add_argument("--clip-min", type=float, default=0.0)
    p.add_argument("--velocity-only-threshold", type=float, default=None,
                   help="Optional threshold for velocity_only scheme.")
    p.add_argument("--deciles", type=int, default=DEFAULT_DECILES)
    p.add_argument("--out", type=str, default=None,
                   help="Optional JSON output path.")
    args = p.parse_args()

    print(f"reading {args.repo_id}…")
    ep_vel, ep_mean_q = _read_dataset(args.repo_id)
    print(f"  {len(ep_vel)} episodes, {sum(len(v) for v in ep_vel.values())} frames")

    q_min, q_max = _q_min_max(ep_mean_q)
    print(f"  episode-mean Q range: [{q_min:.4f}, {q_max:.4f}]")

    rank_norm = _rank_norm(ep_mean_q)

    schemes = [
        ("velocity_only", dict(mode="velocity_only", threshold=args.velocity_only_threshold)),
        ("multiplicative", dict(mode="multiplicative")),
        ("q_threshold_linear", dict(mode="q_threshold", shape="linear")),
        ("q_threshold_sig_top5", dict(mode="q_threshold", shape="sigmoid", center=0.95, steepness=25.0)),
        ("q_threshold_sig_top10", dict(mode="q_threshold", shape="sigmoid", center=0.90, steepness=20.0)),
        ("q_threshold_sig_top25", dict(mode="q_threshold", shape="sigmoid", center=0.75, steepness=15.0)),
    ]

    # Bucket episodes into deciles by mean Q (rank-based, so each decile has ~equal episode count).
    sorted_eps = sorted(ep_mean_q.keys(), key=lambda e: ep_mean_q[e])
    n_eps = len(sorted_eps)
    decile_of: dict[int, int] = {}
    for i, e in enumerate(sorted_eps):
        d = min(args.deciles - 1, int(args.deciles * i / n_eps))
        decile_of[e] = d

    # Walk every chunk and compute scheme weights.
    print(f"computing per-chunk weights for {len(schemes)} schemes…")
    decile_chunk_count = np.zeros(args.deciles, dtype=np.int64)
    decile_retained_count = {name: np.zeros(args.deciles, dtype=np.int64) for name, _ in schemes}
    decile_weight_mass = {name: np.zeros(args.deciles, dtype=np.float64) for name, _ in schemes}
    total_chunks = 0

    for ep, vel in ep_vel.items():
        chunks = _per_chunk_mean_velocity(vel, args.action_horizon)
        if len(chunks) == 0:
            continue
        d = decile_of[ep]
        decile_chunk_count[d] += len(chunks)
        total_chunks += len(chunks)
        rn = rank_norm[ep]
        mq = ep_mean_q[ep]
        for name, params in schemes:
            w = _scheme_weight(
                chunks, mq, mq, rn, q_min, q_max,
                clip_min=args.clip_min, clip_max=args.clip_max,
                **params,
            )
            decile_retained_count[name][d] += int((w > EPS).sum())
            decile_weight_mass[name][d] += float(w.sum())

    # Print report.
    print(f"\ntotal chunks (action_horizon={args.action_horizon}): {total_chunks}")
    print(f"chunks per decile: {decile_chunk_count.tolist()}")

    headers = ["decile", "n_chunks"] + [n for n, _ in schemes]
    print()
    print(" | ".join(f"{h:>22s}" if i >= 2 else f"{h:>8s}" for i, h in enumerate(headers)))
    print("-" * (10 + 12 + 24 * len(schemes)))
    print("retained chunks (weight > eps):")
    for d in range(args.deciles):
        row = [f"{d:>8d}", f"{decile_chunk_count[d]:>8d}"]
        for name, _ in schemes:
            n = decile_retained_count[name][d]
            pct = 100.0 * n / max(decile_chunk_count[d], 1)
            row.append(f"{n:>10d} ({pct:5.1f}%)")
        print(" | ".join(row))

    print()
    print("weight mass (sum of weights in this decile / dataset total):")
    for name, _ in schemes:
        total_mass = decile_weight_mass[name].sum()
        decile_weight_mass[name] = decile_weight_mass[name] / max(total_mass, 1e-12)
    for d in range(args.deciles):
        row = [f"{d:>8d}", f"{decile_chunk_count[d]:>8d}"]
        for name, _ in schemes:
            row.append(f"{100.0 * decile_weight_mass[name][d]:>16.2f}%")
        print(" | ".join(row))

    if args.out:
        out = {
            "repo_id": args.repo_id,
            "action_horizon": args.action_horizon,
            "clip_min": args.clip_min,
            "clip_max": args.clip_max,
            "deciles": args.deciles,
            "n_episodes": n_eps,
            "total_chunks": int(total_chunks),
            "chunks_per_decile": decile_chunk_count.tolist(),
            "schemes": {},
            "q_range": [q_min, q_max],
        }
        for name, _ in schemes:
            out["schemes"][name] = {
                "retained_count_per_decile": decile_retained_count[name].tolist(),
                "weight_mass_frac_per_decile": decile_weight_mass[name].tolist(),
            }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote report to {args.out}")


if __name__ == "__main__":
    main()
