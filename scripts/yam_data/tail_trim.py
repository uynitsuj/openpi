"""Detect the operator stop-button tail in bimanual YAM episodes.

Empirics from siemens_simple_d405_v2 (2,892 eps): after the final placement the
operator releases both grippers, retracts the arms to a stereotyped park pose
(end-pose std < 0.09 rad across the fleet vs ~0.2 mid-episode), and presses stop
~5.3 s (p50; p95 7.5 s) after the last gripper opening. The tail never re-enters
the workspace (max re-approach toward the workspace 0.75 rad), so everything
after the last gripper opening + a settle buffer is non-task motion, safe to cut
(16.2% of all frames on v2).

Detector: trim_frame = (last frame where NOT both grippers open) + 1 + buffer.
Anchored on the final gripper opening in absolute time — an episode-fraction
gate (e.g. "after 75%") misfires because the tail is ~constant in seconds while
episode length varies (median last close sits at ~79% of the episode).

Guards (episode is kept whole when violated):
  - both grippers open through the final 0.5 s
  - tail length within [min_tail_s, max_tail_s]
  - tail is a pure retract: L1 re-approach away from the park pose stays < 1 rad
  - episode ends parked (< 1 rad from the park pose)

Used at conversion time by convert_xdof_mcap_job.py (--trim-tails, fixed
PARK_POSE_SIMPLE_D405) and as a standalone CLI on a converted LeRobot v3
dataset (fleet-self-calibrated park pose):

  uv run python scripts/yam_data/tail_trim.py \
      --dataset-root ~/.cache/huggingface/lerobot/siemens_simple_d405_v2 \
      --out trim_frames_v2.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FPS = 30
OPEN_THR = 0.9
LEFT_GRIP, RIGHT_GRIP = 6, 13
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

# Fleet-median end pose of siemens_simple_d405_v2 (yam_0_61 stations, 2026-09-02).
# Only the ARM_JOINTS entries are used by detect_trim; gripper entries are padding.
# The guards keep episodes whole when an episode's ending doesn't match this pose,
# so a drifted constant degrades to "no trim", never to a bad trim.
PARK_POSE_SIMPLE_D405 = np.array(
    [-0.1, -0.178, 1.174, 0.249, 1.52, 0.017, 1.0, 0.037, 0.176, 1.173, 0.233, 1.496, 0.046, 1.0]
)


def detect_trim(
    states: np.ndarray,
    park: np.ndarray,
    buffer_s: float = 1.0,
    min_tail_s: float = 2.0,
    max_tail_s: float = 15.0,
) -> tuple[int | None, str]:
    """Return (trim_frame, flag). trim_frame is None when the episode should be kept whole."""
    T = len(states)
    lg, rg = states[:, LEFT_GRIP], states[:, RIGHT_GRIP]
    both_open = (lg > OPEN_THR) & (rg > OPEN_THR)

    if not both_open[-int(0.5 * FPS) :].all():
        return None, "grippers_not_open_at_end"

    not_open = np.flatnonzero(~both_open)
    if not len(not_open):
        return None, "grippers_never_close"
    open_done = int(not_open[-1]) + 1

    tail_s = (T - open_done) / FPS
    if tail_s < min_tail_s:
        return None, "tail_too_short"
    if tail_s > max_tail_s:
        return None, "tail_too_long"

    dist = np.abs(states[:, ARM_JOINTS] - park[ARM_JOINTS]).sum(axis=1)
    if dist[-5:].mean() > 1.0:
        return None, "end_pose_not_parked"
    seg = dist[open_done:]
    if float((seg - np.minimum.accumulate(seg)).max()) > 1.0:
        return None, "tail_reapproaches_workspace"

    trim = min(open_done + int(buffer_s * FPS), T)
    return trim, "ok"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--buffer-s", type=float, default=1.0, help="settle time kept after the last gripper opening")
    ap.add_argument("--min-tail-s", type=float, default=2.0)
    ap.add_argument("--max-tail-s", type=float, default=15.0)
    args = ap.parse_args()

    root = args.dataset_root.expanduser()
    parts = sorted(root.glob("data/chunk-*/file-*.parquet"))
    frames = pd.concat(
        [pd.read_parquet(p, columns=["state", "episode_index"]) for p in parts], ignore_index=True
    )
    episodes = {int(ep): np.stack(g.state.values).astype(np.float64) for ep, g in frames.groupby("episode_index")}

    # self-calibrated park pose: fleet-median of the final 5-frame mean pose
    park = np.median(np.stack([s[-5:].mean(axis=0) for s in episodes.values()]), axis=0)
    logger.info("fleet park pose: %s", np.round(park, 3).tolist())

    rows = []
    for ep, s in sorted(episodes.items()):
        trim, flag = detect_trim(s, park, args.buffer_s, args.min_tail_s, args.max_tail_s)
        rows.append(
            dict(
                episode_index=ep,
                T=len(s),
                trim_frame=trim if trim is not None else len(s),
                tail_s=round((len(s) - trim) / FPS, 2) if trim is not None else 0.0,
                kept_frac=round((trim if trim is not None else len(s)) / len(s), 4),
                flag=flag,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    trimmed = out[out.flag == "ok"]
    logger.info("episodes: %d  trimmed: %d  flagged-kept-whole: %d", len(out), len(trimmed), len(out) - len(trimmed))
    for flag, n in out[out.flag != "ok"].flag.value_counts().items():
        logger.info("  %s: %d", flag, n)
    logger.info(
        "tail_s p5/p50/p95: %.2f / %.2f / %.2f   cut: %.2f h of %.2f h (%.1f%% of frames)",
        *np.percentile(trimmed.tail_s, [5, 50, 95]),
        trimmed.tail_s.sum() / 3600,
        out["T"].sum() / FPS / 3600,
        100 * trimmed.tail_s.sum() * FPS / out["T"].sum(),
    )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
