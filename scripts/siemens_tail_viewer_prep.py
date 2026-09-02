"""Prep data for the tail-trim viewer (siemens_tail_viewer.py).

For every episode of a bimanual-YAM LeRobot v3 dataset:
  - run the tail detector (same logic as lerobot_tail_trim.py) -> trim frame + flag
  - export 5 Hz traces (grippers, arm speeds, distance-to-park) + event frames
  - slice a composite left|top|right mp4 of the tail window
    [last_close - PRE_S ... episode end] from the packed chunk videos

Outputs under --out-root:
  results.json   traces.json   clips/episode_XXXXXX.mp4

Run:
    uv run python scripts/siemens_tail_viewer_prep.py \
        --dataset-root ~/.cache/huggingface/lerobot/siemens_simple_d405_v2 \
        --out-root /nfs_old/karim/webviewer_data/siemens_simple_d405_v2_tail
"""

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# import tail_trim directly (not via the yam_data package: its __init__ pulls in
# convert_yam_data, which needs deps this viewer prep doesn't)
sys.path.insert(0, str(Path(__file__).resolve().parent / "yam_data"))
from tail_trim import ARM_JOINTS, FPS, LEFT_GRIP, OPEN_THR, RIGHT_GRIP, detect_trim  # noqa: E402

logger = logging.getLogger("tail.prep")

CLOSE_THR = 0.5
CLOSE_HOLD = 15
SMOOTH_K = 9
PRE_S = 3.0          # context kept before the last gripper close in the clip
TRACE_STRIDE = 6     # 30 fps -> 5 Hz traces
CAMS = ["left_camera-images-rgb", "top_camera-images-rgb", "right_camera-images-rgb"]


def smooth(x: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def last_sustained_close(g: np.ndarray) -> int | None:
    below = g < CLOSE_THR
    idx = np.flatnonzero(np.diff(np.concatenate(([0], below.view(np.int8), [0]))))
    runs = [(s, e) for s, e in zip(idx[::2], idx[1::2]) if e - s >= CLOSE_HOLD]
    return int(runs[-1][1] - 1) if runs else None


def slice_clip(job: dict) -> str | None:
    """One ffmpeg call: seek each packed cam video to the tail window, hstack, encode."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    for src, seek in zip(job["srcs"], job["seeks"]):
        cmd += ["-ss", f"{seek:.3f}", "-t", f"{job['dur']:.3f}", "-i", src]
    cmd += [
        "-filter_complex", "hstack=inputs=3",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-threads", "2", "-movflags", "+faststart", job["dst"],
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return None
    except subprocess.CalledProcessError as e:
        return f"ep {job['ep']}: {e.stderr.strip()[-200:]}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--buffer-s", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-clips", action="store_true", help="only rewrite results/traces json")
    args = ap.parse_args()

    root = args.dataset_root.expanduser()
    out = args.out_root.expanduser()
    (out / "clips").mkdir(parents=True, exist_ok=True)

    frames = pd.concat(
        [pd.read_parquet(p, columns=["state", "episode_index"]) for p in sorted(root.glob("data/chunk-*/file-*.parquet"))],
        ignore_index=True,
    )
    episodes = {int(ep): np.stack(g.state.values).astype(np.float64) for ep, g in frames.groupby("episode_index")}
    park = np.median(np.stack([s[-5:].mean(axis=0) for s in episodes.values()]), axis=0)

    meta = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True
    ).set_index("episode_index")

    results, traces, jobs = {}, {}, []
    for ep, s in sorted(episodes.items()):
        T = len(s)
        lg, rg = s[:, LEFT_GRIP], s[:, RIGHT_GRIP]
        trim, flag = detect_trim(s, park, args.buffer_s, 2.0, 15.0)
        lc = max(
            [x for x in (last_sustained_close(lg), last_sustained_close(rg)) if x is not None],
            default=T - int(12 * FPS),
        )
        clip_start = max(0, lc - int(PRE_S * FPS))
        lspd = smooth(np.abs(np.diff(s[:, ARM_JOINTS[:6]], axis=0)).sum(axis=1))
        rspd = smooth(np.abs(np.diff(s[:, ARM_JOINTS[6:]], axis=0)).sum(axis=1))
        dist = np.abs(s[:, ARM_JOINTS] - park[ARM_JOINTS]).sum(axis=1)
        not_open = np.flatnonzero(~((lg > OPEN_THR) & (rg > OPEN_THR)))
        open_done = int(not_open[-1]) + 1 if len(not_open) else 0

        results[ep] = dict(
            episode=ep, T=T, fps=FPS, flag=flag,
            trim=trim if trim is not None else T,
            tail_s=round((T - trim) / FPS, 2) if trim is not None else 0.0,
            last_close=lc, open_done=open_done, clip_start=clip_start,
        )
        traces[ep] = dict(
            stride=TRACE_STRIDE,
            lg=np.round(lg[::TRACE_STRIDE], 3).tolist(),
            rg=np.round(rg[::TRACE_STRIDE], 3).tolist(),
            lspd=np.round(lspd[::TRACE_STRIDE], 4).tolist(),
            rspd=np.round(rspd[::TRACE_STRIDE], 4).tolist(),
            dist=np.round(dist[::TRACE_STRIDE], 3).tolist(),
        )

        m = meta.loc[ep]
        dst = out / "clips" / f"episode_{ep:06d}.mp4"
        if not args.skip_clips and not dst.exists():
            jobs.append(
                dict(
                    ep=ep, dst=str(dst), dur=(T - clip_start) / FPS,
                    srcs=[
                        str(root / "videos" / cam / f"chunk-{int(m[f'videos/{cam}/chunk_index']):03d}"
                            / f"file-{int(m[f'videos/{cam}/file_index']):03d}.mp4")
                        for cam in CAMS
                    ],
                    seeks=[float(m[f"videos/{cam}/from_timestamp"]) + clip_start / FPS for cam in CAMS],
                )
            )

    (out / "results.json").write_text(json.dumps(results))
    (out / "traces.json").write_text(json.dumps(traces))
    logger.info("wrote results/traces for %d episodes; slicing %d clips...", len(results), len(jobs))

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, err in enumerate(pool.map(slice_clip, jobs), 1):
            if err:
                errors.append(err)
            if i % 200 == 0:
                logger.info("  %d/%d clips (%d errors)", i, len(jobs), len(errors))
    logger.info("done: %d clips, %d errors", len(jobs) - len(errors), len(errors))
    for err in errors[:10]:
        logger.warning("  %s", err)


if __name__ == "__main__":
    main()
