"""Which arm goes in for the bag grab first, per episode of siemens_simple_d405.

Task: one arm reaches into the bin, picks a bag, places it into the box (the other
arm often assists afterwards). Grippers are 1.0=open / ~0.0=closed, so the grabbing
arm = the one whose gripper first closes (sustained). Also extracts the "pre-descent"
joint configuration: the last quasi-static pose of the grabbing arm before the
motion bout that ends in that first close.

Outputs (to --out-dir):
  results.json  — per-episode: label, confidence, close/onset frames, pre-descent pose
  traces.json   — per-episode 5Hz gripper + arm-speed curves for the viewer charts
  summary.json  — counts + pre-descent joint region (mean/std per joint, per side)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30
CLOSE_THR = 0.5
CLOSE_HOLD = 15        # frames the gripper must stay closed (0.5s) — rejects blips
STILL_SPEED = 0.015    # rad/frame summed over 6 joints: "quasi-static"
STILL_RUN = 3          # consecutive still frames that end the descent bout
TRACE_HZ = 5


def first_close(g: np.ndarray) -> int | None:
    below = g < CLOSE_THR
    for i in np.flatnonzero(below):
        if bool(below[i : i + CLOSE_HOLD].all()):
            return int(i)
    return None


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def descent_onset(speed_sm: np.ndarray, t_close: int, min_disp: float = 0.5) -> int | None:
    """Staging pose: last sustained-still frame before the descent bout ending at t_close.

    Walks backwards, skipping the at-grab pause (the arm hovers ~still while the
    gripper closes), then consumes the descent motion bout (absorbing micro-pauses
    shorter than STILL_RUN), and stops at the first sustained still region once the
    bout has accumulated min_disp radians of motion.
    """
    t = t_close
    while t > 0 and speed_sm[t - 1] < STILL_SPEED:  # at-grab pause
        t -= 1
    disp = 0.0
    while t > 0:
        if speed_sm[t - 1] >= STILL_SPEED:
            disp += speed_sm[t - 1]
            t -= 1
            continue
        run = 1
        while t - run > 0 and speed_sm[t - 1 - run] < STILL_SPEED:
            run += 1
        if run >= STILL_RUN and disp >= min_disp:
            return int(t - 1)
        t -= run  # micro-pause inside the descent: absorb it
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/home/karim/.cache/huggingface/lerobot/siemens_simple_d405/data/chunk-000/file-000.parquet")
    ap.add_argument("--out-dir", default="/nfs_old/karim/webviewer_data/siemens_simple_d405_armgrab")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet, columns=["state", "episode_index"])
    results, traces = [], {}
    for ep, g in df.groupby("episode_index", sort=True):
        s = np.stack(g.state.values)
        lg, rg = s[:, 6], s[:, 13]
        lspd = smooth(np.abs(np.diff(s[:, 0:6], axis=0)).sum(axis=1))
        rspd = smooth(np.abs(np.diff(s[:, 7:13], axis=0)).sum(axis=1))
        lc, rc = first_close(lg), first_close(rg)

        if lc is None and rc is None:
            label, conf, t_close = "unknown", 0.0, None
        elif rc is None or (lc is not None and lc < rc):
            label, t_close = "left", lc
            conf = 1.0 if rc is None else min(1.0, (rc - lc) / FPS / 6.0)
        else:
            label, t_close = "right", rc
            conf = 1.0 if lc is None else min(1.0, (lc - rc) / FPS / 6.0)

        onset = pre_pose = grab_pose = None
        if t_close is not None:
            spd = lspd if label == "left" else rspd
            onset = descent_onset(spd, t_close)
            arm = slice(0, 6) if label == "left" else slice(7, 13)
            pre_pose = s[onset if onset is not None else max(t_close - 90, 0), arm].tolist()
            grab_pose = s[t_close, arm].tolist()

        results.append({
            "episode": int(ep), "label": label, "confidence": round(float(conf), 3),
            "close_left": lc, "close_right": rc, "onset": onset,
            "n_frames": len(s), "pre_descent_pose": pre_pose, "grab_pose": grab_pose,
            "onset_found": onset is not None,
        })
        step = FPS // TRACE_HZ
        traces[int(ep)] = {
            "hz": TRACE_HZ,
            "grip_l": np.round(lg[::step], 3).tolist(),
            "grip_r": np.round(rg[::step], 3).tolist(),
            "spd_l": np.round(lspd[::step] * FPS, 3).tolist(),  # rad/s
            "spd_r": np.round(rspd[::step] * FPS, 3).tolist(),
        }
        if int(ep) % 300 == 0:
            print(f"ep {ep}: {label} conf={conf:.2f}", flush=True)

    labels = [r["label"] for r in results]
    summary = {
        "n": len(results),
        "counts": {k: labels.count(k) for k in ("left", "right", "unknown")},
        "low_confidence_lt_0.5": sum(1 for r in results if r["confidence"] < 0.5),
        "onset_found": sum(1 for r in results if r["onset_found"]),
    }
    for side in ("left", "right"):
        for key in ("pre_descent_pose", "grab_pose"):
            poses = np.array([r[key] for r in results if r["label"] == side and r["onset_found"]])
            if len(poses):
                summary[f"{key}_region_{side}"] = {
                    "n": len(poses),
                    "mean": np.round(poses.mean(axis=0), 4).tolist(),
                    "std": np.round(poses.std(axis=0), 4).tolist(),
                }
    (out / "results.json").write_text(json.dumps(results))
    (out / "traces.json").write_text(json.dumps(traces))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
