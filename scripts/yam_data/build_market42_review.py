"""Generate a Market42 DAgger review manifest from collection directories.

Encodes an explicit, episode-level batch review policy (documented in the
output's `policy_note`); it is NOT interval-level human scrubbing:

- teleop segments are approved for every completed episode (operator
  corrections are retained even when the episode ultimately failed);
- policy segments are approved only when the operator marked the episode
  successful (`eval_anno.json` scores.success, gello_buttons annotation), so
  autonomous continuation from failed episodes never becomes a target;
- approved intervals are the recorded authority segments trimmed by 5 ms per
  side (float-boundary safety); the converter's ramp + boundary-guard
  exclusion applies on top.

Groups are `<collection>_<YYYYMMDD_HH>` hour blocks; validation groups are
chosen deterministically and evenly within each collection (at least one per
collection). Both image-mode variants should be generated from the same
inputs so cc/tc exports share episodes, groups, and splits exactly.

Example:
    uv run python scripts/yam_data/build_market42_review.py \
      --collection 20260910=/nfs_exp/yiming/data/20260910 \
      --collection 20260911=/nfs_exp/yiming/data/20260911/20260911 \
      --modes center_crop,center_crop,center_crop \
      --assumed-ramp-s 1.5 --reviewer karim \
      --out /path/review_cc.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

CAMERAS = ("top", "left", "right")
EDGE_TRIM_S = 0.005
REQUIRED_FILES = (
    "session_meta.json",
    "metadata.json",
    "eval_anno.json",
    "left.mcap",
    "right.mcap",
    *(f"{camera}_camera-images-rgb.mp4" for camera in CAMERAS),
    *(f"{camera}_camera-timestamp.npy" for camera in CAMERAS),
)


def camera_ts_monotonic(path: Path) -> bool:
    values = np.load(path, allow_pickle=False).astype(np.float64)
    return values.ndim == 1 and len(values) > 0 and np.all(np.isfinite(values)) and np.all(np.diff(values) > 0)


def scan_episode(episode: Path, reviewer: str) -> dict | None:
    """Pre-validate cheaply (so the fail-closed converter doesn't die mid-run)
    and build the review entry. Returns None (with a printed reason) on skip."""
    def skip(reason: str):
        print(f"  SKIP {episode.name}: {reason}")
        return None

    if not any((episode / flag).exists() for flag in ("tmp_write_complete.flag", "write_complete.flag")):
        return skip("recording not finalized")
    missing = [name for name in REQUIRED_FILES if not (episode / name).exists()]
    if missing:
        return skip(f"missing files {missing}")
    if not all(camera_ts_monotonic(episode / f"{camera}_camera-timestamp.npy") for camera in CAMERAS):
        return skip("non-monotonic camera timestamps")
    session = json.loads((episode / "session_meta.json").read_text())
    dagger = session.get("dagger") or {}
    segments = dagger.get("segments") or []
    epoch = dagger.get("started_at")
    if not segments or epoch is None:
        return skip("no dagger segments")
    previous_end = -math.inf
    for segment in segments:
        start, end = float(segment["started_at"]), float(segment["ended_at"])
        if not math.isfinite(start + end) or end <= start or start < previous_end:
            return skip("segments not finite/ordered/disjoint")
        previous_end = end
    anno = json.loads((episode / "eval_anno.json").read_text())
    success = bool(anno.get("scores", {}).get("success"))
    if not anno.get("completed"):
        return skip("episode not marked completed")

    basis = f"operator eval_anno success={success} (annotator={anno.get('annotator')})"
    review = []
    for segment in segments:
        authority = segment["authority"]
        start_s = float(segment["started_at"]) - float(epoch) + EDGE_TRIM_S
        end_s = float(segment["ended_at"]) - float(epoch) - EDGE_TRIM_S
        if end_s <= start_s or start_s < 0:
            continue
        approved = authority == "teleop" or success
        reason = (
            f"batch policy: {'teleop correction retained' if authority == 'teleop' else 'autonomous segment of successful episode'}; {basis}"
            if approved
            else f"batch policy: autonomous segment of failed episode excluded; {basis}"
        )
        review.append(
            {
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "authority": authority,
                "approved": approved,
                "reviewer": reviewer if approved else "",
                "reason": reason,
            }
        )
    return {"path": str(episode), "review": review, "_success": success}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--modes", required=True, help="top,left,right modes, e.g. center_crop,pad,pad")
    parser.add_argument("--prompt", default="Pack one transparent bag into the cardboard box and flatten the bag.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--assumed-ramp-s", type=float, default=1.5)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    modes = dict(zip(CAMERAS, args.modes.split(","), strict=True))
    entries = []
    for spec in args.collection:
        label, _, root = spec.partition("=")
        episodes = sorted(Path(root).glob("episode_*"))
        print(f"{label}: {len(episodes)} episodes in {root}")
        collection_entries = []
        for episode in episodes:
            entry = scan_episode(episode, args.reviewer)
            if entry is not None:
                hour = episode.name.split("_")[1] + "_" + episode.name.split("_")[2][:2]
                entry["group"] = f"{label}_{hour}"
                collection_entries.append(entry)
        # Deterministic, evenly spaced val groups within each collection.
        groups = sorted({entry["group"] for entry in collection_entries})
        n_val = max(1, round(len(groups) * args.val_fraction))
        val_groups = {groups[i] for i in np.linspace(0, len(groups) - 1, n_val * 2 + 1).astype(int)[1::2]}
        for entry in collection_entries:
            entry["split"] = "val" if entry["group"] in val_groups else "train"
        entries.extend(collection_entries)

    approved_s = {"train": {"teleop": 0.0, "policy": 0.0}, "val": {"teleop": 0.0, "policy": 0.0}}
    for entry in entries:
        for interval in entry["review"]:
            if interval["approved"]:
                approved_s[entry["split"]][interval["authority"]] += interval["end_s"] - interval["start_s"]
        del entry["_success"]
    for split, per_authority in approved_s.items():
        print(f"{split}: " + "  ".join(f"{k}={v/60:.1f}min" for k, v in per_authority.items()))
        if min(per_authority.values()) <= 0:
            raise SystemExit(f"{split} split has an authority with zero approved time — adjust val groups")

    manifest = {
        "prompt": args.prompt,
        "image_modes": modes,
        "options": {"allow_subscriber_driven": True, "assumed_command_ramp_s": args.assumed_ramp_s},
        "policy_note": (
            "Generated by build_market42_review.py: episode-level batch review (teleop always approved, "
            "policy segments only for operator-marked successes), NOT interval-level human scrubbing. "
            "assumed_command_ramp_s measured from 288 handoffs on 2026-09-11 (median settle 2ms, max 1.33s); "
            "recordings are subscriber-driven and predate recorder ramp/sync archiving."
        ),
        "episodes": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.out}: {len(entries)} episodes, "
          f"{sum(len(e['review']) for e in entries)} intervals")


if __name__ == "__main__":
    main()
