"""Apply the episode-level batch review policy to an existing Market42 DAgger
export IN PLACE — no video re-decoding (2026-09-11 convention for
metadata/mask-only edits).

For every episode whose operator outcome is success (from the episode's
provenance.json outcome_annotation), the reviewed mask becomes true over all
recorded authority segments (batch policy: corrections and successful
autonomous segments approved). Episodes without a success outcome abort the
run — the current policy excludes failures from exports entirely, so their
presence means the export needs regeneration, not review.

Updates coherently, matching the converter's contracts:
- data.npz: rewritten with the new `reviewed` array (all other arrays intact);
- manifest.json: reviewed_frames, chunks_h30 (via full_chunk_starts), the
  data.npz checksum, review_sha256, and an in_place_edits audit entry;
- review.json: per-episode approved intervals reconstructed from the recorded
  segments (seconds relative to session dagger.started_at), reviewer + reason.

Idempotency: episodes with existing reviewed frames abort unless --force.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi.training.dagger_dataset import AUTHORITY
from openpi.training.dagger_dataset import file_hash
from openpi.training.dagger_dataset import full_chunk_starts
from openpi.training.dagger_dataset import read_json

CODE_TO_AUTHORITY = {code: name for name, code in AUTHORITY.items()}


def replace_json(path: Path, payload) -> None:
    """Deliberate in-place overwrite via atomic rename (write_json is
    exclusive-create by converter design and must stay that way)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def segment_intervals(data, epoch: float) -> list[dict]:
    """Reconstruct per-segment [start_s, end_s) intervals from recorded arrays."""
    intervals = []
    segment = data["segment"]
    timestamps = data["timestamps"]
    for seg_id in sorted({int(s) for s in segment if s >= 0}):
        mask = segment == seg_id
        code = int(data["authority"][mask][0])
        start_s = float(timestamps[mask][0]) - epoch
        end_s = float(timestamps[mask][-1]) + 1 / 30 - epoch
        intervals.append((max(start_s, 0.0), end_s, CODE_TO_AUTHORITY[code]))
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--force", action="store_true", help="Re-apply even if reviewed frames already exist")
    args = parser.parse_args()

    root = args.export.resolve()
    manifest = read_json(root / "manifest.json")
    review = read_json(root / "review.json")
    review_by_id = {Path(entry["path"]).name: entry for entry in review["episodes"]}

    for episode in manifest["episodes"]:
        directory = root / episode["directory"]
        if episode["reviewed_frames"] and not args.force:
            raise SystemExit(f"{episode['id']} already has reviewed frames; use --force to re-apply")
        provenance = read_json(directory / "provenance.json")
        outcome = provenance.get("outcome_annotation") or {}
        if not (outcome.get("scores") or {}).get("success"):
            raise SystemExit(
                f"{episode['id']} has no operator success outcome — failures are excluded from "
                "exports by policy; regenerate the export instead of reviewing in place"
            )
        epoch = float(provenance["session"]["dagger"]["started_at"])

        data = dict(np.load(directory / "data.npz"))
        reviewed = data["authority"] > 0
        data["reviewed"] = reviewed
        np.savez_compressed(directory / "data.npz", **data)

        episode["reviewed_frames"] = int(reviewed.sum())
        episode["chunks_h30"] = {
            name: len(full_chunk_starts(data["valid"] & reviewed & (data["authority"] == code), data["segment"], 30))
            for name, code in AUTHORITY.items()
        }
        episode["checksums"]["data.npz"] = file_hash(directory / "data.npz")

        basis = f"operator eval_anno success=True (annotator={outcome.get('annotator')})"
        review_by_id[episode["id"]]["review"] = [
            {
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "authority": authority,
                "approved": True,
                "reviewer": args.reviewer,
                "reason": f"batch policy applied in place: full recorded {authority} segment; {basis}",
            }
            for start_s, end_s, authority in segment_intervals(data, epoch)
        ]
        print(f"{episode['id']}: reviewed={episode['reviewed_frames']} chunks={episode['chunks_h30']}")

    replace_json(root / "review.json", review)
    manifest["review_sha256"] = file_hash(root / "review.json")
    manifest.setdefault("in_place_edits", []).append(
        {
            "edit": "apply_batch_review",
            "reviewer": args.reviewer,
            "policy": "success-only batch approval over recorded authority segments",
            "tool_sha256": file_hash(Path(__file__)),
        }
    )
    replace_json(root / "manifest.json", manifest)
    totals = {
        name: sum(episode["chunks_h30"][name] for episode in manifest["episodes"]) for name in AUTHORITY
    }
    print(f"updated {root}: totals={totals}")


if __name__ == "__main__":
    main()
