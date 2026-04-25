"""Re-encode LeRobot video dataset with dense keyframes for faster random-access decode.

Default H.264 GOP is ~230 frames (1 keyframe per ~8s at 30fps). Seeking to an
arbitrary frame then requires decoding up to 230 intermediate frames. Setting
``-g 10 -bf 0`` caps worst-case seek at 10 decodes and removes B-frames entirely
— roughly 5x faster per-camera decode in benchmarks.

Works on both v2.1 and v3.0 lerobot datasets — videos are rglob'd, so the
on-disk layout doesn't matter. ``data/`` and ``meta/`` are hardlinked into the
destination so the new dataset is a drop-in with the same per-episode stats and
RORM metadata; only videos are re-encoded content.

Usage (v3.0 dataset):
    python scripts/reencode_dense_keyframes.py \\
        --src ~/.cache/huggingface/lerobot/tshirt_folding_d405_v010_20260420 \\
        --dst ~/.cache/huggingface/lerobot/tshirt_folding_d405_v010_20260420_gop10 \\
        --workers 32
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import tyro
from tqdm import tqdm


@dataclasses.dataclass
class Args:
    src: Path
    """Source dataset root (e.g. ~/.cache/huggingface/lerobot/<repo>)."""
    dst: Path
    """Destination dataset root."""
    gop: int = 10
    """Keyframe interval. openpi benchmarks show 10 is a good sweet spot."""
    crf: int = 23
    """x264 quality (lower = better, larger). 23 matches default; ok for training."""
    preset: str = "veryfast"
    """Encoder preset. veryfast saves ~3x re-encode time vs. medium with minor size hit."""
    workers: int = 32
    """Parallel ffmpeg processes. Cap at nproc to avoid oversubscription."""
    overwrite: bool = False
    """If False (default), skip any destination video that already exists."""


def _ffmpeg_cmd(src: Path, dst: Path, gop: int, crf: int, preset: str) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-g", str(gop), "-bf", "0",
        "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # no audio
        str(dst),
    ]


def _reencode_one(args: tuple[Path, Path, int, int, str, bool]) -> Optional[str]:
    src, dst, gop, crf, preset, overwrite = args
    if dst.exists() and not overwrite:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use ".tmp.mp4" (not ".mp4.tmp") so ffmpeg's format auto-detection still works.
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    try:
        r = subprocess.run(
            _ffmpeg_cmd(src, tmp, gop, crf, preset),
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            return f"FAIL {src.relative_to(src.parents[3])}: {r.stderr.strip()[:200]}"
        tmp.replace(dst)
        return None
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"EXC {src.relative_to(src.parents[3])}: {type(e).__name__}: {e}"


def _hardlink_tree(src_root: Path, dst_root: Path) -> int:
    """Hardlink every file under src_root into the mirrored position under dst_root.

    Parquets and meta jsonls are identical between the sparse and dense datasets;
    hardlinks save disk without duplicating bytes. Returns count of files linked.
    """
    n = 0
    for src_path in src_root.rglob("*"):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            continue
        try:
            os.link(src_path, dst_path)
        except OSError:
            # Cross-device or other: fall back to copy.
            import shutil
            shutil.copy2(src_path, dst_path)
        n += 1
    return n


def main():
    args = tyro.cli(Args)
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"{src} doesn't look like a lerobot dataset (no meta/info.json)")
    if src == dst:
        raise SystemExit("--src and --dst must differ")

    dst.mkdir(parents=True, exist_ok=True)

    # Hardlink parquets + meta. Cheap; enables the dst to be a drop-in dataset.
    for sub in ("data", "meta"):
        if (src / sub).is_dir():
            n = _hardlink_tree(src / sub, dst / sub)
            print(f"[OK] hardlinked {n} files into {dst / sub}")

    # Collect video jobs.
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    if not src_videos.is_dir():
        raise SystemExit(f"no videos dir at {src_videos}")
    jobs: list[tuple[Path, Path, int, int, str, bool]] = []
    for src_file in sorted(src_videos.rglob("*.mp4")):
        rel = src_file.relative_to(src_videos)
        dst_file = dst_videos / rel
        jobs.append((src_file, dst_file, args.gop, args.crf, args.preset, args.overwrite))

    pending = [j for j in jobs if args.overwrite or not j[1].exists()]
    print(f"[INFO] re-encoding {len(pending)} / {len(jobs)} videos "
          f"(gop={args.gop} crf={args.crf} preset={args.preset} workers={args.workers})")
    if not pending:
        print("[OK] nothing to do — destination already complete")
        return

    failures = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for err in tqdm(ex.map(_reencode_one, pending, chunksize=4), total=len(pending)):
            if err:
                failures.append(err)

    if failures:
        print(f"[WARN] {len(failures)} failures; first 5:")
        for f in failures[:5]:
            print(" ", f)
        raise SystemExit(1)
    print(f"[OK] re-encoded {len(pending)} videos → {dst_videos}")


if __name__ == "__main__":
    main()
