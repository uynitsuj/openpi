#!/usr/bin/env python3
"""Export raw xdof YAM MCAP episodes into the ABC training layout (abc's export_mcap.py format).

Produces, per episode:
    <out>/<split>/<episode_id>/
        states_actions.bin              # float64 rows: [state(14) | action(14)] per 30Hz tick
        combined_camera-images-rgb.mp4  # top/left/right vstacked, 224x224 each, strict GOP-30 CFR
        episode_metadata.json
plus <out>/norm_stats.json (mean/std over train-split rows).

Faithful to ~/abc/export_mcap.py: same 30Hz tick grid over the stream-overlap window,
same causal floor alignment, same bicubic scale+pad, same strict x264 params (timebase
1/15360, pts=512k, keyint 30, no B-frames) that abc_minimal's decode_frame requires.

Differences from abc's exporter (source data is the raw station format, not release MCAPs):
  - scalars come from left/right.mcap (/{side}-robot-state, /{side}-gripper-state) and
    action-left/right.mcap (/action-{side}-robot-state, /action-{side}-gripper-state)
  - camera frames come from standalone MP4s + *-timestamp.npy (not MCAP video packets)
  - top view is always the ZED-X left eye (the station's calibrated world eye) instead of
    abc's per-episode sha1 eye pick; the dataloader is eye-agnostic ("top" either way)

State/action layout matches abc: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
in raw MCAP joint order (no flipping — this is NOT the openpi yam convention).

Usage (typically via sky/convert_siemens_abc_layout.yaml):
    python scripts/yam_data/export_abc_layout_job.py --episode-csv job_episodes.csv \
        --out-dir ~/abc_dataset/industrial_packing_abc224 --min-duration-s 10 --workers 48
"""

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

S3_RAW_BUCKET = "s3://xdof-de-prod"
TICK_NS = 33333333  # int(1e9 / 30)
FPS, OUT_W, OUT_H = 30, 224, 224
TIMESCALE = 15360
TICKS_PER_FRAME = 512

# (mcap file, topic, dim) in states_actions.bin column order; states then actions.
SCALAR_STREAMS = [
    ("left.mcap", "/left-robot-state", 6),
    ("left.mcap", "/left-gripper-state", 1),
    ("right.mcap", "/right-robot-state", 6),
    ("right.mcap", "/right-gripper-state", 1),
    ("action-left.mcap", "/action-left-robot-state", 6),
    ("action-left.mcap", "/action-left-gripper-state", 1),
    ("action-right.mcap", "/action-right-robot-state", 6),
    ("action-right.mcap", "/action-right-gripper-state", 1),
]
# (camera key, raw video, raw timestamp npy) in combined.mp4 stack order (abc: top first).
# The top source differs per station: ZED stations (yam_zed_0_61) -> stereo left eye file,
# D405 stations (yam_0_61) -> mono top_camera-images-rgb.mp4. Exactly one exists per episode.
TOP_VIDEO_CANDIDATES = ["top_camera-images-left_rgb.mp4", "top_camera-images-rgb.mp4"]
CAMERA_STREAMS = [
    ("left", "left_camera-images-rgb.mp4", "left_camera-timestamp.npy"),
    ("right", "right_camera-images-rgb.mp4", "right_camera-timestamp.npy"),
]
STACK_ORDER = ["top", "left", "right"]
RAW_FILES = sorted(
    {f for f, _, _ in SCALAR_STREAMS}
    | {v for _, v, _ in CAMERA_STREAMS}
    | {t for _, _, t in CAMERA_STREAMS}
    | {"metadata.json", "top_camera-timestamp.npy"}
)

# ZED top -> D405-reference FOV crop (see convert_xdof_mcap_job.py for the rationale;
# derived per-episode from the episode's own intrinsics, centered on the principal point).
# Side cameras are NOT cropped (centered crops cut off the partner arm / hand-off region).
D405_REF_HFOV_DEG = 78.7
D405_REF_VFOV_DEG = 63.2


def top_crop_from_metadata(meta):
    import math

    cam = meta.get("camera_info", {}).get("top_camera", {})
    if not str(cam.get("camera_type", "")).upper().startswith("ZED"):
        return None
    intr = cam.get("intrinsics", {})
    stream = intr.get("left_rgb") or intr.get("rgb")
    if not stream:
        return None
    K = stream["intrinsics_matrix"]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    width, height = cam["width"], cam["height"]
    w = 2 * fx * math.tan(math.radians(D405_REF_HFOV_DEG / 2))
    h = 2 * fy * math.tan(math.radians(D405_REF_VFOV_DEG / 2))
    if w >= width or h >= height:
        return None
    # floor to even (matches the interactive viewer's window: 1194x896 on sz_48)
    w, h = int(w / 2) * 2, int(h / 2) * 2
    # Placement chosen by Karim 2026-08-26 in the interactive crop viewer (sz_48 frame:
    # x0=435, y0=304 => window center 86.7px right of the principal point, bottom margin 0).
    # The horizontal shift generalizes across stations as an angular offset:
    # 86.7px at fx=729 => tan(theta) = 86.7/729.
    x_center = cx + fx * (86.7 / 729.0)
    x0 = min(max(int(round(x_center - w / 2)), 0), width - w)
    # Bottom-anchored: never crop anything off the bottom of the frame (the robot
    # bases/rails live there).
    y0 = height - h
    return (x0, y0, w, h)

X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p"]
X264_STRICT_PARAMS = (
    f"keyint={FPS}:min-keyint={FPS}:scenecut=0:"
    f"fps={FPS}/1:timebase=1/{TIMESCALE}:force-cfr=1"
)
X264_STRICT_FFMPEG_ARGS = [
    "-vsync", "0",
    "-enc_time_base", f"1/{TIMESCALE}",
    "-video_track_timescale", str(TIMESCALE),
    "-bf", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-x264-params", X264_STRICT_PARAMS,
    "-threads", "1",
]


def floor_indices(source_ts, target_ts):
    """Index of the latest source message at or before each target tick (abc semantics)."""
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1, 0, len(source_ts) - 1)


def to_ns(ts):
    """Normalize a unix-epoch timestamp array to int64 ns.

    Global timestamp.npy is float seconds, ZED camera ts are int64 ns, D405 camera
    ts are float MILLISECONDS. Infer the unit from magnitude (valid 2001-2286).
    """
    m = float(np.nanmax(ts))
    scale = 1e9 if m < 1e11 else 1e6 if m < 1e14 else 1e3 if m < 1e17 else 1.0
    return (np.asarray(ts, dtype=np.float64) * scale).astype(np.int64)


def probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *entries, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return [int(x) for x in out.split(",")]


def encode_aligned(video_path, width, height, needed, out_path, crop=None, resize_mode="pad"):
    """Decode raw video, emit frame needed[i] at tick i (duplicating as required), re-encode.

    Same as abc's encode_aligned but the input is an mp4 container rather than a raw
    .h264 elementary stream (ffmpeg handles both identically). crop: optional
    (x0, y0, w, h) applied before the scale (ZED top FOV harmonization).
    resize_mode: "pad" letterboxes to OUT_WxOUT_H (openpi/abc convention, full FOV,
    black bars); "center_crop" takes the largest centered square then scales — full
    pixel budget, but narrows the horizontal FOV (wrists lose 37.5% of width; the
    FOV-matched top window loses ~25%).
    """
    frame_bytes = width * height * 3
    crop_vf = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else ""
    if resize_mode == "center_crop":
        vf = (f"{crop_vf}crop='min(iw\\,ih)':'min(iw\\,ih)',"
              f"scale={OUT_W}:{OUT_H}:flags=bicubic")
    else:
        vf = (f"{crop_vf}scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
              f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2")
    dec = subprocess.Popen(
        ["ffmpeg", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-vsync", "0", "-vf", vf, *X264, "-threads", "1", str(out_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    src_idx, frame = -1, None
    try:
        for wanted in needed:
            while src_idx < wanted:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                src_idx, frame = src_idx + 1, raw
            if frame is None:
                raise RuntimeError(f"decoder produced no frames (wanted index {wanted})")
            enc.stdin.write(frame)
    finally:
        dec.stdout.close(); dec.terminate(); dec.wait()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def read_scalar_streams(ep_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """topic -> (timestamps_ns, values) for all 8 scalar streams."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    out = {}
    for mcap_name in sorted({f for f, _, _ in SCALAR_STREAMS}):
        wanted = {t for f, t, _ in SCALAR_STREAMS if f == mcap_name}
        acc = {t: ([], []) for t in wanted}
        with open(ep_dir / mcap_name, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, proto in reader.iter_decoded_messages(topics=list(wanted)):
                pos = list(proto.position)
                if not pos:
                    continue
                acc[channel.topic][0].append(message.log_time)
                acc[channel.topic][1].append(pos)
        for topic, (ts, vals) in acc.items():
            if not ts:
                raise ValueError(f"{ep_dir.name}: no messages on {topic}")
            order = np.argsort(np.array(ts, dtype=np.int64), kind="stable")
            out[topic] = (np.array(ts, dtype=np.int64)[order], np.array(vals, dtype=np.float64)[order])
    return out


def export_episode(nfs_path: str, out_root: Path, split: str, raw_cache: Path, keep_raw: bool, resize_mode: str = "pad") -> dict | None:
    ep_name = Path(nfs_path).name
    ep_id = ep_name.removesuffix(".npy.mp4")
    raw_dir = raw_cache / ep_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / split / ep_id
    try:
        cmd = ["aws", "s3", "sync", S3_RAW_BUCKET + nfs_path, str(raw_dir), "--size-only",
               "--only-show-errors", "--exclude", "*"]
        for f in RAW_FILES + TOP_VIDEO_CANDIDATES:
            cmd += ["--include", f]
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        missing = [f for f in RAW_FILES if not (raw_dir / f).exists()]
        top_video = next((f for f in TOP_VIDEO_CANDIDATES if (raw_dir / f).exists()), None)
        if missing or top_video is None:
            print(f"[SKIP] {ep_id}: missing raw files {missing or [TOP_VIDEO_CANDIDATES]}")
            return None

        meta_raw = json.loads((raw_dir / "metadata.json").read_text())
        top_crop = top_crop_from_metadata(meta_raw)
        # (camera key, video file, ts file, crop) in STACK_ORDER: top first, like abc.
        cam_streams = [("top", top_video, "top_camera-timestamp.npy", top_crop)] + [
            (k, v, t, None) for k, v, t in CAMERA_STREAMS
        ]

        scalars = read_scalar_streams(raw_dir)
        cam_ts = {}
        for cam_key, _video, ts_npy, _crop in cam_streams:
            raw_ts = np.load(raw_dir / ts_npy)
            if len(raw_ts) == 0:
                print(f"[SKIP] {ep_id}: empty {ts_npy}")
                return None
            cam_ts[cam_key] = to_ns(raw_ts)

        # 30Hz tick grid over the overlap window of ALL streams (abc semantics).
        starts = [ts[0] for ts, _ in scalars.values()] + [ts[0] for ts in cam_ts.values()]
        ends = [ts[-1] for ts, _ in scalars.values()] + [ts[-1] for ts in cam_ts.values()]
        t0, t_end = max(starts), min(ends)
        ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
        num_steps = len(ticks)
        if num_steps < 10:
            print(f"[SKIP] {ep_id}: too short ({num_steps} steps)")
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for _mcap, topic, dim in SCALAR_STREAMS:
            ts, vals = scalars[topic]
            if vals.shape[1] != dim:
                raise ValueError(f"{ep_id}: {topic} dim {vals.shape[1]} != {dim}")
            parts.append(vals[floor_indices(ts, ticks)])
        sa = np.concatenate(parts, axis=-1)
        sa.tofile(out_dir / "states_actions.bin")

        with tempfile.TemporaryDirectory(dir=raw_cache) as work:
            mp4s = []
            for cam_key, video, _ts_npy, crop in cam_streams:
                ts = cam_ts[cam_key]
                video_path = raw_dir / video
                width, height = probe(video_path, "-show_entries", "stream=width,height")
                (n_frames,) = probe(video_path, "-count_frames", "-show_entries", "stream=nb_read_frames")
                if n_frames > 0 and n_frames != len(ts):  # container frames != timestamps; respace
                    ts = np.linspace(ts[0], ts[-1], n_frames, dtype=np.int64)
                mp4 = str(Path(work) / f"{cam_key}.mp4")
                encode_aligned(video_path, width, height, floor_indices(ts, ticks), mp4, crop=crop, resize_mode=resize_mode)
                mp4s.append(mp4)

            combined = str(out_dir / "combined_camera-images-rgb.mp4")
            filt = (
                "".join(f"[{i}:v]" for i in range(len(mp4s)))
                + f"vstack=inputs={len(mp4s)}[v0];"
                + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
            )
            subprocess.run(
                ["ffmpeg", "-y", *sum((["-i", p] for p in mp4s), []),
                 "-filter_complex", filt, "-map", "[out]",
                 *X264_STRICT_FFMPEG_ARGS, combined],
                capture_output=True, check=True,
            )
            for mp4 in mp4s + [combined]:
                (n,) = probe(mp4, "-count_frames", "-show_entries", "stream=nb_read_frames")
                if n != num_steps:
                    raise RuntimeError(f"{ep_id}: {Path(mp4).name} has {n} frames, expected {num_steps}")

        task_name = meta_raw.get("task_name", "industrial packing")
        meta = {"task_name": task_name, "cameras": [k for k, _, _, _ in cam_streams],
                "camera_resolutions": {k: [OUT_W, OUT_H] for k, _, _, _ in cam_streams},
                "alignment": "fixed_clock_30hz_causal", "t0_ns": int(t0), "tick_ns": TICK_NS,
                "num_steps": num_steps,
                "station_type": meta_raw.get("station_metadata", {}).get("station_type"),
                "top_fov_crop": list(top_crop) if top_crop else None,
                "resize_mode": resize_mode}
        (out_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[OK] {ep_id} ({split}): {num_steps} steps")
        return {"episode_id": ep_id, "split": split, "num_steps": num_steps}
    except Exception:
        print(f"[FAIL] {ep_id}:\n{traceback.format_exc()}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
    finally:
        if not keep_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)


def compute_norm_stats(train_root: Path, out_path: Path):
    """Mean/std of state and action dims over all train-split rows (abc norm_stats.json shape)."""
    n_total, s = 0, None
    for bin_path in sorted(train_root.glob("*/states_actions.bin")):
        rows = np.fromfile(bin_path, dtype=np.float64).reshape(-1, 28)
        if s is None:
            s = {"sum": np.zeros(28), "sq": np.zeros(28)}
        s["sum"] += rows.sum(axis=0)
        s["sq"] += (rows ** 2).sum(axis=0)
        n_total += len(rows)
    if not n_total:
        raise RuntimeError("no train episodes for norm stats")
    mean = s["sum"] / n_total
    std = np.sqrt(np.maximum(s["sq"] / n_total - mean ** 2, 0))
    stats = {
        "norm_stats": {
            "state": {"mean": mean[:14].tolist(), "std": std[:14].tolist()},
            "actions": {"mean": mean[14:].tolist(), "std": std[14:].tolist()},
        }
    }
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"norm_stats over {n_total} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-csv", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-duration-s", type=float, default=10.0)
    ap.add_argument("--val-episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--raw-cache-dir", type=Path, default=Path("/tmp/xdof_raw_abc"))
    ap.add_argument("--resize-mode", choices=["pad", "center_crop"], default="pad",
                    help="pad = letterbox (abc/openpi convention); center_crop = largest centered square, full pixel budget, narrower FOV")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    with open(args.episode_csv) as f:
        rows = [r for r in csv.DictReader(f)]
    n_total = len(rows)
    rows = [r for r in rows if float(r["duration_s"]) >= args.min_duration_s]
    rows.sort(key=lambda r: r["created_at"])  # oldest first, deterministic
    if args.max_episodes:
        rows = rows[: args.max_episodes]
    print(f"{n_total} episodes in job; {len(rows)} pass the >={args.min_duration_s}s filter")

    out_root = args.out_dir.expanduser()
    if out_root.exists():
        raise FileExistsError(f"{out_root} already exists; remove it first")
    args.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # newest N episodes held out for val
    splits = ["train"] * (len(rows) - args.val_episodes) + ["val"] * args.val_episodes
    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(export_episode, r["nfs_path"], out_root, split, args.raw_cache_dir, args.keep_raw, args.resize_mode)
            for r, split in zip(rows, splits)
        ]
        for i, fut in enumerate(as_completed(futs)):
            if fut.result() is not None:
                ok += 1
            if (i + 1) % 25 == 0:
                print(f"progress: {i + 1}/{len(futs)} processed, {ok} ok", flush=True)

    print(f"exported {ok}/{len(rows)} episodes")
    compute_norm_stats(out_root / "train", out_root / "norm_stats.json")
    n_train = len(list((out_root / "train").iterdir())) if (out_root / "train").exists() else 0
    n_val = len(list((out_root / "val").iterdir())) if (out_root / "val").exists() else 0
    print(f"done: train={n_train} val={n_val} -> {out_root}")


if __name__ == "__main__":
    main()
