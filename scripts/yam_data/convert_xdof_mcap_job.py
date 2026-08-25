#!/usr/bin/env python3
"""Convert raw xdof YAM MCAP episodes (from a DataEngine collection job) to a LeRobot v2.1 dataset.

Unlike convert_yam_data.py (which expects pre-extracted *-joint_pos.npy files), this reads the
modern xdof station format directly:

    episode_<ts>_<id>.npy.mp4/
        left.mcap / right.mcap            # /​{side}-robot-state (6D pos @ ~290Hz), /{side}-gripper-state (1D)
        timestamp.npy                     # 30Hz global clock (seconds)
        {left,right}_camera-images-rgb.mp4
        top_camera-images-left_rgb.mp4    # ZED-X stereo left eye = canonical top view
        {left,right,top}_camera-timestamp.npy
        metadata.json                     # task_name, station_metadata, ...

Episodes are streamed from S3 (s3://xdof-de-prod + nfs_path), converted, and raw files deleted
immediately, so peak disk usage stays ~num_workers x 150MB.

Conventions match convert_yam_data.py / LeRobotYamDataConfig exactly:
    state[0:6]  = flip(left-joint_pos)   state[6]  = left-gripper_pos
    state[7:13] = flip(right-joint_pos)  state[13] = right-gripper_pos
    actions = state (absolute joint positions), fps 30, 224x224 resize-with-pad videos.

Camera frames are aligned to the global 30Hz clock via nearest-neighbor on the per-camera
timestamps (instead of take-first-N).

Output is LeRobot v2.1; run lerobot's convert_dataset_v21_to_v30 afterwards for the v3.0
layout that lerobot>=0.4 loads.

Usage:
    uv run scripts/yam_data/convert_xdof_mcap_job.py \
        --episode-csv job_episodes.csv --min-duration-s 10 \
        --output-dir ~/.cache/huggingface/lerobot --repo-name industrial_packing_yam
"""

import dataclasses
import json
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
import tyro
from PIL import Image

S3_RAW_BUCKET = "s3://xdof-de-prod"
CAMERA_KEYS = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
# top view comes from the ZED-X stereo left eye
RAW_VIDEO_FOR_KEY = {
    "left_camera-images-rgb": ("left_camera-images-rgb.mp4", "left_camera-timestamp.npy"),
    "right_camera-images-rgb": ("right_camera-images-rgb.mp4", "right_camera-timestamp.npy"),
    "top_camera-images-rgb": ("top_camera-images-left_rgb.mp4", "top_camera-timestamp.npy"),
}
RAW_FILES = ["left.mcap", "right.mcap", "timestamp.npy", "metadata.json"] + [
    f for pair in RAW_VIDEO_FOR_KEY.values() for f in pair
]


@dataclasses.dataclass
class Config:
    episode_csv: str
    output_dir: Path = Path("~/.cache/huggingface/lerobot").expanduser()
    repo_name: str = "industrial_packing_yam"
    min_duration_s: float = 10.0
    raw_cache_dir: Path = Path("/tmp/xdof_raw_eps")
    resize_size: int = 224
    fps: int = 30
    chunk_size: int = 1000
    max_workers: int = 24
    keep_raw: bool = False
    max_episodes: int | None = None  # for smoke tests


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """BILINEAR resize preserving aspect ratio, zero-pad to size x size (centered)."""
    im = Image.fromarray(img)
    ratio = max(im.width / size, im.height / size)
    rw, rh = int(im.width / ratio), int(im.height / ratio)
    im = im.resize((rw, rh), resample=Image.BILINEAR)
    out = Image.new("RGB", (size, size), 0)
    out.paste(im, ((size - rw) // 2, (size - rh) // 2))
    return np.asarray(out)


def nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest source timestamp for each target timestamp."""
    idx = np.searchsorted(source_ts, target_ts)
    idx = np.clip(idx, 1, len(source_ts) - 1)
    left = source_ts[idx - 1]
    right = source_ts[idx]
    idx -= target_ts - left < right - target_ts
    return np.clip(idx, 0, len(source_ts) - 1)


def read_side_mcap(path: Path, side: str) -> dict[str, np.ndarray]:
    """Extract joint/gripper positions and their timestamps (ns) from a {side}.mcap file."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    out = {}
    joint_ts, joint_pos, grip_ts, grip_pos = [], [], [], []
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, proto in reader.iter_decoded_messages(
            topics=[f"/{side}-robot-state", f"/{side}-gripper-state"]
        ):
            pos = list(proto.position)
            if not pos:
                continue
            if channel.topic == f"/{side}-robot-state":
                joint_ts.append(message.log_time)
                joint_pos.append(pos)
            else:
                grip_ts.append(message.log_time)
                grip_pos.append(pos)
    out["joint_ts"] = np.array(joint_ts, dtype=np.int64)
    out["joint_pos"] = np.array(joint_pos, dtype=np.float32)
    out["grip_ts"] = np.array(grip_ts, dtype=np.int64)
    out["grip_pos"] = np.array(grip_pos, dtype=np.float32)
    if out["joint_pos"].shape[1:] != (6,) or out["grip_pos"].shape[1:] != (1,):
        raise ValueError(f"{path}: unexpected shapes joint={out['joint_pos'].shape} grip={out['grip_pos'].shape}")
    return out


def build_state(ep_dir: Path) -> np.ndarray:
    """(N,14) float32 state aligned to timestamp.npy, yam converter joint order (flipped)."""
    ts_global_ns = (np.load(ep_dir / "timestamp.npy") * 1e9).astype(np.int64)
    n = len(ts_global_ns)
    state = np.empty((n, 14), dtype=np.float32)
    for side, joint_slice, grip_col in [("left", slice(0, 6), 6), ("right", slice(7, 13), 13)]:
        d = read_side_mcap(ep_dir / f"{side}.mcap", side)
        ji = nearest_indices(d["joint_ts"], ts_global_ns)
        gi = nearest_indices(d["grip_ts"], ts_global_ns)
        state[:, joint_slice] = np.flip(d["joint_pos"][ji], axis=1)
        state[:, grip_col] = d["grip_pos"][gi][:, 0]
    return state


def transcode_camera(
    raw_video: Path, raw_ts: Path, ts_global: np.ndarray, out_path: Path, size: int, fps: int
) -> int:
    """Decode raw video, pick nearest frame per global timestamp, resize-with-pad, encode h264."""
    cam_ts = np.load(raw_ts)
    needed = nearest_indices(cam_ts, ts_global)  # non-decreasing frame indices, may repeat
    n_out = len(needed)

    out = av.open(str(out_path), "w", format="mp4")
    vs = out.add_stream("h264", rate=fps)
    vs.width = vs.height = size
    vs.pix_fmt = "yuv420p"
    vs.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}

    written = 0
    pos = 0  # pointer into `needed`
    last_small = None
    with av.open(str(raw_video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for f_idx, frame in enumerate(container.decode(stream)):
            if pos >= n_out:
                break
            if needed[pos] > f_idx:
                continue
            small = resize_with_pad(frame.to_ndarray(format="rgb24"), size)
            last_small = small
            while pos < n_out and needed[pos] == f_idx:
                nf = av.VideoFrame.from_ndarray(small, format="rgb24")
                for pk in vs.encode(nf):
                    out.mux(pk)
                written += 1
                pos += 1
    # camera stream ended early (fewer decodable frames than timestamps): repeat last frame
    while pos < n_out and last_small is not None:
        nf = av.VideoFrame.from_ndarray(last_small, format="rgb24")
        for pk in vs.encode(nf):
            out.mux(pk)
        written += 1
        pos += 1
    for pk in vs.encode(None):
        out.mux(pk)
    out.close()
    return written


def process_episode(ep_idx: int, nfs_path: str, cfg: Config, base_dir: Path) -> dict | None:
    ep_name = Path(nfs_path).name
    raw_dir = cfg.raw_cache_dir / ep_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 1. download only the files we need
        s3_src = S3_RAW_BUCKET + nfs_path
        cmd = ["aws", "s3", "cp", s3_src, str(raw_dir), "--recursive", "--only-show-errors", "--exclude", "*"]
        for f in RAW_FILES:
            cmd += ["--include", f]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        missing = [f for f in RAW_FILES if not (raw_dir / f).exists()]
        if missing:
            print(f"  ep {ep_idx} ({ep_name}): missing raw files {missing}; skipping")
            return None

        # 2. state/actions aligned to the global clock; drop last frame (yam converter convention)
        state = build_state(raw_dir)
        n_use = len(state) - 1
        if n_use < cfg.fps:  # <1s of frames: junk
            print(f"  ep {ep_idx} ({ep_name}): only {n_use} frames; skipping")
            return None
        state = state[:n_use]
        ts_global_ns = (np.load(raw_dir / "timestamp.npy") * 1e9).astype(np.int64)[:n_use]

        task_name = json.loads((raw_dir / "metadata.json").read_text()).get("task_name", "industrial packing")

        # 3. videos
        chunk_id = ep_idx // cfg.chunk_size
        for cam_key, (raw_video, raw_ts) in RAW_VIDEO_FOR_KEY.items():
            video_dir = base_dir / "videos" / f"chunk-{chunk_id:03d}" / cam_key
            video_dir.mkdir(parents=True, exist_ok=True)
            written = transcode_camera(
                raw_dir / raw_video, raw_dir / raw_ts, ts_global_ns,
                video_dir / f"episode_{ep_idx:06d}.mp4", cfg.resize_size, cfg.fps,
            )
            if written != n_use:
                raise ValueError(f"{cam_key}: wrote {written} frames, expected {n_use}")

        # 4. parquet (index column is fixed to global offsets in the finalize step)
        df = pd.DataFrame(
            {
                "state": [row for row in state],
                "actions": [row for row in state.copy()],
                "timestamp": (np.arange(n_use) / cfg.fps).astype(np.float32),
                "frame_index": np.arange(n_use, dtype=np.int64),
                "episode_index": np.full(n_use, ep_idx, dtype=np.int64),
                "index": np.arange(n_use, dtype=np.int64),
                "task_index": np.zeros(n_use, dtype=np.int64),
            }
        )
        data_dir = base_dir / "data" / f"chunk-{chunk_id:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet")

        # 5. per-episode stats (legacy v2.1 serialization; image stats use the [0,1] defaults
        # like convert_yam_data.py — openpi norm stats are computed separately)
        stats = {}
        for feat, arr in [("state", state), ("actions", state)]:
            stats[feat] = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [n_use],
            }
        for feat in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            col = df[feat].to_numpy().astype(np.float64)
            stats[feat] = {
                "min": [float(col.min())],
                "max": [float(col.max())],
                "mean": [float(col.mean())],
                "std": [float(col.std())],
                "count": [n_use],
            }
        for cam_key in CAMERA_KEYS:
            stats[cam_key] = {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.25]], [[0.25]], [[0.25]]],
                "count": [n_use],
            }

        return {"episode_index": ep_idx, "tasks": [task_name], "length": n_use, "stats": stats}
    except Exception:
        print(f"  ep {ep_idx} ({ep_name}) FAILED:\n{traceback.format_exc()}")
        return None
    finally:
        if not cfg.keep_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)


def main(cfg: Config):
    df = pd.read_csv(cfg.episode_csv)
    n_total = len(df)
    df = df[df["duration_s"].astype(float) >= cfg.min_duration_s].copy()
    # oldest-first for deterministic episode numbering
    df = df.sort_values("created_at").reset_index(drop=True)
    if cfg.max_episodes:
        df = df.head(cfg.max_episodes)
    print(f"{n_total} episodes in job; {len(df)} pass the >={cfg.min_duration_s}s filter")

    base_dir = cfg.output_dir / cfg.repo_name
    if base_dir.exists():
        raise FileExistsError(f"{base_dir} already exists; remove it first")
    (base_dir / "meta").mkdir(parents=True)

    results = {}
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(process_episode, idx, row.nfs_path, cfg, base_dir): idx
            for idx, row in enumerate(df.itertuples())
        }
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is not None:
                results[res["episode_index"]] = res
            if (i + 1) % 25 == 0:
                print(f"progress: {i + 1}/{len(futures)} processed, {len(results)} ok")

    ok_indices = sorted(results)
    print(f"converted {len(ok_indices)}/{len(df)} episodes; renumbering...")

    # Renumber to a contiguous 0..K-1 (failures leave holes) and fix global `index` offsets.
    tasks: dict[str, int] = {}
    episodes, episodes_stats = [], []
    global_offset = 0
    for new_idx, old_idx in enumerate(ok_indices):
        res = results[old_idx]
        task = res["tasks"][0]
        task_index = tasks.setdefault(task, len(tasks))
        old_chunk, new_chunk = old_idx // cfg.chunk_size, new_idx // cfg.chunk_size
        old_pq = base_dir / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet"
        new_pq = base_dir / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_idx:06d}.parquet"
        pq = pd.read_parquet(old_pq)
        pq["episode_index"] = np.int64(new_idx)
        pq["index"] = np.arange(global_offset, global_offset + len(pq), dtype=np.int64)
        pq["task_index"] = np.int64(task_index)
        new_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.to_parquet(new_pq)
        if old_pq != new_pq:
            old_pq.unlink()
        for cam_key in CAMERA_KEYS:
            old_mp4 = base_dir / "videos" / f"chunk-{old_chunk:03d}" / cam_key / f"episode_{old_idx:06d}.mp4"
            new_mp4 = base_dir / "videos" / f"chunk-{new_chunk:03d}" / cam_key / f"episode_{new_idx:06d}.mp4"
            if old_mp4 != new_mp4:
                new_mp4.parent.mkdir(parents=True, exist_ok=True)
                old_mp4.rename(new_mp4)

        stats = res["stats"]
        n = res["length"]
        stats["episode_index"] = {"min": [float(new_idx)], "max": [float(new_idx)], "mean": [float(new_idx)], "std": [0.0], "count": [n]}
        stats["index"] = {
            "min": [float(global_offset)],
            "max": [float(global_offset + n - 1)],
            "mean": [float(global_offset) + (n - 1) / 2.0],
            "std": [float(np.arange(n).std())],
            "count": [n],
        }
        stats["task_index"] = {"min": [float(task_index)], "max": [float(task_index)], "mean": [float(task_index)], "std": [0.0], "count": [n]}
        episodes.append({"episode_index": new_idx, "tasks": [task], "length": n})
        episodes_stats.append({"episode_index": new_idx, "stats": stats})
        global_offset += n

    with open(base_dir / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(base_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")
    with open(base_dir / "meta" / "tasks.jsonl", "w") as f:
        for task, task_index in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    total_frames = global_offset
    n_eps = len(episodes)
    features = {
        "state": {"dtype": "float32", "shape": [14], "names": ["state"]},
        "actions": {"dtype": "float32", "shape": [14], "names": ["actions"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam_key in CAMERA_KEYS:
        features[cam_key] = {
            "dtype": "video",
            "shape": [cfg.resize_size, cfg.resize_size, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": cfg.fps,
                "video.height": cfg.resize_size,
                "video.width": cfg.resize_size,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yams",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(CAMERA_KEYS) * n_eps,
        "total_chunks": (n_eps + cfg.chunk_size - 1) // cfg.chunk_size,
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(base_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"done: {n_eps} episodes, {total_frames} frames, tasks={list(tasks)} -> {base_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
