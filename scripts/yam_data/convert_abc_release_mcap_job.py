#!/usr/bin/env python3
"""Convert official ABC-130k release MCAP episodes to a LeRobot dataset.

This is the release-MCAP counterpart of ``convert_xdof_mcap_job.py``.  The two
sources contain the same YAM policy signals but package them differently:

* DataEngine jobs have separate state/action MCAPs, camera MP4s, and
  ``timestamp.npy``.
* ABC releases bundle protobuf position streams and H264 camera packets into one
  ``episode.mcap``.

The output intentionally follows the established OpenPI/YAM LeRobot contract:

* 30 Hz, absolute 14-D state and recorded 14-D commanded actions;
* state/action layout ``left arm(6), left gripper(1), right arm(6), right
  gripper(1)``;
* per-arm joint reversal by default (disable with ``--no-flip-joints``);
* wrist views padded to a square and the top view center-cropped by default;
* standard LeRobot camera keys and deterministic source provenance.

The converter first writes the well-tested LeRobot v2.1 episode layout used by
``convert_xdof_mcap_job.py`` and, by default, migrates it locally to v3.0 with
LeRobot's official migrator.  It never modifies or deletes source MCAPs.

Example:

    uv run scripts/yam_data/convert_abc_release_mcap_job.py \
      --input-root /path/to/hf_tasks/load_the_plates_into_the_dish_rack/train \
      --output-dir ~/.cache/huggingface/lerobot \
      --repo-name abc130k_real_load_plates_lerobot_v1 \
      --task-override "Load the plates into the dish rack" \
      --max-workers 24
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import traceback
from typing import Literal

import av
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import tyro

CAMERA_KEYS = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
STATE_TOPICS = [
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
]
ACTION_TOPICS = [
    ("/left-arm-action", 6),
    ("/left-ee-action", 1),
    ("/right-arm-action", 6),
    ("/right-ee-action", 1),
]
WRIST_TOPICS = {
    "left_camera-images-rgb": "/left-wrist-camera",
    "right_camera-images-rgb": "/right-wrist-camera",
}
TOP_TOPIC_CANDIDATES = ("/top-left-camera", "/top-right-camera", "/top-camera")
JOINT_FLIP_ORDER = np.asarray([5, 4, 3, 2, 1, 0, 6, 12, 11, 10, 9, 8, 7, 13], dtype=np.int64)
EPISODE_START_VIDEO_NAME = "episode_start_frames.mp4"
EPISODE_START_INDEX_NAME = "episode_start_frames.json"
EPISODE_START_COMPACT_VIDEO_NAME = "episode_start_frames_compact.mp4"
EPISODE_START_COMPACT_INDEX_NAME = "episode_start_frames_compact.json"
EPISODE_START_ARTIFACT_NAMES = (
    EPISODE_START_VIDEO_NAME,
    EPISODE_START_INDEX_NAME,
    EPISODE_START_COMPACT_VIDEO_NAME,
    EPISODE_START_COMPACT_INDEX_NAME,
)
EPISODE_START_CAMERA_ORDER = [
    "top_camera-images-rgb",
    "left_camera-images-rgb",
    "right_camera-images-rgb",
]
EPISODE_START_HEADER_HEIGHT = 40


@dataclasses.dataclass(frozen=True)
class Config:
    input_root: Path
    output_dir: Path = Path("~/.cache/huggingface/lerobot")
    repo_name: str = "abc130k_real_load_plates_lerobot_v1"
    task_override: str | None = None
    resize_size: int = 224
    resize_mode: Literal["pad", "center_crop", "top_center_crop"] = "top_center_crop"
    fps: int = 30
    chunk_size: int = 1000
    max_workers: int = 24
    flip_joints: bool = True
    max_episodes: int | None = None
    allow_failed_episodes: bool = False
    convert_to_v30: bool = True
    # A single large parquet avoids a known lerobot 0.4.5 v2.1->v3.0 migration
    # boundary bug that can double-count frames when a data file rolls over.
    v30_data_file_size_mb: int = 100_000
    v30_video_file_size_mb: int = 1_000
    # The converter always writes one three-camera first-frame overview frame
    # per episode. At 2 fps, 1,335 episodes take about 11 minutes to review.
    episode_start_video_fps: int = 2
    # Also emit a deterministic 30-second random sample by default.
    episode_start_compact_count: int = 60
    episode_start_sample_seed: int = 0

    @property
    def camera_resize_modes(self) -> dict[str, str]:
        if self.resize_mode == "top_center_crop":
            return {key: "center_crop" if key == "top_camera-images-rgb" else "pad" for key in CAMERA_KEYS}
        return dict.fromkeys(CAMERA_KEYS, self.resize_mode)


def nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Return the nearest source index for each target timestamp."""
    source_ts = np.asarray(source_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    if source_ts.size == 0:
        raise ValueError("cannot align an empty timestamp stream")
    if source_ts.size == 1:
        return np.zeros(target_ts.shape, dtype=np.int64)
    idx = np.searchsorted(source_ts, target_ts)
    idx = np.clip(idx, 1, len(source_ts) - 1)
    left = source_ts[idx - 1]
    right = source_ts[idx]
    idx -= target_ts - left < right - target_ts
    return np.clip(idx, 0, len(source_ts) - 1)


def center_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    return np.asarray(
        Image.fromarray(image[y0 : y0 + side, x0 : x0 + side]).resize((size, size), resample=Image.BILINEAR)
    )


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    source = Image.fromarray(image)
    ratio = max(source.width / size, source.height / size)
    resized_width, resized_height = int(source.width / ratio), int(source.height / ratio)
    source = source.resize((resized_width, resized_height), resample=Image.BILINEAR)
    output = Image.new("RGB", (size, size), 0)
    output.paste(source, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(output)


def _episode_id(path: Path, metadata: list[dict[str, str]]) -> str:
    for item in metadata:
        session_uuid = item.get("session-uuid")
        if session_uuid:
            return f"episode_{session_uuid}"
    return path.parent.name


def select_top_topic(episode_id: str, cameras: dict[str, list[tuple[int, bytes]]]) -> str:
    if "/top-left-camera" in cameras and "/top-right-camera" in cameras:
        # Match ABC's production stereo_top_policy="random" deterministically.
        return "/top-left-camera" if hashlib.sha1(episode_id.encode()).digest()[0] % 2 == 0 else "/top-right-camera"
    if "/top-camera" in cameras:
        return "/top-camera"
    raise ValueError("missing top camera stream")


def read_release_mcap(
    path: Path,
) -> tuple[
    str,
    dict[str, list[tuple[int, np.ndarray]]],
    dict[str, list[tuple[int, bytes]]],
    list[dict[str, str]],
]:
    """Read the policy streams from one official ABC release MCAP."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    scalar_topics = {topic for topic, _ in STATE_TOPICS + ACTION_TOPICS}
    camera_topics = set(WRIST_TOPICS.values()) | set(TOP_TOPIC_CANDIDATES)
    scalars: dict[str, list[tuple[int, np.ndarray]]] = {}
    cameras: dict[str, list[tuple[int, bytes]]] = {}
    metadata: list[dict[str, str]] = []

    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        metadata = [dict(item.metadata) for item in reader.iter_metadata()]
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=sorted(scalar_topics | camera_topics)
        ):
            topic = channel.topic
            if topic in scalar_topics:
                position = np.asarray(decoded.position, dtype=np.float32)
                if position.size:
                    scalars.setdefault(topic, []).append((int(message.log_time), position))
            elif topic in camera_topics:
                data = bytes(decoded.data)
                if data:
                    cameras.setdefault(topic, []).append((int(message.log_time), data))

    for messages in (*scalars.values(), *cameras.values()):
        messages.sort(key=lambda item: item[0])
    return _episode_id(path, metadata), scalars, cameras, metadata


def validate_required_streams(
    path: Path,
    scalars: dict[str, list[tuple[int, np.ndarray]]],
    cameras: dict[str, list[tuple[int, bytes]]],
    top_topic: str,
) -> None:
    missing_scalars = [topic for topic, _ in STATE_TOPICS + ACTION_TOPICS if not scalars.get(topic)]
    camera_plan = {**WRIST_TOPICS, "top_camera-images-rgb": top_topic}
    missing_cameras = [topic for topic in camera_plan.values() if not cameras.get(topic)]
    if missing_scalars or missing_cameras:
        raise ValueError(f"{path}: missing required streams: scalars={missing_scalars}, cameras={missing_cameras}")
    for topic, expected_dim in STATE_TOPICS + ACTION_TOPICS:
        dimensions = {value.shape for _, value in scalars[topic]}
        if dimensions != {(expected_dim,)}:
            raise ValueError(f"{path}: {topic} shapes={sorted(dimensions)}, expected {(expected_dim,)}")


def common_ticks(
    scalars: dict[str, list[tuple[int, np.ndarray]]],
    cameras: dict[str, list[tuple[int, bytes]]],
    top_topic: str,
    fps: int,
) -> np.ndarray:
    streams = [scalars[topic] for topic, _ in STATE_TOPICS + ACTION_TOPICS]
    streams += [cameras[WRIST_TOPICS[key]] for key in WRIST_TOPICS]
    streams.append(cameras[top_topic])
    start_ns = max(messages[0][0] for messages in streams)
    end_ns = min(messages[-1][0] for messages in streams)
    tick_ns = int(1_000_000_000 / fps)
    ticks = np.arange(start_ns + tick_ns, end_ns + 1, tick_ns, dtype=np.int64)
    if len(ticks) < fps:
        raise ValueError(f"stream overlap is only {len(ticks)} frames (<1 second at {fps} Hz)")
    return ticks


def align_state_actions(
    scalars: dict[str, list[tuple[int, np.ndarray]]], ticks: np.ndarray, *, flip_joints: bool
) -> tuple[np.ndarray, np.ndarray]:
    aligned_groups = []
    for topics in (STATE_TOPICS, ACTION_TOPICS):
        parts = []
        for topic, _dim in topics:
            messages = scalars[topic]
            timestamps = np.asarray([timestamp for timestamp, _ in messages], dtype=np.int64)
            values = np.stack([value for _, value in messages]).astype(np.float32, copy=False)
            parts.append(values[nearest_indices(timestamps, ticks)])
        aligned_groups.append(np.concatenate(parts, axis=1).astype(np.float32, copy=False))
    state, actions = aligned_groups
    if flip_joints:
        state = state[:, JOINT_FLIP_ORDER].copy()
        actions = actions[:, JOINT_FLIP_ORDER].copy()
    return state, actions


def _decoded_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return sum(1 for _ in container.decode(stream))


def transcode_camera_packets(
    messages: list[tuple[int, bytes]],
    ticks: np.ndarray,
    output_path: Path,
    size: int,
    fps: int,
    resize_mode: str,
) -> int:
    """Align H264 packets from a release MCAP and encode a LeRobot MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".h264") as encoded:
        for _, payload in messages:
            encoded.write(payload)
        encoded.flush()
        raw_path = Path(encoded.name)
        frame_count = _decoded_frame_count(raw_path)
        if frame_count <= 0:
            raise ValueError("camera stream has no decodable frames")
        camera_timestamps = np.asarray([timestamp for timestamp, _ in messages], dtype=np.int64)
        if frame_count != len(camera_timestamps):
            # Some release messages are H264 chunks rather than one complete frame.
            camera_timestamps = np.linspace(camera_timestamps[0], camera_timestamps[-1], frame_count, dtype=np.int64)
        needed = nearest_indices(camera_timestamps, ticks)

        output = av.open(str(output_path), "w", format="mp4")
        video_stream = output.add_stream("h264", rate=fps)
        video_stream.width = video_stream.height = size
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
        position = 0
        written = 0
        last_frame = None
        with av.open(str(raw_path)) as source:
            source_stream = source.streams.video[0]
            source_stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(source.decode(source_stream)):
                if position >= len(needed):
                    break
                if needed[position] > frame_index:
                    continue
                image = frame.to_ndarray(format="rgb24")
                small = (
                    center_crop_resize(image, size) if resize_mode == "center_crop" else resize_with_pad(image, size)
                )
                last_frame = small
                while position < len(needed) and needed[position] == frame_index:
                    for packet in video_stream.encode(av.VideoFrame.from_ndarray(small, format="rgb24")):
                        output.mux(packet)
                    written += 1
                    position += 1
        while position < len(needed) and last_frame is not None:
            for packet in video_stream.encode(av.VideoFrame.from_ndarray(last_frame, format="rgb24")):
                output.mux(packet)
            written += 1
            position += 1
        for packet in video_stream.encode(None):
            output.mux(packet)
        output.close()
    return written


def _episode_stats(state: np.ndarray, actions: np.ndarray, frame: pd.DataFrame) -> dict:
    stats = {}
    for feature, values in (("state", state), ("actions", actions)):
        stats[feature] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [len(values)],
        }
    for feature in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        values = frame[feature].to_numpy(dtype=np.float64)
        stats[feature] = {
            "min": [float(values.min())],
            "max": [float(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [len(values)],
        }
    for camera_key in CAMERA_KEYS:
        stats[camera_key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [len(frame)],
        }
    return stats


def _first_video_frame(path: Path) -> np.ndarray:
    """Decode the first RGB frame from a converted per-episode video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise ValueError(f"video has no decodable frames: {path}")


def _episode_start_canvas(
    camera_frames: dict[str, np.ndarray],
    *,
    episode_index: int,
    video_frame_index: int,
    episode_count: int,
    source_episode_id: str,
    size: int,
) -> np.ndarray:
    """Lay out the exact converted first frames with a compact episode label."""
    width = size * len(EPISODE_START_CAMERA_ORDER)
    canvas = Image.new("RGB", (width, size + EPISODE_START_HEADER_HEIGHT), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 3),
        f"episode {episode_index:04d}  overview {video_frame_index + 1:04d}/{episode_count:04d}  {source_episode_id}",
        fill=(255, 255, 255),
    )
    panel_labels = ("TOP", "LEFT WRIST", "RIGHT WRIST")
    for panel_index, (camera_key, panel_label) in enumerate(zip(EPISODE_START_CAMERA_ORDER, panel_labels, strict=True)):
        image = Image.fromarray(camera_frames[camera_key]).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), resample=Image.BILINEAR)
        x = panel_index * size
        canvas.paste(image, (x, EPISODE_START_HEADER_HEIGHT))
        draw.text((x + 6, 21), panel_label, fill=(190, 220, 255))
    return np.asarray(canvas)


def write_episode_start_video(
    root: Path,
    manifest_rows: list[dict],
    *,
    resize_size: int,
    video_fps: int,
    frame_loader: Callable[[int], dict[str, np.ndarray]],
    video_name: str = EPISODE_START_VIDEO_NAME,
    index_name: str = EPISODE_START_INDEX_NAME,
    additional_metadata: dict | None = None,
) -> dict:
    """Write one labeled three-camera first-frame overview frame per episode."""
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    output_path = meta / video_name
    temp_path = meta / f".{video_name}.tmp.mp4"
    index_path = meta / index_name
    frame_index = []

    output = av.open(str(temp_path), "w", format="mp4")
    stream = output.add_stream("h264", rate=video_fps)
    stream.width = resize_size * len(EPISODE_START_CAMERA_ORDER)
    stream.height = resize_size + EPISODE_START_HEADER_HEIGHT
    stream.pix_fmt = "yuv420p"
    # Every video frame represents a different episode. All-intra encoding
    # makes frame-accurate seeking cheap in ordinary video players.
    stream.options = {"crf": "20", "preset": "veryfast", "g": "1", "movflags": "+faststart"}
    try:
        episode_count = len(manifest_rows)
        for video_frame_index, row in enumerate(manifest_rows):
            episode_index = int(row["episode_index"])
            camera_frames = frame_loader(episode_index)
            canvas = _episode_start_canvas(
                camera_frames,
                episode_index=episode_index,
                video_frame_index=video_frame_index,
                episode_count=episode_count,
                source_episode_id=str(row["source_episode_id"]),
                size=resize_size,
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                output.mux(packet)
            frame_index.append(
                {
                    "video_frame_index": video_frame_index,
                    "episode_index": episode_index,
                    "source_episode_id": str(row["source_episode_id"]),
                    "source_split": str(row["source_split"]),
                    "source_task": str(row["source_task"]),
                    "top_topic": str(row["top_topic"]),
                }
            )
        for packet in stream.encode(None):
            output.mux(packet)
    except Exception:
        output.close()
        temp_path.unlink(missing_ok=True)
        raise
    output.close()
    temp_path.replace(output_path)

    metadata = {
        "schema_version": 1,
        "video_path": f"meta/{video_name}",
        "index_path": f"meta/{index_name}",
        "frame_rate": video_fps,
        "seconds_per_episode": 1 / video_fps,
        "frame_count": len(frame_index),
        "camera_panel_order": EPISODE_START_CAMERA_ORDER,
        "frame_layout": {
            "panel_width": resize_size,
            "panel_height": resize_size,
            "header_height": EPISODE_START_HEADER_HEIGHT,
        },
        "frames": frame_index,
    }
    if additional_metadata is not None:
        metadata.update(additional_metadata)
    index_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {key: value for key, value in metadata.items() if key != "frames"}


def sample_episode_start_rows(manifest_rows: list[dict], *, count: int, seed: int) -> list[dict]:
    """Sample episode rows without replacement in deterministic random order."""
    if count <= 0:
        raise ValueError("episode_start_compact_count must be positive")
    sample_count = min(count, len(manifest_rows))
    indices = np.random.default_rng(seed).choice(len(manifest_rows), size=sample_count, replace=False)
    return [manifest_rows[int(index)] for index in indices]


def write_episode_start_video_v21(
    root: Path,
    manifest_rows: list[dict],
    config: Config,
) -> dict:
    """Write the overview while videos are still episode-separated in v2.1."""

    def load_episode_frames(episode_index: int) -> dict[str, np.ndarray]:
        chunk = episode_index // config.chunk_size
        return {
            camera_key: _first_video_frame(
                root / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"
            )
            for camera_key in EPISODE_START_CAMERA_ORDER
        }

    return write_episode_start_video(
        root,
        manifest_rows,
        resize_size=config.resize_size,
        video_fps=config.episode_start_video_fps,
        frame_loader=load_episode_frames,
    )


def write_compact_episode_start_video_v21(
    root: Path,
    manifest_rows: list[dict],
    config: Config,
) -> dict:
    """Write the deterministic random compact overview from v2.1 videos."""
    sampled_rows = sample_episode_start_rows(
        manifest_rows,
        count=config.episode_start_compact_count,
        seed=config.episode_start_sample_seed,
    )

    def load_episode_frames(episode_index: int) -> dict[str, np.ndarray]:
        chunk = episode_index // config.chunk_size
        return {
            camera_key: _first_video_frame(
                root / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"
            )
            for camera_key in EPISODE_START_CAMERA_ORDER
        }

    return write_episode_start_video(
        root,
        sampled_rows,
        resize_size=config.resize_size,
        video_fps=config.episode_start_video_fps,
        frame_loader=load_episode_frames,
        video_name=EPISODE_START_COMPACT_VIDEO_NAME,
        index_name=EPISODE_START_COMPACT_INDEX_NAME,
        additional_metadata={
            "sampling": "random_without_replacement",
            "sample_seed": config.episode_start_sample_seed,
            "requested_sample_count": config.episode_start_compact_count,
            "source_episode_count": len(manifest_rows),
        },
    )


def _source_split(path: Path) -> str:
    for part in reversed(path.parts):
        if part in {"train", "val", "test"}:
            return part
    return "unknown"


def _source_task(path: Path) -> str:
    if path.parent.name.startswith("episode_"):
        return path.parent.parent.name.replace("_", " ")
    return ""


def cleanup_episode_artifacts(root: Path, source_index: int, chunk_size: int) -> None:
    """Remove only partial artifacts created for one failed output episode."""
    chunk = source_index // chunk_size
    data_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{source_index:06d}.parquet"
    data_path.unlink(missing_ok=True)
    for camera_key in CAMERA_KEYS:
        video_path = root / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{source_index:06d}.mp4"
        video_path.unlink(missing_ok=True)


def process_episode(source_index: int, mcap_path: Path, config: Config, root: Path) -> dict | None:
    try:
        episode_id, scalars, cameras, _metadata = read_release_mcap(mcap_path)
        top_topic = select_top_topic(episode_id, cameras)
        validate_required_streams(mcap_path, scalars, cameras, top_topic)
        ticks = common_ticks(scalars, cameras, top_topic, config.fps)
        state, actions = align_state_actions(scalars, ticks, flip_joints=config.flip_joints)
        num_frames = len(ticks)
        task = config.task_override or _source_task(mcap_path)

        chunk = source_index // config.chunk_size
        camera_plan = {**WRIST_TOPICS, "top_camera-images-rgb": top_topic}
        for camera_key, topic in camera_plan.items():
            output = root / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{source_index:06d}.mp4"
            written = transcode_camera_packets(
                cameras[topic],
                ticks,
                output,
                config.resize_size,
                config.fps,
                config.camera_resize_modes[camera_key],
            )
            if written != num_frames:
                raise ValueError(f"{camera_key}: wrote {written} frames, expected {num_frames}")

        frame = pd.DataFrame(
            {
                "state": list(state),
                "actions": list(actions),
                "timestamp": (np.arange(num_frames) / config.fps).astype(np.float32),
                "frame_index": np.arange(num_frames, dtype=np.int64),
                "episode_index": np.full(num_frames, source_index, dtype=np.int64),
                "index": np.arange(num_frames, dtype=np.int64),
                "task_index": np.zeros(num_frames, dtype=np.int64),
            }
        )
        data_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{source_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_path)
        return {
            "source_index": source_index,
            "source_mcap": str(mcap_path.resolve()),
            "source_episode_id": episode_id,
            "source_split": _source_split(mcap_path),
            "source_task": _source_task(mcap_path),
            "task": task,
            "top_topic": top_topic,
            "length": num_frames,
            "stats": _episode_stats(state, actions, frame),
        }
    except Exception:
        cleanup_episode_artifacts(root, source_index, config.chunk_size)
        print(f"episode {source_index} ({mcap_path}) FAILED:\n{traceback.format_exc()}", flush=True)
        return None


def discover_mcaps(input_root: Path, max_episodes: int | None = None) -> list[Path]:
    paths = sorted(path for path in input_root.rglob("episode.mcap") if path.is_file())
    if max_episodes is not None:
        paths = paths[:max_episodes]
    return paths


def _features(config: Config) -> dict:
    features = {
        "state": {"dtype": "float32", "shape": [14], "names": ["state"]},
        "actions": {"dtype": "float32", "shape": [14], "names": ["actions"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera_key in CAMERA_KEYS:
        features[camera_key] = {
            "dtype": "video",
            "shape": [config.resize_size, config.resize_size, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": config.fps,
                "video.height": config.resize_size,
                "video.width": config.resize_size,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def finalize_v21(root: Path, results: dict[int, dict], config: Config) -> tuple[int, int]:
    source_indices = sorted(results)
    tasks: dict[str, int] = {}
    episodes = []
    episodes_stats = []
    manifest_rows = []
    global_offset = 0

    for episode_index, source_index in enumerate(source_indices):
        result = results[source_index]
        task_index = tasks.setdefault(result["task"], len(tasks))
        old_chunk, new_chunk = source_index // config.chunk_size, episode_index // config.chunk_size
        old_data = root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{source_index:06d}.parquet"
        new_data = root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        frame = pd.read_parquet(old_data)
        frame["episode_index"] = np.int64(episode_index)
        frame["index"] = np.arange(global_offset, global_offset + len(frame), dtype=np.int64)
        frame["task_index"] = np.int64(task_index)
        new_data.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(new_data)
        if old_data != new_data:
            old_data.unlink()

        for camera_key in CAMERA_KEYS:
            old_video = root / "videos" / f"chunk-{old_chunk:03d}" / camera_key / f"episode_{source_index:06d}.mp4"
            new_video = root / "videos" / f"chunk-{new_chunk:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"
            if old_video != new_video:
                new_video.parent.mkdir(parents=True, exist_ok=True)
                old_video.rename(new_video)

        length = int(result["length"])
        stats = result["stats"]
        stats["episode_index"] = {
            "min": [float(episode_index)],
            "max": [float(episode_index)],
            "mean": [float(episode_index)],
            "std": [0.0],
            "count": [length],
        }
        stats["index"] = {
            "min": [float(global_offset)],
            "max": [float(global_offset + length - 1)],
            "mean": [float(global_offset) + (length - 1) / 2.0],
            "std": [float(np.arange(length).std())],
            "count": [length],
        }
        stats["task_index"] = {
            "min": [float(task_index)],
            "max": [float(task_index)],
            "mean": [float(task_index)],
            "std": [0.0],
            "count": [length],
        }
        episodes.append({"episode_index": episode_index, "tasks": [result["task"]], "length": length})
        episodes_stats.append({"episode_index": episode_index, "stats": stats})
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "source_mcap": result["source_mcap"],
                "source_episode_id": result["source_episode_id"],
                "source_split": result["source_split"],
                "source_task": result["source_task"],
                "task": result["task"],
                "top_topic": result["top_topic"],
                "length_frames": length,
            }
        )
        global_offset += length

    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "episodes.jsonl").open("w") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as stream:
        for episode in episodes_stats:
            stream.write(json.dumps(episode) + "\n")
    with (meta / "tasks.jsonl").open("w") as stream:
        for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
            stream.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(meta / "source_manifest.csv", index=False)
    manifest.to_csv(root.parent / f"{config.repo_name}_source_manifest.csv", index=False)
    episode_start_video = write_episode_start_video_v21(root, manifest_rows, config)
    episode_start_compact_video = write_compact_episode_start_video_v21(root, manifest_rows, config)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yams",
        "source_format": "abc_release_mcap",
        "alignment": "fixed_clock_nearest",
        "resize_mode": config.resize_mode,
        "camera_resize_modes": config.camera_resize_modes,
        "top_camera_policy": "abc_deterministic_stereo_eye_else_mono",
        "joint_order": "flipped" if config.flip_joints else "driver",
        "action_source": "recorded_release_action_topics",
        "total_episodes": len(episodes),
        "total_frames": global_offset,
        "total_tasks": len(tasks),
        "total_videos": len(CAMERA_KEYS) * len(episodes),
        "total_chunks": (len(episodes) + config.chunk_size - 1) // config.chunk_size,
        "chunks_size": config.chunk_size,
        "fps": config.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "episode_start_video": episode_start_video,
        "episode_start_compact_video": episode_start_compact_video,
        "features": _features(config),
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    return len(episodes), global_offset


def validate_v30(root: Path, expected_episodes: int, expected_frames: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"expected a LeRobot v3 dataset, found {info.get('codebase_version')}")
    if int(info["total_episodes"]) != expected_episodes or int(info["total_frames"]) != expected_frames:
        raise ValueError(
            "v3 totals changed during migration: "
            f"episodes={info['total_episodes']}/{expected_episodes}, "
            f"frames={info['total_frames']}/{expected_frames}"
        )
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise ValueError("v3 migration produced no data parquet files")
    data_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
    if data_rows != expected_frames:
        raise ValueError(f"v3 parquet row count {data_rows} != expected {expected_frames}")
    metadata = LeRobotDatasetMetadata(repo_id=root.name, root=root)
    if metadata.total_episodes != expected_episodes or metadata.total_frames != expected_frames:
        raise ValueError(
            f"LeRobot metadata totals differ: {metadata.total_episodes} episodes, {metadata.total_frames} frames"
        )


def migrate_to_v30(root: Path, config: Config, expected_episodes: int, expected_frames: int) -> None:
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    provenance = root.parent / f"{config.repo_name}_source_manifest.csv"
    info = json.loads((root / "meta" / "info.json").read_text())
    episode_start_metadata = {
        "episode_start_video": info["episode_start_video"],
        "episode_start_compact_video": info["episode_start_compact_video"],
    }
    with tempfile.TemporaryDirectory(prefix=f".{config.repo_name}_preserved_meta_", dir=root.parent) as temp_dir:
        preserve_dir = Path(temp_dir)
        for filename in EPISODE_START_ARTIFACT_NAMES:
            shutil.copy2(root / "meta" / filename, preserve_dir / filename)
        convert_dataset(
            repo_id=config.repo_name,
            root=root,
            push_to_hub=False,
            force_conversion=True,
            data_file_size_in_mb=config.v30_data_file_size_mb,
            video_file_size_in_mb=config.v30_video_file_size_mb,
        )
        for filename in EPISODE_START_ARTIFACT_NAMES:
            shutil.copy2(preserve_dir / filename, root / "meta" / filename)
    if provenance.exists():
        shutil.copy2(provenance, root / "meta" / "source_manifest.csv")
    migrated_info = json.loads((root / "meta" / "info.json").read_text())
    migrated_info.update(episode_start_metadata)
    (root / "meta" / "info.json").write_text(json.dumps(migrated_info, indent=2) + "\n")
    validate_v30(root, expected_episodes, expected_frames)


def main(config: Config) -> None:
    input_root = config.input_root.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    paths = discover_mcaps(input_root, config.max_episodes)
    if not paths:
        raise FileNotFoundError(f"no episode.mcap files under {input_root}")
    root = output_dir / config.repo_name
    if root.exists() or Path(f"{root}_old").exists():
        raise FileExistsError(f"refusing to overwrite existing output or migration backup for {root}")
    (root / "meta").mkdir(parents=True)
    print(f"discovered {len(paths)} release MCAP episodes under {input_root}", flush=True)

    results: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {pool.submit(process_episode, index, path, config, root): index for index, path in enumerate(paths)}
        for processed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                results[result["source_index"]] = result
            if processed % 25 == 0 or processed == len(futures):
                print(f"progress: {processed}/{len(futures)} processed, {len(results)} ok", flush=True)
    if not results:
        raise RuntimeError("all release episodes failed conversion")
    if len(results) != len(paths) and not config.allow_failed_episodes:
        raise RuntimeError(
            f"{len(paths) - len(results)}/{len(paths)} release episodes failed; "
            "partial artifacts were removed and finalization was withheld"
        )

    episodes, frames = finalize_v21(root, results, config)
    print(f"wrote LeRobot v2.1: {episodes} episodes, {frames} frames -> {root}", flush=True)
    print(f"wrote episode-start overview -> {root / 'meta' / EPISODE_START_VIDEO_NAME}", flush=True)
    print(f"wrote compact episode-start overview -> {root / 'meta' / EPISODE_START_COMPACT_VIDEO_NAME}", flush=True)
    if config.convert_to_v30:
        migrate_to_v30(root, config, episodes, frames)
        print(f"validated LeRobot v3.0: {episodes} episodes, {frames} frames -> {root}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
