#!/usr/bin/env python3
"""Backfill the episode-start overview for an existing LeRobot v3 dataset.

New ABC release conversions create this artifact automatically. This utility
exists for datasets converted before that behavior was added.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import av
import convert_abc_release_mcap_job as abc_converter
import numpy as np
import pandas as pd
import tyro


@dataclasses.dataclass(frozen=True)
class Config:
    dataset_root: Path
    output_fps: int = 2


class PackedV30FrameLoader:
    """Seek to each episode boundary in LeRobot v3 packed camera videos."""

    def __init__(self, root: Path, episodes: pd.DataFrame):
        self._root = root
        self._episodes = episodes.set_index("episode_index", drop=False)
        self._containers: dict[Path, tuple[av.container.InputContainer, av.video.stream.VideoStream]] = {}

    def close(self) -> None:
        for container, _stream in self._containers.values():
            container.close()
        self._containers.clear()

    def _open(self, path: Path) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
        if path not in self._containers:
            container = av.open(str(path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            self._containers[path] = (container, stream)
        return self._containers[path]

    def _read_at(self, path: Path, timestamp: float) -> np.ndarray:
        container, stream = self._open(path)
        if stream.time_base is None:
            raise ValueError(f"video has no time base: {path}")
        seek_offset = max(0, int(timestamp / float(stream.time_base)))
        container.seek(seek_offset, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time >= timestamp - 1e-3:
                return frame.to_ndarray(format="rgb24")
        raise ValueError(f"could not decode frame at {timestamp:.6f}s from {path}")

    def __call__(self, episode_index: int) -> dict[str, np.ndarray]:
        row = self._episodes.loc[episode_index]
        frames = {}
        for camera_key in abc_converter.EPISODE_START_CAMERA_ORDER:
            prefix = f"videos/{camera_key}"
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            timestamp = float(row[f"{prefix}/from_timestamp"])
            video_path = self._root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            frames[camera_key] = self._read_at(video_path, timestamp)
        if episode_index % 25 == 0 or episode_index == len(self._episodes) - 1:
            print(f"episode-start overview: {episode_index + 1}/{len(self._episodes)}", flush=True)
        return frames


def _load_episode_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no LeRobot v3 episode metadata under {root / 'meta' / 'episodes'}")
    episodes = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    return episodes.sort_values("episode_index").reset_index(drop=True)


def main(config: Config) -> None:
    root = config.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    manifest_path = root / "meta" / "source_manifest.csv"
    if not info_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"expected info.json and source_manifest.csv under {root / 'meta'}")

    info = json.loads(info_path.read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"backfill utility expects LeRobot v3, found {info.get('codebase_version')}")
    episodes = _load_episode_metadata(root)
    manifest = pd.read_csv(manifest_path).sort_values("episode_index").reset_index(drop=True)
    if len(episodes) != len(manifest) or not np.array_equal(episodes["episode_index"], manifest["episode_index"]):
        raise ValueError("episode metadata and source manifest do not have identical episode indices")

    first_camera = abc_converter.EPISODE_START_CAMERA_ORDER[0]
    resize_size = int(info["features"][first_camera]["shape"][0])
    loader = PackedV30FrameLoader(root, episodes)
    try:
        overview = abc_converter.write_episode_start_video(
            root,
            manifest.to_dict("records"),
            resize_size=resize_size,
            video_fps=config.output_fps,
            frame_loader=loader,
        )
    finally:
        loader.close()
    info["episode_start_video"] = overview
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {root / 'meta' / abc_converter.EPISODE_START_VIDEO_NAME}", flush=True)
    print(f"wrote {root / 'meta' / abc_converter.EPISODE_START_INDEX_NAME}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
