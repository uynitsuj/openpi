"""CPU video decoding for local LeRobot v3 datasets, without torchvision I/O."""

from pathlib import Path

import av
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch


def decode_timestamps(path: Path, timestamps: list[float], tolerance_s: float) -> torch.Tensor:
    """Preserve LeRobot's nearest-frame semantics and explicit timestamp tolerance."""
    requested = np.asarray(timestamps, dtype=np.float64)
    if requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all():
        raise ValueError("Invalid video query timestamps")
    frames, actual = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        # Start at the previous keyframe so the nearest predecessor is available.
        target = int(max(0, requested.min()) / float(stream.time_base))
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing video timestamp: {path}")
            timestamp = float(frame.pts * stream.time_base)
            actual.append(timestamp)
            frames.append(frame.to_ndarray(format="rgb24"))
            if timestamp >= requested.max():
                break
    if not actual:
        raise ValueError(f"No frames at requested timestamps: {path}")
    distances = np.abs(requested[:, None] - np.asarray(actual)[None, :])
    indices = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(requested)), indices] > tolerance_s):
        raise ValueError(f"Video timestamp mismatch exceeds {tolerance_s}s: {path}")
    pixels = np.stack([frames[index] for index in indices]).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(pixels).float() / 255


class PyAvLeRobotDataset(LeRobotDataset):
    """Override only v3 video I/O; retain LeRobot actions, tasks, and episode indexing."""

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        episode = self.meta.episodes[ep_idx]
        result = {}
        for key, timestamps in query_timestamps.items():
            # v3 concatenates episodes into videos; ignoring this offset reads
            # the first episode's images for later episodes.
            offset = episode[f"videos/{key}/from_timestamp"]
            path = self.root / self.meta.get_video_file_path(ep_idx, key)
            result[key] = decode_timestamps(path, [offset + ts for ts in timestamps], self.tolerance_s).squeeze(0)
        return result
