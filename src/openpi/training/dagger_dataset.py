"""Validated, reviewed Market42 data; no hardware or model dependencies."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np

FORMAT = "market42-dagger-v1"
CAMERAS = ("top", "left", "right")
AUTHORITY = {"policy": 1, "teleop": 2}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def full_chunk_starts(valid: np.ndarray, segment: np.ndarray, horizon: int) -> np.ndarray:
    """Exclude incomplete chunks, invalid rows, and boundaries, without padding."""
    if horizon <= 0 or valid.ndim != 1 or valid.shape != segment.shape:
        raise ValueError("Invalid horizon or validity/segment shape")
    if len(valid) < horizon:
        return np.empty(0, dtype=np.int64)
    bad = np.concatenate(([0], np.cumsum(~valid)))
    starts = np.arange(len(valid) - horizon + 1)
    # A segment ID denotes one contiguous interval, not just its authority.
    transitions = np.concatenate(([0], np.cumsum(segment[1:] != segment[:-1])))
    keep = (bad[starts + horizon] == bad[starts]) & (transitions[starts + horizon - 1] == transitions[starts])
    return starts[keep]


def decode_frame(path: Path, frame_index: int) -> np.ndarray:
    """Seek a validated CFR export; decode from its preceding keyframe."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        target = round(frame_index / 30 / float(stream.time_base))
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts >= target:
                actual = round(float(frame.pts * stream.time_base) * 30)
                if actual != frame_index:
                    raise ValueError(f"Video alignment error: requested {frame_index}, got {actual}: {path}")
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"Missing frame {frame_index}: {path}")


class DaggerDataset:
    """One reviewed authority/split, uniformly indexed by eligible chunk start."""

    def __init__(self, root: str | Path, *, authority: str, split: str, horizon: int):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("format") != FORMAT or self.manifest.get("fps") != 30:
            raise ValueError("Unsupported DAgger export; reconvert with convert_market42_dagger.py")
        if authority not in AUTHORITY or split not in ("train", "val"):
            raise ValueError("Expected policy/teleop authority and train/val split")
        self.authority, self.split, self.horizon = authority, split, horizon
        self.episodes = []
        self.starts = []
        self.arrays = []
        self.groups = set()
        self.episode_ids = set()
        seen = set()
        group_splits = {}
        for episode in self.manifest["episodes"]:
            if not episode["id"] or not episode["group"] or episode["split"] not in ("train", "val"):
                raise ValueError("Invalid export episode identity/group/split")
            if episode["id"] in seen:
                raise ValueError(f"Duplicate episode: {episode['id']}")
            seen.add(episode["id"])
            group = episode["group"]
            if group in group_splits and group_splits[group] != episode["split"]:
                raise ValueError(f"Train/val leakage in group {group}")
            group_splits[group] = episode["split"]
            if episode["split"] != split:
                continue
            directory = (self.root / episode["directory"]).resolve()
            if not directory.is_relative_to(self.root.resolve()):
                raise ValueError("Episode path escapes export directory")
            for filename, checksum in episode["checksums"].items():
                if filename not in ("data.npz", "combined.mp4", "provenance.json"):
                    raise ValueError(f"Unexpected export artifact: {filename}")
                if file_hash(directory / filename) != checksum:
                    raise ValueError(f"Export modified after validation: {directory / filename}")
            if set(episode["checksums"]) != {"data.npz", "combined.mp4", "provenance.json"}:
                raise ValueError("Incomplete export checksums")
            with np.load(directory / "data.npz", allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            n = len(data["timestamps"])
            if n == 0 or data["timestamps"].shape != (n,) or not np.all(np.isfinite(data["timestamps"])):
                raise ValueError("Invalid export timestamps")
            for key in ("valid", "reviewed", "authority", "segment"):
                if data[key].shape != (n,):
                    raise ValueError(f"Invalid {key} mask shape")
            if data["valid"].dtype != bool or data["reviewed"].dtype != bool:
                raise ValueError("Validity and review masks must be boolean")
            if not np.isin(data["authority"], (0, *AUTHORITY.values())).all():
                raise ValueError("Unknown action authority")
            if data["state"].shape != (n, 14) or data["actions"].shape != (n, 14):
                raise ValueError("Expected 14D follower states/actions")
            if not np.all(np.isfinite(data["state"])) or not np.all(np.isfinite(data["actions"])):
                raise ValueError("Non-finite state/action data")
            if not np.allclose(np.diff(data["timestamps"]), 1 / 30, atol=1e-6, rtol=0):
                raise ValueError("Non-contiguous 30 Hz export timeline")
            valid = data["valid"] & data["reviewed"] & (data["authority"] == AUTHORITY[authority])
            starts = full_chunk_starts(valid, data["segment"], horizon)
            if len(starts):
                self.episodes.append({**episode, "path": directory})
                self.arrays.append(data)
                self.starts.append(starts)
                self.groups.add(group)
                self.episode_ids.add(episode["id"])
        self.cumulative = np.cumsum([len(starts) for starts in self.starts])
        if not len(self.cumulative):
            raise ValueError(f"No eligible {split}/{authority} chunks of length {horizon} in {root}")

    def __len__(self):
        return int(self.cumulative[-1])

    def locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.cumulative, index, side="right"))
        offset = int(self.cumulative[ep - 1]) if ep else 0
        return ep, int(self.starts[ep][index - offset])

    def __getitem__(self, index):
        ep, start = self.locate(int(index))
        data, episode = self.arrays[ep], self.episodes[ep]
        image = decode_frame(episode["path"] / "combined.mp4", start)
        if image.shape != (672, 224, 3):
            raise ValueError(f"Unexpected composite dimensions: {image.shape}")
        return {
            "state": data["state"][start].copy(),
            "actions": data["actions"][start : start + self.horizon].copy(),
            "prompt": self.manifest["prompt"],
            **{f"{camera}_camera-images-rgb": image[i * 224 : (i + 1) * 224] for i, camera in enumerate(CAMERAS)},
        }

    def report(self) -> dict:
        return {
            "authority": self.authority,
            "split": self.split,
            "chunks": len(self),
            "episodes": len(self.episodes),
            "groups": sorted(self.groups),
            "manifest_sha256": file_hash(self.root / "manifest.json"),
        }
