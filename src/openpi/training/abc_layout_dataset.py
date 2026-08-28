"""Torch dataset over the ABC training layout, yielding openpi YAM-schema samples.

This ports the dataloader approach of the abc project (abc_minimal/train_loop.py:
EpisodeDataset + decode_frame) into openpi, so pi0/pi0.5 policies can train directly
from the MCAP-exported layout produced by scripts/yam_data/export_abc_layout_job.py
(or abc's own export_mcap.py):

    <root>/[train/]<episode_id>/
        states_actions.bin              float64 rows [state(14) | action(14)] per 30Hz tick
        combined_camera-images-rgb.mp4  cameras vstacked (224px each), strict GOP-30 CFR
        episode_metadata.json           task_name, cameras (stack order), num_steps

Frame access: abc synthesizes torchcodec custom_frame_mappings (pts=512k @ 1/15360)
to random-access the strict CFR video without probing. torchcodec 0.4 (openpi's pin,
torch 2.7) predates custom_frame_mappings, but ``seek_mode="approximate"`` computes
the identical pts arithmetic from container metadata — verified byte-identical to
exact mode on these encodes — so each __getitem__ opens a decoder in approximate
mode and indexes the frame, exactly abc's per-sample access pattern.

Sample conventions follow abc, NOT the LeRobot yam converter:
  - state/actions keep the raw MCAP joint order (no np.flip)
  - actions are the *commanded* streams (action-*.mcap), not observed positions
Serving a policy trained on this data must use the same conventions.
"""

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STATE_DIM = 14
ACTION_DIM = 14
ROW_WIDTH = STATE_DIM + ACTION_DIM
ROW_BYTES = ROW_WIDTH * 8

# combined-video stack key -> openpi YAM camera key
CAMERA_KEY_MAP = {
    "top": "top_camera-images-rgb",
    "left": "left_camera-images-rgb",
    "right": "right_camera-images-rgb",
}


@dataclasses.dataclass(frozen=True)
class _Episode:
    ep_dir: Path
    length: int
    usable: int  # frames k where a full action chunk [k, k+horizon) exists
    cameras: tuple[str, ...]  # vstack order in the combined video
    task_name: str
    station_type: str | None


def scan_episodes(
    data_dir: Path, action_horizon: int, station_types: tuple[str, ...] | None = None
) -> list[_Episode]:
    """abc's scan_episodes: any subdir with states_actions.bin is an episode.

    station_types: keep only episodes whose episode_metadata.json station_type is in
    this set (v2 exports record it). None = keep everything.
    """
    episodes = []
    skipped_station = 0
    for ep_dir in sorted(Path(data_dir).iterdir()):
        bin_path = ep_dir / "states_actions.bin"
        if not bin_path.exists():
            continue
        length = bin_path.stat().st_size // ROW_BYTES
        usable = length - (action_horizon - 1)
        if usable <= 0:
            continue
        meta = {}
        meta_path = ep_dir / "episode_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        station = meta.get("station_type")
        if station_types is not None and station not in station_types:
            skipped_station += 1
            continue
        cameras = tuple(meta.get("cameras") or ("top", "left", "right"))
        episodes.append(
            _Episode(ep_dir, length, usable, cameras, meta.get("task_name") or "", station)
        )
    if skipped_station:
        logger.info(f"scan_episodes: station filter {station_types} dropped {skipped_station} episodes")
    return episodes


def read_state_action_rows(ep_dir: Path, start: int, end: int) -> np.ndarray:
    with open(ep_dir / "states_actions.bin", "rb") as f:
        f.seek(start * ROW_BYTES)
        raw = f.read((end - start) * ROW_BYTES)
    return np.frombuffer(raw, dtype=np.float64).reshape(-1, ROW_WIDTH)


def decode_combined_frame(ep_dir: Path, idx: int, cameras: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Random-access one frame of the combined video, split into per-camera HWC uint8."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(
        str(ep_dir / "combined_camera-images-rgb.mp4"),
        seek_mode="approximate",  # exact on these strict-CFR encodes; skips the file scan
        dimension_order="NHWC",
    )
    frame = decoder[idx].numpy()  # (n_cams * H, W, C) uint8
    h = frame.shape[0] // len(cameras)
    return {name: frame[i * h : (i + 1) * h] for i, name in enumerate(cameras)}


class AbcLayoutDataset:
    """Map-style dataset over all usable (episode, frame) pairs of an ABC-layout dataset.

    Yields dicts in the openpi YAM schema so LeRobotYamDataConfig-style transforms
    (YamInputs etc.) apply unchanged:
        left/right/top_camera-images-rgb: (H, W, 3) uint8
        state: (14,) float32
        actions: (action_horizon, 14) float32
        prompt: str
    """

    def __init__(
        self,
        root: Path | str,
        action_horizon: int,
        default_prompt: str | None = None,
        station_types: tuple[str, ...] | None = None,
    ):
        root = Path(root)
        # dataset roots produced by export_abc_layout_job.py have train/ + val/ splits;
        # train on train/. A flat root (episodes directly inside) also works.
        data_dir = root / "train" if (root / "train").exists() else root
        self._episodes = scan_episodes(data_dir, action_horizon, station_types)
        if not self._episodes:
            raise ValueError(f"no ABC-layout episodes found in {data_dir} (station filter: {station_types})")
        self._action_horizon = action_horizon
        self._default_prompt = default_prompt or ""
        self._cum = np.cumsum([e.usable for e in self._episodes])
        logger.info(
            f"AbcLayoutDataset: {len(self._episodes)} episodes, {int(self._cum[-1])} usable frames from {data_dir}"
        )

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, global_idx: int) -> dict:
        ep_idx = int(np.searchsorted(self._cum, global_idx, side="right"))
        k = int(global_idx - (self._cum[ep_idx - 1] if ep_idx > 0 else 0))
        ep = self._episodes[ep_idx]

        rows = read_state_action_rows(ep.ep_dir, k, k + self._action_horizon)
        state = rows[0, :STATE_DIM].astype(np.float32)
        actions = rows[:, STATE_DIM:].astype(np.float32)

        cams = decode_combined_frame(ep.ep_dir, k, ep.cameras)
        sample = {CAMERA_KEY_MAP[name]: img for name, img in cams.items() if name in CAMERA_KEY_MAP}
        sample["state"] = state
        sample["actions"] = actions
        sample["prompt"] = ep.task_name or self._default_prompt
        return sample
