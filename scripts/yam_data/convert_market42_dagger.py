"""Offline Market42 -> reviewed DAgger dataset. Never modifies raw recordings."""

import argparse
import contextlib
import dataclasses
import logging
from pathlib import Path

import av
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
import numpy as np
from PIL import Image

from openpi.training.dagger_dataset import AUTHORITY
from openpi.training.dagger_dataset import CAMERAS
from openpi.training.dagger_dataset import FORMAT
from openpi.training.dagger_dataset import file_hash
from openpi.training.dagger_dataset import full_chunk_starts
from openpi.training.dagger_dataset import read_json
from openpi.training.dagger_dataset import write_json

logger = logging.getLogger(__name__)
PROMPT = "Pack one transparent bag into the cardboard box and flatten the bag."


@dataclasses.dataclass(frozen=True)
class Options:
    state_max_age_s: float = 0.05
    command_max_age_s: float = 0.05
    producer_max_age_s: float = 0.2
    camera_max_age_s: float = 0.1
    boundary_guard_s: float = 0.1
    command_tolerance_rad: float = 0.05
    gripper_tolerance: float = 0.05

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in dataclasses.asdict(self).values()):
            raise ValueError("All freshness, boundary, and agreement thresholds must be finite and positive")


def camera_seconds(values: np.ndarray) -> np.ndarray:
    """Recognize epoch seconds/ms/us/ns; never confuse ms with ns."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid camera timestamps")
    magnitude = float(np.median(values))
    divisor = 1e9 if magnitude > 1e17 else 1e6 if magnitude > 1e14 else 1e3 if magnitude > 1e11 else 1
    result = values / divisor
    if np.any(np.diff(result) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing")
    return result


def read_topics(path: Path, topics: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    records = {topic: ([], []) for topic in topics}
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=list(topics), log_time_order=False):
            value = np.asarray(decoded.position, dtype=np.float32)
            if value.shape != (topics[channel.topic],) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid positions on {path}:{channel.topic}: {value.shape}")
            times, values = records[channel.topic]
            times.append(message.log_time / 1e9)
            values.append(value)
    result = {}
    for topic, (times, values) in records.items():
        timestamps = np.asarray(times, dtype=np.float64)
        if not len(times) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Missing or non-monotonic channel {path}:{topic}")
        result[topic] = timestamps, np.asarray(values)
    return result


def floor_sample(times: np.ndarray, values: np.ndarray, ticks: np.ndarray, max_age_s: float):
    indices = np.searchsorted(times, ticks, side="right") - 1
    clipped = np.clip(indices, 0, len(times) - 1)
    age = ticks - times[clipped]
    valid = (indices >= 0) & (age >= 0) & (age <= max_age_s)
    return values[clipped], valid


def authority_labels(segments: list[dict], ticks: np.ndarray, ramp_s: float, options: Options):
    authority = np.zeros(len(ticks), dtype=np.int8)
    segment_ids = np.full(len(ticks), -1, dtype=np.int32)
    boundary_valid = np.zeros(len(ticks), dtype=bool)
    previous_end = -np.inf
    for index, segment in enumerate(segments):
        start, end = float(segment["started_at"]), float(segment["ended_at"])
        if not np.isfinite(start + end) or end <= start or start < previous_end:
            raise ValueError("Authority segments must be finite, ordered, disjoint, and closed")
        previous_end = end
        code = AUTHORITY[segment["authority"]]
        inside = (ticks >= start) & (ticks < end)
        authority[inside], segment_ids[inside] = code, index
        boundary_valid |= (ticks >= start + ramp_s + options.boundary_guard_s) & (
            ticks < end - options.boundary_guard_s
        )
    return authority, segment_ids, boundary_valid


def reviewed_mask(intervals: list[dict], ticks: np.ndarray, epoch: float, authority: np.ndarray):
    reviewed = np.zeros(len(ticks), dtype=bool)
    previous_end = -np.inf
    for interval in intervals:
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if not np.isfinite(start + end) or start < 0 or end <= start or start < previous_end:
            raise ValueError("Review intervals must be finite, nonnegative, sorted and non-overlapping")
        previous_end = end
        if interval.get("approved") is not True:
            continue
        if not interval.get("reviewer") or not interval.get("reason"):
            raise ValueError("Approved intervals require reviewer and reason")
        code = AUTHORITY[interval["authority"]]
        inside = (ticks >= epoch + start) & (ticks < epoch + end)
        if np.any(inside & (authority != code)):
            raise ValueError("Approved review interval disagrees with recorded authority")
        reviewed |= inside
    return reviewed


def resized(image: np.ndarray, mode: str) -> np.ndarray:
    """Match convert_xdof_mcap_job's PIL BILINEAR preprocessing exactly."""
    h, w = image.shape[:2]
    if mode == "center_crop":
        size = min(h, w)
        image = image[(h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2]
        return np.asarray(Image.fromarray(image).resize((224, 224), resample=Image.Resampling.BILINEAR))
    if mode != "pad":
        raise ValueError(f"Unknown image mode {mode}")
    ratio = max(w / 224, h / 224)
    width, height = int(w / ratio), int(h / ratio)
    small = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
    result = Image.new("RGB", (224, 224), 0)
    result.paste(small, ((224 - width) // 2, (224 - height) // 2))
    return np.asarray(result)


def export_video(episode: Path, destination: Path, camera_ts: dict, ticks: np.ndarray, modes: dict):
    """Strictly decode all inputs; never repeat a frame to hide a truncated video."""
    with contextlib.ExitStack() as stack:
        containers = {
            camera: stack.enter_context(av.open(str(episode / f"{camera}_camera-images-rgb.mp4"))) for camera in CAMERAS
        }
        decoders = {camera: iter(container.decode(video=0)) for camera, container in containers.items()}
        for container in containers.values():
            container.streams.video[0].thread_count = 2
        positions = dict.fromkeys(CAMERAS, -1)
        current = {}
        wanted = {camera: np.searchsorted(camera_ts[camera], ticks, side="right") - 1 for camera in CAMERAS}
        output = stack.enter_context(av.open(str(destination), mode="w"))
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 224, 672, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast", "g": "30", "bf": "0"}
        for row in range(len(ticks)):
            for camera in CAMERAS:
                while positions[camera] < wanted[camera][row]:
                    frame = next(decoders[camera], None)
                    if frame is None:
                        raise ValueError(f"Truncated {camera} video in {episode}")
                    positions[camera] += 1
                    current[camera] = resized(frame.to_ndarray(format="rgb24"), modes[camera])
            image = np.concatenate([current[camera] for camera in CAMERAS], axis=0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = row
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        for camera in CAMERAS:
            count = positions[camera] + 1 + sum(1 for _ in decoders[camera])
            if count != len(camera_ts[camera]):
                raise ValueError(f"{camera}: {count} frames != {len(camera_ts[camera])} timestamps in {episode}")


def convert_episode(entry: dict, out: Path, prompt: str, modes: dict, options: Options) -> dict:
    episode = Path(entry["path"]).resolve()
    if not any((episode / flag).exists() for flag in ("tmp_write_complete.flag", "write_complete.flag")):
        raise ValueError(f"Recording is not finalized: {episode}")
    session = read_json(episode / "session_meta.json")
    metadata = read_json(episode / "metadata.json")
    dagger = session["dagger"]
    nodes = {node["name"]: node for node in session["nodes"]}
    arms = [node for node in nodes.values() if node.get("node_type") == "YamRobotNode"]
    if len(arms) != 2:
        raise ValueError("Expected exactly two recorded YAM nodes")
    ramp_s = 0.0
    # Recorded collection tuning, surfaced into the manifest so training
    # provenance and serving can match controller profiles without digging
    # through per-episode session metadata. None = recorded before the
    # gravity_comp_profile field existed (unknown stays unknown).
    controller_profiles = {arm["name"]: arm["config"].get("gravity_comp_profile") for arm in arms}
    for arm in arms:
        cfg = arm["config"]
        sync = any(link.get("target_node") == arm["name"] and link.get("sync_group") for link in session["links"])
        if cfg.get("poll_freq") is None and not sync:
            raise ValueError(
                "Subscriber-driven command logs can contain pre-ramp targets; use fixed-rate/sync recordings"
            )
        ramp = cfg.get("command_ramp_duration_s")
        if ramp is None or not np.isfinite(ramp) or ramp <= 0:
            raise ValueError("Unknown/velocity-limited ramp: cannot establish safe handoff exclusion window")
        ramp_s = max(ramp_s, ramp)
    streams, camera_ts = {}, {}
    source_files = {"session_meta.json", "metadata.json"}
    for side in ("left", "right"):
        filename = f"{side}.mcap"
        streams.update(
            read_topics(
                episode / filename,
                {
                    f"/{side}-robot-state": 6,
                    f"/{side}-gripper-state": 1,
                    f"/{side}-command-state": 7,
                },
            )
        )
        source_files.add(filename)
    producers = {}
    for authority, node_type in (("policy", "PolicyClientNode"), ("teleop", "PassiveGelloIKNode")):
        if not any(segment["authority"] == authority for segment in dagger["segments"]):
            continue
        matching = [node for node in nodes.values() if node.get("node_type") == node_type]
        if len(matching) != 1:
            raise ValueError(f"Cannot resolve unique {authority} producer")
        prefix = ("action-" if authority == "teleop" else "") + matching[0]["name"]
        filename = f"{prefix}.mcap"
        producers[authority] = read_topics(
            episode / filename, {f"/{prefix}-{side}-robot-state": 7 for side in ("left", "right")}
        )
        source_files.add(filename)
    for camera in CAMERAS:
        filename = f"{camera}_camera-timestamp.npy"
        camera_ts[camera] = camera_seconds(np.load(episode / filename, allow_pickle=False))
        source_files.update((filename, f"{camera}_camera-images-rgb.mp4"))
    if (episode / "eval_anno.json").exists():
        source_files.add("eval_anno.json")
    first = max(
        float(dagger["started_at"]), *(times[0] for times, _ in streams.values()), *(ts[0] for ts in camera_ts.values())
    )
    last = min(
        float(dagger["ended_at"]), *(times[-1] for times, _ in streams.values()), *(ts[-1] for ts in camera_ts.values())
    )
    if last <= first:
        raise ValueError("No common clock overlap across observations, commands, and cameras")
    ticks = first + np.arange(int((last - first) * 30) + 1) / 30
    authority, segment, boundaries = authority_labels(dagger["segments"], ticks, ramp_s, options)
    reviewed = reviewed_mask(entry.get("review", []), ticks, float(dagger["started_at"]), authority)
    state, actions = np.empty((len(ticks), 14), np.float32), np.empty((len(ticks), 14), np.float32)
    checks = {"handoff_or_unknown_authority": boundaries}
    for i, side in enumerate(("left", "right")):
        for suffix, target, selection in (
            ("robot-state", state, slice(i * 7, i * 7 + 6)),
            ("gripper-state", state, slice(i * 7 + 6, i * 7 + 7)),
            ("command-state", actions, slice(i * 7, i * 7 + 7)),
        ):
            times, values = streams[f"/{side}-{suffix}"]
            max_age = options.command_max_age_s if suffix == "command-state" else options.state_max_age_s
            target[:, selection], checks[f"stale_{side}_{suffix}"] = floor_sample(times, values, ticks, max_age)
    for camera, times in camera_ts.items():
        _, checks[f"stale_{camera}_camera"] = floor_sample(times, times, ticks, options.camera_max_age_s)
    for mode, topics in producers.items():
        for i, (_, (times, values)) in enumerate(topics.items()):
            # read_topics preserves the requested left/right insertion order.
            sampled, fresh = floor_sample(times, values, ticks, options.producer_max_age_s)
            error = np.abs(sampled - actions[:, i * 7 : (i + 1) * 7])
            agrees = np.all(error[:, :6] <= options.command_tolerance_rad, axis=1) & (
                error[:, 6] <= options.gripper_tolerance
            )
            checks[f"{mode}_{i}_producer_stale_or_disagreement"] = (authority != AUTHORITY[mode]) | (fresh & agrees)
    valid = np.logical_and.reduce(list(checks.values()))
    identifier = episode.name
    directory = out / identifier
    directory.mkdir(exist_ok=False)
    np.savez_compressed(
        directory / "data.npz",
        state=state,
        actions=actions,
        timestamps=ticks,
        valid=valid,
        reviewed=reviewed,
        authority=authority,
        segment=segment,
    )
    export_video(episode, directory / "combined.mp4", camera_ts, ticks, modes)
    provenance = {
        "raw_episode": str(episode),
        "review": entry.get("review", []),
        "session": session,
        "metadata": metadata,
        "outcome_annotation": read_json(episode / "eval_anno.json") if "eval_anno.json" in source_files else None,
        "source_files": {
            filename: {"sha256": file_hash(episode / filename), "bytes": (episode / filename).stat().st_size}
            for filename in sorted(source_files)
        },
        "options": dataclasses.asdict(options),
        "ramp_guard_s": ramp_s,
        "prompt": prompt,
        "image_modes": modes,
        "converter_sha256": file_hash(Path(__file__)),
        "image_resize": "pil_bilinear_224",
        "action_source": "verified_follower_command_state",
        "action_order": "left_j1_to_j6_gripper_right_j1_to_j6_gripper",
    }
    write_json(directory / "provenance.json", provenance)
    return {
        "id": identifier,
        "directory": identifier,
        "split": entry["split"],
        "group": entry["group"],
        "controller_profiles": controller_profiles,
        "frames": len(ticks),
        "valid_frames": int(valid.sum()),
        "reviewed_frames": int(reviewed.sum()),
        "rejections": {reason: int((~mask).sum()) for reason, mask in checks.items()},
        "chunks_h30": {
            mode: len(full_chunk_starts(valid & reviewed & (authority == code), segment, 30))
            for mode, code in AUTHORITY.items()
        },
        "checksums": {
            filename: file_hash(directory / filename) for filename in ("data.npz", "combined.mp4", "provenance.json")
        },
    }


def convert_manifest(review_path: Path, output: Path) -> dict:
    review = read_json(review_path)
    prompt = review["prompt"]
    modes = review["image_modes"]
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or set(modes) != set(CAMERAS)
        or any(mode not in ("pad", "center_crop") for mode in modes.values())
    ):
        raise ValueError("Explicit prompt and top/left/right image_modes required")
    options = Options(**review.get("options", {}))
    groups, seen = {}, set()
    if not review["episodes"]:
        raise ValueError("Empty review manifest")
    for entry in review["episodes"]:
        entry["path"] = str((review_path.parent / entry["path"]).resolve())
        if output.resolve().is_relative_to(Path(entry["path"])):
            raise ValueError("Export output must not be inside a raw recording")
        identifier = Path(entry["path"]).name
        if identifier in seen or entry["split"] not in ("train", "val") or not entry["group"]:
            raise ValueError("Duplicate episode, missing group, or invalid split")
        seen.add(identifier)
        if entry["group"] in groups and groups[entry["group"]] != entry["split"]:
            raise ValueError(f"Train/val leakage: {entry['group']}")
        groups[entry["group"]] = entry["split"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "review.json", review)
    episodes = []
    for entry in review["episodes"]:
        logger.info("Converting %s", entry["path"])
        episodes.append(convert_episode(entry, output, prompt, modes, options))
    manifest = {
        "format": FORMAT,
        "fps": 30,
        "prompt": prompt,
        "image_modes": modes,
        "image_resize": "pil_bilinear_224",
        # Distinct recorded YAM gravity_comp_profile values across all episodes;
        # "unrecorded" = collected before the profile field existed. More than
        # one entry means mixed controller regimes (see runbook sections 1–2).
        "controller_profiles": sorted(
            {
                profile if profile is not None else "unrecorded"
                for episode in episodes
                for profile in episode["controller_profiles"].values()
            }
        ),
        "review_sha256": file_hash(output / "review.json"),
        "episodes": episodes,
    }
    # Only publish a loadable manifest after every requested episode succeeds.
    write_json(output / "manifest.json", manifest)
    logger.info("Converted %d episodes: %s", len(episodes), output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing output is never overwritten"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert_manifest(args.review_manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
