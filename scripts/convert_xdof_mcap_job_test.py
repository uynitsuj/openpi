"""Regression coverage for recorded actions and per-camera export preprocessing."""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import importlib
import json
from pathlib import Path

import av
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
from mcap_protobuf.writer import Writer
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def exporter():
    # Import the standalone CLI without the unrelated legacy yam_data package initializer.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parent / "yam_data"))
        yield importlib.import_module("convert_xdof_mcap_job")


@pytest.fixture
def position_message():
    # Only position is consumed by the exporter; keep this fixture independent of xdof-sdk.
    schema = descriptor_pb2.FileDescriptorProto(name="exporter_test.proto", syntax="proto3")
    message = schema.message_type.add(name="Position")
    message.field.add(
        name="position",
        number=1,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Position"))


def write_mcap(path, streams, position_message):
    with path.open("wb") as f:
        writer = Writer(f)
        for topic, (timestamps, positions) in streams.items():
            for timestamp, position in zip(timestamps, positions, strict=True):
                writer.write_message(
                    topic=topic,
                    message=position_message(position=position),
                    log_time=int(timestamp),
                    publish_time=int(timestamp),
                )
        writer.finish()


@pytest.fixture
def raw_episode(tmp_path, position_message):
    raw_dir = tmp_path / "raw" / "episode_test.npy.mp4"
    raw_dir.mkdir(parents=True)
    timestamps = 1_700_000_000 + np.arange(5) / 4
    timestamp_ns = (timestamps * 1e9).astype(np.int64)
    np.save(raw_dir / "timestamp.npy", timestamps)
    (raw_dir / "metadata.json").write_text(json.dumps({"task_name": "packing fixture"}))

    state = np.empty((5, 14), dtype=np.float32)
    actions = np.empty((5, 14), dtype=np.float32)
    for side, offset in [("left", 0), ("right", 7)]:
        joints = np.arange(30).reshape(5, 6) / 100 + offset
        grippers = np.arange(5)[:, None] / 10
        write_mcap(
            raw_dir / f"{side}.mcap",
            {
                f"/{side}-robot-state": (timestamp_ns, joints),
                f"/{side}-gripper-state": (timestamp_ns, grippers),
            },
            position_message,
        )
        state[:, offset : offset + 6] = joints
        state[:, offset + 6] = grippers[:, 0]

        command_joints = np.arange(18).reshape(3, 6) / 10 + offset + 1
        command_grippers = np.array([[0.2], [0.8], [0.4]])
        # Commands and grippers have independent clocks and fewer samples than observations.
        write_mcap(
            raw_dir / f"action-{side}.mcap",
            {
                f"/action-{side}-robot-state": (timestamp_ns[::2] - 40_000_000, command_joints),
                f"/action-{side}-gripper-state": (timestamp_ns[::2] + 40_000_000, command_grippers),
            },
            position_message,
        )
        actions[:, offset : offset + 6] = command_joints[[0, 1, 1, 2, 2]]
        actions[:, offset + 6] = command_grippers[[0, 0, 1, 1, 2], 0]

    # The centered square is green. Padding must preserve the red/blue edges and add black bars.
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    image[:, 16:48, 1] = 255
    image[:, 48:, 2] = 255
    for camera in ("left", "right", "top"):
        np.save(raw_dir / f"{camera}_camera-timestamp.npy", timestamps)
        with av.open(str(raw_dir / f"{camera}_camera-images-rgb.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=4)
            stream.width, stream.height = 64, 32
            stream.pix_fmt = "yuv420p"
            for _ in timestamps:
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    return raw_dir, state, actions


@pytest.mark.parametrize("flip_joints", [False, True])
def test_aligns_recorded_actions_independently(exporter, raw_episode, flip_joints):
    raw_dir, expected_state, expected_actions = raw_episode
    if flip_joints:
        order = [5, 4, 3, 2, 1, 0, 6, 12, 11, 10, 9, 8, 7, 13]
        expected_state = expected_state[:, order]
        expected_actions = expected_actions[:, order]
    state, actions = exporter.build_state_and_actions(raw_dir, flip_joints=flip_joints)
    np.testing.assert_array_equal(state, expected_state)
    np.testing.assert_array_equal(actions, expected_actions)


@pytest.mark.parametrize(
    ("resize_mode", "trim_frame", "expected_modes"),
    [
        (None, None, ["pad", "pad", "center_crop"]),
        (None, 2, ["pad", "pad", "center_crop"]),
        ("pad", None, ["pad", "pad", "pad"]),
        ("center_crop", None, ["center_crop", "center_crop", "center_crop"]),
    ],
)
def test_export_preserves_commands_stats_and_camera_preprocessing(
    exporter, tmp_path, raw_episode, monkeypatch, resize_mode, trim_frame, expected_modes
):
    raw_dir, state, actions = raw_episode
    csv_path = tmp_path / "episodes.csv"
    pd.DataFrame(
        [{"id": "task-1", "nfs_path": str(raw_dir), "created_at": "2026-09-08", "operator": "test", "duration_s": 2}]
    ).to_csv(csv_path, index=False)
    cfg = exporter.Config(
        episode_csv=str(csv_path),
        output_dir=tmp_path / "export",
        raw_cache_dir=raw_dir.parent,
        resize_size=32,
        fps=4,
        min_duration_s=0,
        max_workers=1,
        flip_joints=False,
        keep_raw=True,
        trim_tails=trim_frame is not None,
    )
    if resize_mode is not None:
        cfg = dataclasses.replace(cfg, resize_mode=resize_mode)

    def local_sync(command, **kwargs):
        assert "action-left.mcap" in command
        assert "action-right.mcap" in command

    monkeypatch.setattr(exporter.subprocess, "run", local_sync)
    monkeypatch.setattr(exporter, "ProcessPoolExecutor", ThreadPoolExecutor)
    if trim_frame is not None:

        def trim_observations(observations, park, buffer_s):
            np.testing.assert_array_equal(observations, state[:-1])
            return trim_frame, "ok"

        monkeypatch.setattr(exporter, "detect_trim", trim_observations)
    exporter.main(cfg)

    root = cfg.output_dir / cfg.repo_name
    n = trim_frame if trim_frame is not None else len(state) - 1
    table = pd.read_parquet(root / "data/chunk-000/episode_000000.parquet")
    np.testing.assert_array_equal(np.stack(table["state"]), state[:n])
    np.testing.assert_array_equal(np.stack(table["actions"]), actions[:n])
    assert not np.array_equal(np.stack(table["state"]), np.stack(table["actions"]))
    stats = json.loads((root / "meta/episodes_stats.jsonl").read_text())["stats"]
    np.testing.assert_allclose(stats["actions"]["mean"], actions[:n].mean(axis=0))
    np.testing.assert_allclose(stats["actions"]["std"], actions[:n].std(axis=0))
    assert stats["actions"]["count"] == [n]
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_frames"] == n
    assert info["action_source"] == "action_mcap"
    assert info["joint_order"] == "driver"
    assert info["camera_resize_modes"] == dict(zip(exporter.CAMERA_KEYS, expected_modes, strict=True))
    if resize_mode is None:
        assert info["resize_mode"] == "top_center_crop"
    manifest = pd.read_csv(root / "meta/source_manifest.csv")
    assert manifest.loc[0, "length_frames"] == n
    assert manifest.loc[0, "trimmed_s"] == (len(state) - 1 - n) / cfg.fps

    for camera, mode in zip(exporter.CAMERA_KEYS, expected_modes, strict=True):
        with av.open(str(root / "videos/chunk-000" / camera / "episode_000000.mp4")) as video:
            frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video.streams.video[0])]
        assert len(frames) == n
        image = frames[0]
        assert image.shape == (32, 32, 3)
        if mode == "center_crop":
            assert image[..., 1].mean() > 240
            assert image[..., [0, 2]].mean() < 10
        else:
            # H.264 is lossy, so black padding can contain small color residuals.
            assert image[:6].max() < 20
            assert image[-6:].max() < 20
            assert image[12:20, :6, 0].mean() > 240
            assert image[12:20, -6:, 2].mean() > 240


@pytest.mark.parametrize("missing", ["file", "gripper_topic"])
def test_missing_commands_never_fall_back_to_observations(
    exporter, tmp_path, raw_episode, position_message, monkeypatch, missing
):
    raw_dir, _, _ = raw_episode
    command_path = raw_dir / "action-right.mcap"
    if missing == "file":
        command_path.unlink()
    else:
        write_mcap(
            command_path,
            {"/action-right-robot-state": ([1_700_000_000_000_000_000], [[1.0] * 6])},
            position_message,
        )
    monkeypatch.setattr(exporter.subprocess, "run", lambda *args, **kwargs: None)
    output = tmp_path / "export"
    cfg = exporter.Config(episode_csv="unused", raw_cache_dir=raw_dir.parent, keep_raw=True, fps=4)
    assert exporter.process_episode(0, str(raw_dir), cfg, output) is None
    assert not (output / "data").exists()
