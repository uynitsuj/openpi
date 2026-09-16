"""Regression tests for the official ABC release-MCAP LeRobot converter."""

import importlib
import json
from pathlib import Path

import av
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
from mcap_protobuf.writer import Writer
import numpy as np
import pytest


@pytest.fixture(scope="module")
def converter():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parent / "yam_data"))
        yield importlib.import_module("convert_abc_release_mcap_job")


@pytest.fixture(scope="module")
def message_types():
    schema = descriptor_pb2.FileDescriptorProto(name="abc_release_converter_test.proto", syntax="proto3")
    position = schema.message_type.add(name="Position")
    position.field.add(
        name="position",
        number=1,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )
    video = schema.message_type.add(name="Video")
    video.field.add(
        name="data",
        number=1,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return (
        message_factory.GetMessageClass(pool.FindMessageTypeByName("Position")),
        message_factory.GetMessageClass(pool.FindMessageTypeByName("Video")),
    )


def write_release_mcap(path, scalar_streams, camera_streams, message_types):
    position_message, video_message = message_types
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        for topic, (timestamps, values) in scalar_streams.items():
            for timestamp, value in zip(timestamps, values, strict=True):
                writer.write_message(
                    topic=topic,
                    message=position_message(position=value),
                    log_time=int(timestamp),
                    publish_time=int(timestamp),
                )
        for topic, (timestamps, payloads) in camera_streams.items():
            for timestamp, payload in zip(timestamps, payloads, strict=True):
                writer.write_message(
                    topic=topic,
                    message=video_message(data=payload),
                    log_time=int(timestamp),
                    publish_time=int(timestamp),
                )
        writer.finish()


@pytest.fixture
def release_fixture(tmp_path, converter, message_types):
    path = tmp_path / "train" / "load_the_plates_into_the_dish_rack" / "episode_fixture" / "episode.mcap"
    tick = 500_000_000
    timestamps = 1_700_000_000_000_000_000 + np.arange(5, dtype=np.int64) * tick
    scalar_streams = {}
    expected_state = np.empty((5, 14), dtype=np.float32)
    expected_actions = np.empty((5, 14), dtype=np.float32)
    for topics, target, base in (
        (converter.STATE_TOPICS, expected_state, 0.0),
        (converter.ACTION_TOPICS, expected_actions, 100.0),
    ):
        offset = 0
        for topic, dimension in topics:
            values = base + offset + np.arange(5 * dimension).reshape(5, dimension) / 10
            scalar_streams[topic] = (timestamps, values)
            target[:, offset : offset + dimension] = values
            offset += dimension
    camera_streams = {
        topic: (timestamps, [f"{topic}-{index}".encode() for index in range(5)])
        for topic in (
            "/left-wrist-camera",
            "/right-wrist-camera",
            "/top-left-camera",
            "/top-right-camera",
            "/top-camera",
        )
    }
    write_release_mcap(path, scalar_streams, camera_streams, message_types)
    return path, timestamps, expected_state, expected_actions


def test_reads_release_topics_and_aligns_recorded_actions(converter, release_fixture):
    path, timestamps, expected_state, expected_actions = release_fixture
    episode_id, scalars, cameras, _metadata = converter.read_release_mcap(path)
    assert episode_id == "episode_fixture"
    top_topic = converter.select_top_topic(episode_id, cameras)
    assert top_topic in {"/top-left-camera", "/top-right-camera"}
    converter.validate_required_streams(path, scalars, cameras, top_topic)

    ticks = converter.common_ticks(scalars, cameras, top_topic, fps=2)
    np.testing.assert_array_equal(ticks, timestamps[1:])
    state, actions = converter.align_state_actions(scalars, ticks, flip_joints=False)
    np.testing.assert_array_equal(state, expected_state[1:])
    np.testing.assert_array_equal(actions, expected_actions[1:])
    assert not np.array_equal(state, actions)


def test_joint_flip_is_applied_to_state_and_actions(converter, release_fixture):
    path, _timestamps, expected_state, expected_actions = release_fixture
    episode_id, scalars, cameras, _metadata = converter.read_release_mcap(path)
    top_topic = converter.select_top_topic(episode_id, cameras)
    ticks = converter.common_ticks(scalars, cameras, top_topic, fps=2)
    state, actions = converter.align_state_actions(scalars, ticks, flip_joints=True)
    np.testing.assert_array_equal(state, expected_state[1:][:, converter.JOINT_FLIP_ORDER])
    np.testing.assert_array_equal(actions, expected_actions[1:][:, converter.JOINT_FLIP_ORDER])


def test_requires_all_policy_streams(converter, release_fixture):
    path, _timestamps, _expected_state, _expected_actions = release_fixture
    episode_id, scalars, cameras, _metadata = converter.read_release_mcap(path)
    top_topic = converter.select_top_topic(episode_id, cameras)
    del scalars["/left-arm-action"]
    with pytest.raises(ValueError, match="left-arm-action"):
        converter.validate_required_streams(path, scalars, cameras, top_topic)


@pytest.mark.parametrize(
    ("resize_mode", "expected"),
    [
        (
            "top_center_crop",
            {
                "left_camera-images-rgb": "pad",
                "right_camera-images-rgb": "pad",
                "top_camera-images-rgb": "center_crop",
            },
        ),
        ("pad", dict.fromkeys(("left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"), "pad")),
    ],
)
def test_camera_preprocessing_contract(converter, tmp_path, resize_mode, expected):
    config = converter.Config(input_root=tmp_path, resize_mode=resize_mode)
    assert config.camera_resize_modes == expected


def test_nearest_indices_handles_singleton_stream(converter):
    source = np.asarray([123], dtype=np.int64)
    targets = np.asarray([100, 123, 150], dtype=np.int64)
    np.testing.assert_array_equal(converter.nearest_indices(source, targets), [0, 0, 0])


def test_failed_episode_cleanup_is_scoped(converter, tmp_path):
    target_index = 1001
    neighbor_index = 1002
    chunk = 1
    targets = []
    neighbors = []
    for camera_key in converter.CAMERA_KEYS:
        camera_dir = tmp_path / "videos" / f"chunk-{chunk:03d}" / camera_key
        camera_dir.mkdir(parents=True, exist_ok=True)
        target = camera_dir / f"episode_{target_index:06d}.mp4"
        neighbor = camera_dir / f"episode_{neighbor_index:06d}.mp4"
        target.touch()
        neighbor.touch()
        targets.append(target)
        neighbors.append(neighbor)
    data_dir = tmp_path / "data" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True)
    target_data = data_dir / f"episode_{target_index:06d}.parquet"
    neighbor_data = data_dir / f"episode_{neighbor_index:06d}.parquet"
    target_data.touch()
    neighbor_data.touch()
    targets.append(target_data)
    neighbors.append(neighbor_data)

    converter.cleanup_episode_artifacts(tmp_path, target_index, chunk_size=1000)
    assert all(not path.exists() for path in targets)
    assert all(path.exists() for path in neighbors)


def _write_solid_video(path, color, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as output:
        stream = output.add_stream("h264", rate=30)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        image = np.full((size, size, 3), color, dtype=np.uint8)
        for _ in range(2):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


def test_writes_indexed_three_camera_episode_start_video(converter, tmp_path):
    size = 32
    colors = {
        "top_camera-images-rgb": (230, 20, 20),
        "left_camera-images-rgb": (20, 230, 20),
        "right_camera-images-rgb": (20, 20, 230),
    }
    manifest_rows = []
    for episode_index in range(2):
        for camera_key, color in colors.items():
            _write_solid_video(
                tmp_path / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4",
                tuple(min(255, channel + episode_index * 5) for channel in color),
                size,
            )
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "source_episode_id": f"episode-source-{episode_index}",
                "source_split": "train",
                "source_task": "load the plates into the dish rack",
                "top_topic": "/top-left-camera",
            }
        )

    config = converter.Config(
        input_root=tmp_path,
        resize_size=size,
        convert_to_v30=False,
        episode_start_video_fps=2,
    )
    compact_metadata = converter.write_episode_start_video_v21(tmp_path, manifest_rows, config)

    video_path = tmp_path / "meta" / converter.EPISODE_START_VIDEO_NAME
    index_path = tmp_path / "meta" / converter.EPISODE_START_INDEX_NAME
    assert video_path.is_file()
    assert index_path.is_file()
    metadata = json.loads(index_path.read_text())
    assert compact_metadata["frame_count"] == metadata["frame_count"] == 2
    assert metadata["frame_rate"] == 2
    assert metadata["camera_panel_order"] == converter.EPISODE_START_CAMERA_ORDER
    assert [frame["episode_index"] for frame in metadata["frames"]] == [0, 1]
    assert [frame["source_episode_id"] for frame in metadata["frames"]] == [
        "episode-source-0",
        "episode-source-1",
    ]

    with av.open(str(video_path)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(decoded) == 2
    assert decoded[0].shape == (size + converter.EPISODE_START_HEADER_HEIGHT, size * 3, 3)
    centers = [decoded[0][converter.EPISODE_START_HEADER_HEIGHT + size // 2, size * i + size // 2] for i in range(3)]
    assert int(np.argmax(centers[0])) == 0  # top is red
    assert int(np.argmax(centers[1])) == 1  # left wrist is green
    assert int(np.argmax(centers[2])) == 2  # right wrist is blue


def test_compact_episode_sample_is_deterministic_and_without_replacement(converter):
    manifest_rows = [{"episode_index": index} for index in range(100)]
    first = converter.sample_episode_start_rows(manifest_rows, count=60, seed=7)
    repeated = converter.sample_episode_start_rows(manifest_rows, count=60, seed=7)
    different_seed = converter.sample_episode_start_rows(manifest_rows, count=60, seed=8)

    first_indices = [row["episode_index"] for row in first]
    assert first == repeated
    assert first != different_seed
    assert len(first_indices) == len(set(first_indices)) == 60
    assert first_indices != sorted(first_indices)


def test_compact_episode_sample_rejects_nonpositive_count(converter):
    with pytest.raises(ValueError, match="must be positive"):
        converter.sample_episode_start_rows([{"episode_index": 0}], count=0, seed=0)
