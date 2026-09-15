"""Regression tests for the official ABC release-MCAP LeRobot converter."""

import importlib
from pathlib import Path

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
