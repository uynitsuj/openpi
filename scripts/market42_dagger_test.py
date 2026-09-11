"""Synthetic MCAP/video regression tests; no hardware, network, or GPU needed."""

# Heavy training imports are intentionally lazy; patching tiny backbones is test-only.
# ruff: noqa: PLC0415, SLF001

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
import pytest

from openpi.training.dagger_config import TrainingPlan
from openpi.training.dagger_config import build_config
from openpi.training.dagger_dataset import DaggerDataset
from openpi.training.dagger_dataset import full_chunk_starts

with pytest.MonkeyPatch.context() as _import_patch:
    _import_patch.syspath_prepend(str(Path(__file__).parent / "yam_data"))
    converter = importlib.import_module("convert_market42_dagger")

EPOCH = 1_789_100_000.0


def save_json(path, value):
    path.write_text(json.dumps(value))


def position_class():
    file = descriptor_pb2.FileDescriptorProto(name="dagger_fixture.proto", package="fixture", syntax="proto3")
    message = file.message_type.add(name="Position")
    message.field.add(name="position", number=1, type=1, label=3)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("fixture.Position"))


def write_mcap(path, streams):
    message = position_class()
    with path.open("wb") as output:
        writer = Writer(output)
        for topic, times, values in streams:
            for timestamp, value in zip(times, values, strict=True):
                writer.write_message(topic, message(position=value), log_time=round(timestamp * 1e9))
        writer.finish()


def make_episode(root, name="episode_synthetic", *, n=180):
    episode = root / name
    episode.mkdir()
    times = EPOCH + np.arange(n) / 30
    positions = np.tile(np.linspace(0.1, 0.7, 7), (n, 1)).astype(np.float32)
    positions[:, 0] += np.arange(n) * 0.0001
    arms = [
        {
            "name": f"yam_{side}",
            "node_type": "YamRobotNode",
            "config": {
                "poll_freq": 200.0,
                "command_ramp_duration_s": 0.05,
                "gravity_comp_profile": "v8dj_recorded",
            },
        }
        for side in ("left", "right")
    ]
    nodes = [
        *arms,
        {"name": "policy", "node_type": "PolicyClientNode"},
        {"name": "ik", "node_type": "PassiveGelloIKNode"},
    ]
    segments = [
        {"authority": mode, "started_at": EPOCH + start, "ended_at": EPOCH + end}
        for mode, start, end in (("policy", 0, 3), ("teleop", 3, 6))
    ]
    session = {
        "nodes": nodes,
        "links": [],
        "dagger": {"started_at": EPOCH, "ended_at": EPOCH + 6, "segments": segments},
        "node_metadata": {"yam_left": {"robot_info": {"kp": [80, 80, 80, 10, 10, 10, 20]}}},
    }
    save_json(episode / "session_meta.json", session)
    save_json(episode / "metadata.json", {"station_name": "fixture"})
    (episode / "tmp_write_complete.flag").touch()
    for side in ("left", "right"):
        write_mcap(
            episode / f"{side}.mcap",
            [
                (f"/{side}-robot-state", times, positions[:, :6] - 0.01),
                (f"/{side}-gripper-state", times, positions[:, 6:]),
                (f"/{side}-command-state", times, positions),
            ],
        )
        # Deliberately unrelated leader pose: using this for labels must fail tests.
        write_mcap(episode / f"action-{side}.mcap", [(f"/action-{side}-robot-state", times, positions[:, :6] + 5)])
    for prefix in ("policy", "action-ik"):
        write_mcap(
            episode / f"{prefix}.mcap",
            [(f"/{prefix}-{side}-robot-state", times, positions) for side in ("left", "right")],
        )
    for i, camera in enumerate(converter.CAMERAS):
        np.save(episode / f"{camera}_camera-timestamp.npy", times * (1000 if camera == "top" else 1))
        with av.open(str(episode / f"{camera}_camera-images-rgb.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
            image = np.zeros((32, 48, 3), np.uint8)
            image[:, :, i] = 200
            for _ in times:
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return episode


def review_entry(episode, split="train", *, approved=True):
    return {
        "path": str(episode),
        "group": f"session_{episode.name}",
        "split": split,
        "review": [
            {
                "start_s": start,
                "end_s": end,
                "authority": mode,
                "approved": approved,
                "reviewer": "synthetic-test",
                "reason": "fixture, not approved real training data",
            }
            for mode, start, end in (("policy", 0, 3), ("teleop", 3, 6))
        ],
    }


def make_review(root, entries):
    path = root / "review.json"
    save_json(
        path,
        {
            "prompt": converter.PROMPT,
            "image_modes": dict.fromkeys(converter.CAMERAS, "center_crop"),
            "episodes": entries,
        },
    )
    return path


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    root = tmp_path_factory.mktemp("dagger")
    first = make_episode(root, "episode_train")
    second = make_episode(root, "episode_val")
    review = make_review(root, [review_entry(first), review_entry(second, "val")])
    output = root / "export"
    converter.convert_manifest(review, output)
    return output


def test_converter_uses_relay_not_leader_or_measured_state(export):
    dataset = DaggerDataset(export, authority="teleop", split="train", horizon=30)
    sample = dataset[0]
    assert sample["actions"].shape == (30, 14)
    assert np.max(sample["actions"]) < 1
    np.testing.assert_allclose(sample["actions"][0, :6] - sample["state"][:6], 0.01, atol=1e-6)
    for camera, channel in zip(converter.CAMERAS, range(3), strict=True):
        assert sample[f"{camera}_camera-images-rgb"].shape == (224, 224, 3)
        assert sample[f"{camera}_camera-images-rgb"][100, 100, channel] > 180
    assert dataset.report()["chunks"] > 30
    provenance = json.loads((export / "episode_train" / "provenance.json").read_text())
    assert provenance["session"]["node_metadata"]["yam_left"]["robot_info"]["kp"][3] == 10


def test_split_and_boundaries(export):
    for split in ("train", "val"):
        for authority in ("policy", "teleop"):
            dataset = DaggerDataset(export, authority=authority, split=split, horizon=30)
            for index in range(len(dataset)):
                ep, start = dataset.locate(index)
                data = dataset.arrays[ep]
                assert np.all(data["authority"][start : start + 30] == converter.AUTHORITY[authority])
                assert np.all(data["valid"][start : start + 30])
                assert dataset.episodes[ep]["split"] == split


def test_full_chunks_exclude_internal_gaps_and_same_authority_reentry():
    valid = np.ones(100, bool)
    valid[10] = False
    segment = np.repeat([1, 2], 50)
    starts = full_chunk_starts(valid, segment, 30)
    assert starts.tolist() == list(range(11, 21)) + list(range(50, 71))
    assert not len(full_chunk_starts(np.ones(29, bool), np.ones(29), 30))


@pytest.mark.parametrize("scale", [1, 1000, 1e6, 1e9])
def test_timestamp_units(scale):
    ts = EPOCH + np.arange(5) / 30
    np.testing.assert_allclose(converter.camera_seconds(ts * scale), ts, atol=1e-6)


def test_freshness_never_looks_ahead_or_holds_forever():
    values, valid = converter.floor_sample(
        np.array([1.0, 2.0]), np.array([10, 20]), np.array([0.9, 1.01, 1.9, 2.01]), 0.05
    )
    assert valid.tolist() == [False, True, False, True]
    assert values.tolist() == [10, 10, 10, 20]


def test_review_rejects_wrong_authority_and_requires_reviewer():
    interval = {"start_s": 0, "end_s": 2, "authority": "teleop", "approved": True, "reviewer": "test", "reason": "test"}
    with pytest.raises(ValueError, match="disagrees"):
        converter.reviewed_mask([interval], np.array([EPOCH + 1]), EPOCH, np.array([1]))
    with pytest.raises(ValueError, match="reviewer"):
        converter.reviewed_mask([{**interval, "reviewer": ""}], np.array([EPOCH + 1]), EPOCH, np.array([2]))
    assert not converter.reviewed_mask(
        [{**interval, "approved": False}], np.array([EPOCH + 1]), EPOCH, np.array([2])
    ).any()


def test_nonmonotonic_and_missing_mcap_rejected(tmp_path):
    path = tmp_path / "bad.mcap"
    write_mcap(path, [("/x", [1, 1], [[0], [0]])])
    with pytest.raises(ValueError, match="non-monotonic"):
        converter.read_topics(path, {"/x": 1})
    with pytest.raises(ValueError, match="Missing"):
        converter.read_topics(path, {"/absent": 1})


def test_export_tampering_rejected(export, tmp_path):
    # A separate minimal manifest points to its own artifacts; never mutate the fixture.
    manifest = json.loads((export / "manifest.json").read_text())
    manifest["episodes"] = [manifest["episodes"][0]]
    directory = tmp_path / "episode_train"
    directory.mkdir()
    for name in ("data.npz", "combined.mp4", "provenance.json"):
        (directory / name).write_bytes((export / "episode_train" / name).read_bytes())
    (directory / "provenance.json").write_text("{}")
    save_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="modified"):
        DaggerDataset(tmp_path, authority="policy", split="train", horizon=30)


def test_duplicate_and_leaking_review_rejected(export, tmp_path):
    entry = review_entry(export.parent / "episode_train")
    path = make_review(tmp_path, [entry, {**entry, "split": "val"}])
    with pytest.raises(ValueError, match="Duplicate"):
        converter.convert_manifest(path, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("problem", ["async", "missing_command", "unfinalized", "truncated_video"])
def test_bad_recording_fails_closed(tmp_path, problem):
    episode = make_episode(tmp_path)
    session_path = episode / "session_meta.json"
    session = json.loads(session_path.read_text())
    if problem == "async":
        session["nodes"][0]["config"]["poll_freq"] = None
        save_json(session_path, session)
    elif problem == "missing_command":
        write_mcap(episode / "left.mcap", [("/left-robot-state", [EPOCH], [[0] * 6])])
    elif problem == "unfinalized":
        (episode / "tmp_write_complete.flag").unlink()
    else:
        np.save(episode / "top_camera-timestamp.npy", EPOCH + np.arange(181) / 30)
    review = make_review(tmp_path, [review_entry(episode)])
    output = tmp_path / "bad-export"
    with pytest.raises(ValueError, match="Subscriber-driven|Missing|finalized|frames"):
        converter.convert_manifest(review, output)
    assert not (output / "manifest.json").exists()


@pytest.fixture
def plan(export, tmp_path):
    initial = tmp_path / "initial"
    (initial / "params").mkdir(parents=True)
    assets = initial / "assets" / "siemens_simple_d405_v12dj_recent"
    assets.mkdir(parents=True)
    save_json(
        assets / "norm_stats.json",
        {
            "norm_stats": {
                key: {
                    "mean": [0] * 32,
                    "std": [1] * 32,
                    "q01": [-1] * 32,
                    "q99": [1] * 32,
                }
                for key in ("state", "actions")
            }
        },
    )
    old = tmp_path / "old"
    (old / "meta").mkdir(parents=True)
    save_json(old / "meta" / "info.json", {"fps": 30, "total_episodes": 2})
    splits = tmp_path / "old_splits.json"
    save_json(
        splits,
        {
            "repo_id": "siemens_simple_d405_v12dj_recent",
            "image_modes": dict.fromkeys(converter.CAMERAS, "center_crop"),
            "action_order": "driver",
            "action_source": "leader_joint_targets",
            "episodes": [
                {"episode_index": i, "id": f"original_{i}", "group": f"original_group_{i}", "split": split}
                for i, split in enumerate(("train", "val"))
            ],
        },
    )
    return TrainingPlan(
        initial_checkpoint=str(initial),
        old_dataset_root=str(old),
        old_split_manifest=str(splits),
        dagger_root=str(export),
        exp_name="test_dagger",
        batch_size=2,
        fsdp_devices=1,
        num_workers=0,
        num_train_steps=2,
    )


def test_plan_continues_v12_and_shares_assets(plan):
    config = build_config(plan)
    assert config.weight_loader.params_path == str(Path(plan.initial_checkpoint) / "params")
    # Default weights drop the autonomous-rollout source (2026-09-11 policy).
    assert config.data.weights == (0.8, 0.2)
    assert config.data.require_shared_normalization
    assert config.data.components[0].base_config.episodes == (0,)
    assert config.data.components[0].base_config.val_episodes == (1,)
    assert len({component.assets.assets_dir for component in config.data.components}) == 1
    assert not config.overwrite


def test_plan_rejects_split_leakage(plan):
    path = Path(plan.old_split_manifest)
    splits = json.loads(path.read_text())
    splits["episodes"][1]["group"] = splits["episodes"][0]["group"]
    save_json(path, splits)
    with pytest.raises(ValueError, match="leaks"):
        build_config(plan)


def test_plan_rejects_wrong_weights_and_checkpoint(plan):
    with pytest.raises(ValueError, match="weights"):
        build_config(dataclasses.replace(plan, weights=(0.8, 0.2)))
    with pytest.raises(ValueError, match="params"):
        build_config(dataclasses.replace(plan, initial_checkpoint="/nonexistent/dagger-checkpoint"))


@pytest.mark.parametrize(("weights", "count"), [((1, 0, 0), 1), ((0.8, 0.2, 0), 2), ((0.8, 0.1, 0.1), 3)])
def test_plan_supports_controlled_ablations(plan, weights, count):
    assert len(build_config(dataclasses.replace(plan, weights=weights)).data.components) == count


@pytest.mark.parametrize("mode", ["pad", "center_crop"])
def test_image_preprocessing_matches_v12_converter(mode, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "yam_data"))
    original = importlib.import_module("convert_xdof_mcap_job")
    resize = original.center_crop_resize if mode == "center_crop" else original.resize_with_pad
    rng = np.random.default_rng(0)
    for height, width in ((480, 640), (481, 639), (639, 481)):
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        np.testing.assert_array_equal(converter.resized(pixels, mode), resize(pixels, 224))


def test_backwards_clock_rejected(tmp_path):
    path = tmp_path / "backwards.mcap"
    write_mcap(path, [("/x", [2, 1], [[0], [0]])])
    with pytest.raises(ValueError, match="non-monotonic"):
        converter.read_topics(path, {"/x": 1})


def test_output_cannot_modify_raw_episode(tmp_path):
    episode = make_episode(tmp_path)
    review = make_review(tmp_path, [review_entry(episode)])
    with pytest.raises(ValueError, match="inside a raw recording"):
        converter.convert_manifest(review, episode / "export")


def test_normalization_mismatch_rejected(plan, monkeypatch):
    from openpi.training import config as config_lib
    from openpi.training import data_loader

    monkeypatch.setattr(config_lib._tokenizer, "PaligemmaTokenizer", FixtureTokenizer)
    config = build_config(plan)
    components = config.data.create_components(config.assets_dirs, config.model)
    wrong = {key: dataclasses.replace(stats, mean=stats.mean + 1) for key, stats in components[1][0].norm_stats.items()}
    components[1] = (dataclasses.replace(components[1][0], norm_stats=wrong), components[1][1])
    monkeypatch.setattr(config_lib.MixtureDataConfigFactory, "create_components", lambda *_: components)
    with pytest.raises(ValueError, match="normalization differs"):
        data_loader.create_mixture_torch_data_loader(config)


class FixtureTokenizer:
    """No download: only token packing is mocked; the model still trains."""

    def __init__(self, max_len):
        self.max_len = max_len

    def tokenize(self, prompt, state=None):
        del prompt, state
        return np.ones(self.max_len, np.int32), np.ones(self.max_len, bool)


def make_old_lerobot(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {name: {"dtype": "float32", "shape": (14,), "names": None} for name in ("state", "actions")}
    features.update(
        {
            f"{camera}_camera-images-rgb": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in converter.CAMERAS
        }
    )
    dataset = LeRobotDataset.create(
        "siemens_simple_d405_v12dj_recent",
        30,
        features,
        root=root,
        use_videos=True,
        vcodec="h264",
        video_backend="pyav",
    )
    for episode_index in range(2):
        for _ in range(40):
            dataset.add_frame(
                {
                    "state": np.full(14, 0.1, np.float32),
                    "actions": np.full(14, 0.2, np.float32),
                    "task": converter.PROMPT,
                    **{
                        f"{camera}_camera-images-rgb": np.full((224, 224, 3), 100 + 100 * episode_index, np.uint8)
                        for camera in converter.CAMERAS
                    },
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_full_cpu_training_and_serving_smoke(plan, tmp_path, monkeypatch):
    """Real MCAP -> dataset -> mixture -> pi05 gradient -> checkpoint -> inference.

    Uses tiny language/vision backbones and a fixture tokenizer. This checks
    wiring, not the performance or quality of the production v12 model.
    """
    from openpi.models import pi0
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import policy_config
    from openpi.training import config as config_lib
    from openpi.training import weight_loaders
    from scripts import train_dagger

    old_root = tmp_path / "actual_lerobot"
    make_old_lerobot(old_root)
    # ABC-style weights keep the policy source enabled so the smoke covers all
    # three source kinds even though the production default drops rollouts.
    plan = dataclasses.replace(
        plan,
        old_dataset_root=str(old_root),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        weights=(0.8, 0.1, 0.1),
    )
    monkeypatch.setattr(config_lib._tokenizer, "PaligemmaTokenizer", FixtureTokenizer)
    original_vision = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: original_vision(**{**kwargs, "variant": "mu/14"}))
    config = dataclasses.replace(
        build_config(plan),
        model=Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=30, max_token_len=16
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        val_interval=1,
        num_val_batches=1,
        save_interval=1,
        log_interval=1,
        s3_checkpoint_path=None,
    )
    report = train_dagger.preflight(config)
    assert len(report["components"]) == 6
    assert all(row["chunks"] > 0 for row in report["components"])
    # Later LeRobot episodes share a video file: their per-episode offset must
    # select the second episode's brighter frames, never the first episode.
    old_config = config.data.create(config.assets_dirs, config.model)
    old_validation = train_dagger.data_loader.create_torch_dataset(
        dataclasses.replace(old_config, episodes=(1,)), config.model.action_horizon, config.model
    )
    assert float(old_validation[0]["top_camera-images-rgb"].mean()) > 0.7
    train_dagger.validate_resume(config, resume=False)
    train_dagger.train.main(config)
    saved = Path(config.checkpoint_dir) / "1"
    assert (saved / "params" / "_METADATA").exists()
    assert (saved / "assets" / plan.asset_id / "prompt.txt").read_text() == converter.PROMPT
    assert (saved / "assets" / "dagger_training.json").exists()
    train_dagger.validate_resume(config, resume=True)
    with pytest.raises(FileExistsError):
        train_dagger.validate_resume(config, resume=False)
    changed = build_config(dataclasses.replace(plan, weights=(0.6, 0.2, 0.2)))
    with pytest.raises(ValueError, match="changed"):
        train_dagger.validate_resume(changed, resume=True)
    raw = DaggerDataset(plan.dagger_root, authority="teleop", split="val", horizon=30)[0]
    raw.pop("actions")
    policy = policy_config.create_trained_policy(
        config, saved, sample_kwargs={"num_steps": 1}, default_prompt=converter.PROMPT
    )
    result = policy.infer(raw)
    assert result["actions"].shape == (30, 14)
    assert np.isfinite(result["actions"]).all()
    # Exercise the exact weight-loader class used for a new DAgger round.
    warm = dataclasses.replace(
        config, exp_name="warmstart_smoke", weight_loader=weight_loaders.CheckpointWeightLoader(str(saved / "params"))
    )
    train_dagger.train.main(warm)
    assert (Path(warm.checkpoint_dir) / "1" / "params" / "_METADATA").exists()
