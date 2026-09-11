"""Explicit, reproducible Siemens DAgger continuation plans."""

import dataclasses
import logging
from pathlib import Path
import re

import numpy as np

from openpi.shared import normalize
from openpi.training import config as config_lib
from openpi.training import weight_loaders
from openpi.training.dagger_dataset import CAMERAS
from openpi.training.dagger_dataset import file_hash
from openpi.training.dagger_dataset import read_json


@dataclasses.dataclass(frozen=True)
class TrainingPlan:
    initial_checkpoint: str
    old_dataset_root: str
    old_split_manifest: str
    dagger_root: str
    exp_name: str
    old_repo_id: str = "siemens_simple_d405_v12dj_recent"
    base_config: str = "pi05_siemens_simple_d405_v12dj_recent_bs128"
    asset_id: str = "siemens_simple_d405_v12dj_recent"
    weights: tuple[float, ...] = (0.8, 0.1, 0.1)
    num_train_steps: int = 20_000
    batch_size: int = 128
    num_workers: int = 8
    fsdp_devices: int = 2
    seed: int = 0
    checkpoint_base_dir: str = "./checkpoints"
    learning_rate: float = 1e-5
    wandb_enabled: bool = False


def load_plan(path: Path) -> TrainingPlan:
    raw = read_json(path)
    for key in ("initial_checkpoint", "old_dataset_root", "old_split_manifest", "dagger_root", "checkpoint_base_dir"):
        if key in raw:
            raw[key] = str((path.parent / raw[key]).resolve())
    if "weights" in raw:
        raw["weights"] = tuple(raw["weights"])
    return TrainingPlan(**raw)


def build_config(plan: TrainingPlan) -> config_lib.TrainConfig:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", plan.exp_name):
        raise ValueError("exp_name must be a single safe directory name")
    if min(plan.batch_size, plan.num_train_steps, plan.fsdp_devices) <= 0 or plan.num_workers < 0:
        raise ValueError("Training sizes must be positive; worker count cannot be negative")
    if not np.isfinite(plan.learning_rate) or plan.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if len(plan.weights) != 3 or any(not np.isfinite(w) or w < 0 for w in plan.weights) or plan.weights[0] <= 0:
        raise ValueError("Expected nonnegative weights for old / new teleop / new policy, with positive old weight")
    base = config_lib.get_config(plan.base_config)
    # This recipe deliberately preserves the verified v12 camera/action lineage.
    if plan.base_config != "pi05_siemens_simple_d405_v12dj_recent_bs128":
        raise ValueError("This initial DAgger recipe supports the audited v12 config; add other lineages explicitly")
    initial = Path(plan.initial_checkpoint)
    destination = (Path(plan.checkpoint_base_dir) / base.name / plan.exp_name).resolve()
    for source in (initial, Path(plan.old_dataset_root), Path(plan.dagger_root)):
        if destination.is_relative_to(source.resolve()) or source.resolve().is_relative_to(destination):
            raise ValueError("Checkpoint output must not overlap input data or the initial checkpoint")
    if not (initial / "params").is_dir():
        raise ValueError(f"Initial checkpoint must contain params/: {initial}")
    assets = initial / "assets"
    stats_file = assets / plan.asset_id / "norm_stats.json"
    stats = normalize.load(assets / plan.asset_id)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"Missing {key} normalization")
        for field in ("mean", "std", "q01", "q99"):
            values = getattr(stats[key], field)
            if (
                values is None
                or np.asarray(values).ndim != 1
                or len(values) != base.model.action_dim
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(f"Invalid v12 normalization {key}/{field}")
        if len({len(getattr(stats[key], field)) for field in ("mean", "std", "q01", "q99")}) != 1:
            raise ValueError(f"Inconsistent normalization dimensions: {key}")
        if np.any(stats[key].std < 0) or np.any(stats[key].q99 < stats[key].q01):
            raise ValueError(f"Invalid normalization ordering: {key}")
    dataset_manifest = read_json(Path(plan.dagger_root) / "manifest.json")
    # Zero-holdout exports (2026-09-11 decision: the corrections corpus is too
    # small to spare) have no val episodes; disable the offline validation loop
    # entirely — evaluation is physical rollouts. Any val presence keeps the
    # base config's cadence.
    has_new_val = any(row["split"] == "val" for row in dataset_manifest["episodes"])
    if not has_new_val:
        logging.warning(
            "DAgger export has no validation split — offline val loop disabled (val_interval=0); "
            "rely on physical evaluation"
        )
    if dataset_manifest.get("image_resize") != "pil_bilinear_224":
        raise ValueError("DAgger export must match v12 PIL BILINEAR resize; reconvert it")
    if dataset_manifest["image_modes"] != dict.fromkeys(CAMERAS, "center_crop"):
        raise ValueError("v12 DAgger requires all three cameras center-cropped")
    prompt = base.data.default_prompt
    if dataset_manifest["prompt"] != prompt:
        raise ValueError("DAgger prompt differs from the v12 training prompt")
    old_root = Path(plan.old_dataset_root)
    old_info = read_json(old_root / "meta" / "info.json")
    if old_info["fps"] != 30:
        raise ValueError("Expected 30 Hz old dataset")
    splits = read_json(Path(plan.old_split_manifest))
    if splits.get("repo_id") != plan.old_repo_id or splits.get("image_modes") != dataset_manifest["image_modes"]:
        raise ValueError("Old split manifest must attest the same repo and image preprocessing")
    if splits.get("action_order") != "driver" or splits.get("action_source") != "leader_joint_targets":
        raise ValueError("Old split manifest must attest v12 driver-order leader-target lineage")
    rows = splits["episodes"]
    indices = [row["episode_index"] for row in rows]
    if len(indices) != old_info["total_episodes"] or set(indices) != set(range(old_info["total_episodes"])):
        raise ValueError("Old split manifest must cover every dataset episode exactly once")
    identities, groups = set(), {}
    partitions = {"train": [], "val": []}
    for row in rows:
        if row["split"] not in partitions or not row["id"] or not row["group"] or row["id"] in identities:
            raise ValueError("Invalid old episode identity/group/split")
        identities.add(row["id"])
        group = row["group"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Old dataset group leaks across splits: {group}")
        groups[group] = row["split"]
        partitions[row["split"]].append(row["episode_index"])
    for row in dataset_manifest["episodes"]:
        if row["id"] in identities:
            raise ValueError(f"Episode appears in both original and DAgger data: {row['id']}")
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError(f"Collection group leaks between old/new splits: {row['group']}")
    if not all(partitions.values()):
        raise ValueError("Old train and validation partitions must both be nonempty")
    # Controller-gain provenance: recorded per-episode by the converter (None →
    # "unrecorded" predates the gravity_comp_profile field). Serving must match
    # the collection tuning — the FAR bair_daggered_checkpoints kp_scale
    # mismatch is the documented failure mode (runbook §1–2). Mixed or
    # unrecorded profiles are allowed (stratify, don't discard) but loud.
    dagger_controller_profiles = sorted(
        {
            profile if profile is not None else "unrecorded"
            for row in dataset_manifest["episodes"]
            for profile in (row.get("controller_profiles") or {"yam": None}).values()
        }
    )
    if dagger_controller_profiles != ["v8dj_recorded"]:
        logging.warning(
            "DAgger data controller profiles are %s, not the recommended ['v8dj_recorded'] — "
            "verify serving gains match collection and stratify evaluation by profile "
            "(docs/market42_dagger_training.md §1–2)",
            dagger_controller_profiles,
        )
    controller_profiles = {
        "old": "lab42 collection tuning (v8dj_recorded-equivalent per sampled audit, runbook §1)",
        "new_dagger": dagger_controller_profiles,
    }
    provenance = {
        "plan": dataclasses.asdict(plan),
        "mixture_order": ["old", "new_teleop", "new_policy"],
        "source_mixture_commit": "ced6d2c3375cba76c8e055ea85a3b3cb1a747873",
        "dagger_manifest_sha256": file_hash(Path(plan.dagger_root) / "manifest.json"),
        "old_split_manifest_sha256": file_hash(Path(plan.old_split_manifest)),
        "old_info_sha256": file_hash(old_root / "meta" / "info.json"),
        "norm_stats_sha256": file_hash(stats_file),
        "image_modes": dataset_manifest["image_modes"],
        "prompt": prompt,
        "controller_profiles": controller_profiles,
        "action_source": "old_leader_targets_new_verified_follower_commands",
        "chunk_selection": "reviewed_valid_full_horizon_single_authority_segment",
        "new_data_holdout": has_new_val,
        "optimizer_policy": "initialize from checkpoint weights with a fresh optimizer; --resume is same-run only",
    }
    shared_assets = config_lib.AssetsConfig(assets_dir=str(assets), asset_id=plan.asset_id)
    components = [
        config_lib.LeRobotYamDataConfig(
            repo_id=plan.old_repo_id,
            assets=shared_assets,
            default_prompt=prompt,
            base_config=config_lib.DataConfig(
                local_dataset_root=str(old_root),
                video_backend="pyav",
                prompt_from_task=True,
                full_action_chunks=True,
                episodes=tuple(sorted(partitions["train"])),
                val_episodes=tuple(sorted(partitions["val"])),
                training_provenance=provenance,
            ),
        )
    ]
    components.extend(
        config_lib.LeRobotYamDataConfig(
            repo_id=f"market42_dagger_{authority}",
            assets=shared_assets,
            default_prompt=prompt,
            base_config=config_lib.DataConfig(dagger_root=plan.dagger_root, dagger_authority=authority),
        )
        for authority, weight in zip(("teleop", "policy"), plan.weights[1:], strict=True)
        if weight > 0
    )
    # A virtual catalog at least as large as the training budget avoids repeatedly
    # traversing a small, fixed set of RNG-seeded mixture draws.
    mixture = config_lib.MixtureDataConfigFactory(
        components=tuple(components),
        weights=tuple(w for w in plan.weights if w > 0),
        require_shared_normalization=True,
        mixture_length=plan.num_train_steps * plan.batch_size,
    )
    return dataclasses.replace(
        base,
        data=mixture,
        # Surfaced to the policy server so serving stacks can check the station's
        # active controller profile against the collection tuning before running.
        policy_metadata={**(base.policy_metadata or {}), "controller_profiles": controller_profiles},
        exp_name=plan.exp_name,
        batch_size=plan.batch_size,
        num_train_steps=plan.num_train_steps,
        num_workers=plan.num_workers,
        fsdp_devices=plan.fsdp_devices,
        seed=plan.seed,
        checkpoint_base_dir=plan.checkpoint_base_dir,
        wandb_enabled=plan.wandb_enabled,
        val_interval=(base.val_interval if has_new_val else 0),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(initial / "params")),
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            peak_lr=plan.learning_rate,
            decay_lr=plan.learning_rate / 4,
            warmup_steps=min(base.lr_schedule.warmup_steps, plan.num_train_steps // 20),
            decay_steps=plan.num_train_steps,
        ),
        overwrite=False,
        resume=False,
        rabc_enabled=False,
        online_rm_enabled=False,
    )
