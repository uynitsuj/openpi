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
    dagger_root: str
    exp_name: str
    # Optional for zero-holdout runs (no DAgger val split): every old episode
    # trains and the per-episode identity/group bookkeeping is skipped, since
    # there is no offline validation left for it to protect. Runs with a
    # holdout must provide it (leakage checks need real identities).
    old_split_manifest: str | None = None
    # Additional DAgger exports mixed into the same intervention/rollout
    # sources as dagger_root; each authority's weight splits across roots in
    # proportion to their eligible train chunks (uniform over the union).
    extra_dagger_roots: tuple[str, ...] = ()
    old_repo_id: str = "siemens_simple_d405_v12dj_recent"
    base_config: str = "pi05_siemens_simple_d405_v12dj_recent_bs128"
    asset_id: str = "siemens_simple_d405_v12dj_recent"
    # [old teleop demos, DAgger interventions, DAgger autonomous rollouts].
    # 2026-09-11 decision: no autonomous-rollout training (0.0); the
    # ABC-inspired [0.8, 0.1, 0.1] remains a plan-level override.
    weights: tuple[float, ...] = (0.8, 0.2, 0.0)
    # Sirius-style: exclude the last N action chunks of policy segments that
    # end in a takeover (only meaningful when the rollout weight is > 0).
    pre_intervention_exclude_chunks: int = 0
    # Old-dataset video decoder. "pyav" is portable (no CUDA-linked TorchCodec
    # needed — right for offline preflight boxes) but decodes the AV1 videos in
    # slow software; production training should use "torchcodec" (the prod
    # siemens default), which is ~5-10x faster per sample and keeps the GPUs fed.
    old_video_backend: str = "pyav"
    num_train_steps: int = 20_000
    batch_size: int = 128
    num_workers: int = 8
    # None = inherit the base config's checkpoint cadence (5k). Set both to
    # e.g. 1000 for frequent, retained checkpoints during active robot-testing
    # rounds (each save is ~44GB and blocks training for a few minutes).
    save_interval: int | None = None
    keep_period: int | None = None
    fsdp_devices: int = 2
    seed: int = 0
    checkpoint_base_dir: str = "./checkpoints"
    learning_rate: float = 1e-5
    wandb_enabled: bool = False


def load_plan(path: Path) -> TrainingPlan:
    raw = read_json(path)
    for key in ("initial_checkpoint", "old_dataset_root", "old_split_manifest", "dagger_root", "checkpoint_base_dir"):
        if raw.get(key):
            raw[key] = str((path.parent / raw[key]).resolve())
    if raw.get("extra_dagger_roots"):
        raw["extra_dagger_roots"] = tuple(str((path.parent / root).resolve()) for root in raw["extra_dagger_roots"])
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
    # The continuation checkpoint lineage is v12 (the policy family that
    # generated the DAgger data); its config supplies the model, prompt, and
    # normalization assets. The OLD DATASET is ablatable across an explicit
    # allowlist of center-crop-all-cams, driver-order leader-target v12
    # descendants (2026-09-11 ablation: v13dj_recent, v13short25cc) — all
    # normalized with the checkpoint's v12 stats, never their own.
    if plan.base_config != "pi05_siemens_simple_d405_v12dj_recent_bs128":
        raise ValueError("This DAgger recipe continues the audited v12 checkpoint lineage; add others explicitly")
    allowed_old_repos = (
        "siemens_simple_d405_v12dj_recent",
        "siemens_simple_d405_v13dj_recent",
        "siemens_simple_d405_v13short25cc",
    )
    if plan.old_repo_id not in allowed_old_repos:
        raise ValueError(f"Old dataset must be one of the attested cc lineages {allowed_old_repos}")
    if plan.pre_intervention_exclude_chunks < 0:
        raise ValueError("pre_intervention_exclude_chunks must be non-negative")
    initial = Path(plan.initial_checkpoint)
    destination = (Path(plan.checkpoint_base_dir) / base.name / plan.exp_name).resolve()
    for source in (initial, Path(plan.old_dataset_root), Path(plan.dagger_root), *map(Path, plan.extra_dagger_roots)):
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
    dagger_roots = [plan.dagger_root, *plan.extra_dagger_roots]
    if len({str(Path(root).resolve()) for root in dagger_roots}) != len(dagger_roots):
        raise ValueError("Duplicate dagger roots")
    dagger_manifests = {root: read_json(Path(root) / "manifest.json") for root in dagger_roots}
    dataset_manifest = dagger_manifests[plan.dagger_root]
    # An episode present in two exports would train twice under one identity.
    dagger_ids: dict[str, str] = {}
    for root, manifest in dagger_manifests.items():
        for row in manifest["episodes"]:
            if row["id"] in dagger_ids:
                raise ValueError(f"Episode {row['id']} appears in both {dagger_ids[row['id']]} and {root}")
            dagger_ids[row["id"]] = root
    # Zero-holdout exports (2026-09-11 decision: the corrections corpus is too
    # small to spare) have no val episodes; disable the offline validation loop
    # entirely — evaluation is physical rollouts. Any val presence keeps the
    # base config's cadence.
    has_new_val = any(
        row["split"] == "val" for manifest in dagger_manifests.values() for row in manifest["episodes"]
    )
    if not has_new_val:
        logging.warning(
            "DAgger export has no validation split — offline val loop disabled (val_interval=0); "
            "rely on physical evaluation"
        )
    prompt = base.data.default_prompt
    for root, manifest in dagger_manifests.items():
        if manifest.get("image_resize") != "pil_bilinear_224":
            raise ValueError(f"DAgger export must match v12 PIL BILINEAR resize; reconvert {root}")
        if manifest["image_modes"] != dict.fromkeys(CAMERAS, "center_crop"):
            raise ValueError(f"v12 DAgger requires all three cameras center-cropped: {root}")
        if manifest["prompt"] != prompt:
            raise ValueError(f"DAgger prompt differs from the v12 training prompt: {root}")
    old_root = Path(plan.old_dataset_root)
    old_info = read_json(old_root / "meta" / "info.json")
    if old_info["fps"] != 30:
        raise ValueError("Expected 30 Hz old dataset")
    if plan.old_split_manifest is None:
        # Zero-holdout runs only: with no offline validation anywhere, there is
        # no split for identity/group bookkeeping to protect. Every old episode
        # trains; lineage rests on the pinned, audited v12 base config.
        if has_new_val:
            raise ValueError(
                "This plan has held-out DAgger data; runs with a validation split require "
                "old_split_manifest so leakage checks use real episode identities"
            )
        old_episodes = old_val_episodes = None
        old_split_sha = None
        old_data_policy = (
            "zero-holdout: all old episodes train; no per-episode split manifest — "
            "lineage attested by the pinned audited v12 base config"
        )
    else:
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
        for manifest in dagger_manifests.values():
            for row in manifest["episodes"]:
                if row["id"] in identities:
                    raise ValueError(f"Episode appears in both original and DAgger data: {row['id']}")
                if row["group"] in groups and groups[row["group"]] != row["split"]:
                    raise ValueError(f"Collection group leaks between old/new splits: {row['group']}")
        if not all(partitions.values()):
            raise ValueError("Old train and validation partitions must both be nonempty")
        old_episodes = tuple(sorted(partitions["train"]))
        old_val_episodes = tuple(sorted(partitions["val"]))
        old_split_sha = file_hash(Path(plan.old_split_manifest))
        old_data_policy = "per-episode split manifest (train/val partitions with group-leakage checks)"
    # Controller-gain provenance: recorded per-episode by the converter (None →
    # "unrecorded" predates the gravity_comp_profile field). Serving must match
    # the collection tuning — the FAR bair_daggered_checkpoints kp_scale
    # mismatch is the documented failure mode (runbook §1–2). Mixed or
    # unrecorded profiles are allowed (stratify, don't discard) but loud.
    dagger_controller_profiles = sorted(
        {
            profile if profile is not None else "unrecorded"
            for manifest in dagger_manifests.values()
            for row in manifest["episodes"]
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
    # Each authority's plan weight splits across dagger roots in proportion to
    # their eligible train chunks, so sampling is uniform over the union.
    def authority_split(authority: str, weight: float) -> list[tuple[str, float]]:
        counts = {
            root: sum(row["chunks_h30"][authority] for row in manifest["episodes"] if row["split"] == "train")
            for root, manifest in dagger_manifests.items()
        }
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(f"No eligible train {authority} chunks in any dagger root")
        shares = []
        for root, count in counts.items():
            if count == 0:
                logging.warning("dagger root %s has no train %s chunks — omitted from that source", root, authority)
                continue
            shares.append((root, weight * (count / total)))
        return shares

    dagger_sources = {
        f"market42_dagger_{authority}_{Path(root).name}": (root, authority, share)
        for authority, weight in zip(("teleop", "policy"), plan.weights[1:], strict=True)
        if weight > 0
        for root, share in authority_split(authority, weight)
    }
    provenance = {
        "plan": dataclasses.asdict(plan),
        "mixture_order": ["old", *dagger_sources],
        "component_weights": {"old": plan.weights[0], **{name: share for name, (_, _, share) in dagger_sources.items()}},
        "source_mixture_commit": "ced6d2c3375cba76c8e055ea85a3b3cb1a747873",
        "dagger_manifest_sha256": {root: file_hash(Path(root) / "manifest.json") for root in dagger_roots},
        "old_split_manifest_sha256": old_split_sha,
        "old_data_policy": old_data_policy,
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
                video_backend=plan.old_video_backend,
                prompt_from_task=True,
                full_action_chunks=True,
                episodes=old_episodes,
                val_episodes=old_val_episodes,
                training_provenance=provenance,
            ),
        )
    ]
    component_weights = [plan.weights[0]]
    for name, (root, authority, share) in dagger_sources.items():
        components.append(
            config_lib.LeRobotYamDataConfig(
                repo_id=name,
                assets=shared_assets,
                default_prompt=prompt,
                base_config=config_lib.DataConfig(
                    dagger_root=root,
                    dagger_authority=authority,
                    # Sirius-style pre-takeover exclusion; DaggerDataset only
                    # applies it to the policy authority.
                    dagger_pre_intervention_exclude_chunks=plan.pre_intervention_exclude_chunks,
                ),
            )
        )
        component_weights.append(share)
    # A virtual catalog at least as large as the training budget avoids repeatedly
    # traversing a small, fixed set of RNG-seeded mixture draws.
    mixture = config_lib.MixtureDataConfigFactory(
        components=tuple(components),
        weights=tuple(component_weights),
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
        save_interval=(plan.save_interval if plan.save_interval is not None else base.save_interval),
        keep_period=(plan.keep_period if plan.keep_period is not None else base.keep_period),
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
