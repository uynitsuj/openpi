"""Validate an explicit DAgger plan; train only when --train is supplied."""

import argparse
import dataclasses
import importlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from openpi.training import data_loader
from openpi.training.dagger_config import build_config
from openpi.training.dagger_config import load_plan
from openpi.training.dagger_dataset import read_json

train = importlib.import_module("scripts.train" if __package__ else "train")


def validate_resume(config, *, resume: bool) -> None:
    """Never reinterpret an existing experiment as a different DAgger round."""
    directory = Path(config.checkpoint_dir)
    if not resume:
        if directory.exists():
            raise FileExistsError(f"Experiment already exists: {directory}; choose a new exp_name or --resume")
        return
    checkpoints = sorted(
        (path for path in directory.glob("*") if path.name.isdecimal() and (path / "_CHECKPOINT_METADATA").is_file()),
        key=lambda path: int(path.name),
    )
    if not checkpoints:
        raise ValueError("--resume requires a completed checkpoint from this DAgger run")
    saved = read_json(checkpoints[-1] / "assets" / "dagger_training.json")
    expected = config.data.components[0].base_config.training_provenance

    def _comparable(provenance: dict) -> dict:
        # Budget/cadence-only extensions are legitimate resumes (e.g. continue
        # a finished 20k run to 60k with sparser checkpoints); data, weights,
        # and normalization stay strict.
        benign = ("num_train_steps", "save_interval", "keep_period", "num_workers")
        clean = json.loads(json.dumps(provenance))
        for key in benign:
            clean.get("plan", {}).pop(key, None)
        return clean

    if _comparable(saved) != _comparable(json.loads(json.dumps(expected))):
        raise ValueError("DAgger plan/data/normalization changed; use a new experiment, not --resume")


def preflight(config, mode: str = "full") -> dict:
    """Validate data/config before allocating model weights.

    full: build every source/split, decode three samples each, audit 10k
      sampler draws, and exercise a real mixture batch (+ validation batch).
      Exhaustive but slow — the old-dataset construction and a full-batch
      decode dominate (an hour-plus on CPU for the 7k-episode datasets).
    fast: everything cheap that catches real corruption — plan/lineage/manifest
      validation (build_config already ran), DAgger export checksum + mask
      verification (their loaders hash every artifact on construction), and
      one decoded sample per DAgger source. The old LeRobot dataset is NOT
      pre-built (training constructs it minutes later and fails loudly there),
      and no full mixture batch is decoded (step 1 of training is that check).
    skip: build_config validation only.
    """
    if mode == "skip":
        logging.warning("preflight: skipped (plan/lineage validation from build_config only)")
        return {"mode": "skip", "initial_weights": dataclasses.asdict(config.weight_loader)}
    components = config.data.create_components(config.assets_dirs, config.model)
    report = {"mode": mode, "components": [], "initial_weights": dataclasses.asdict(config.weight_loader)}
    # val_interval == 0 means a zero-holdout run: no val split exists to decode
    # or count, and the training loop never builds a validation loader.
    splits = ("train", "val") if config.val_interval > 0 else ("train",)
    if config.val_interval == 0:
        logging.info("preflight: validation disabled (no held-out data) — checking train splits only")
    train_sizes = []
    for source, (data_config, weight) in enumerate(components):
        for split in splits:
            current = data_config
            if split == "val":
                current = (
                    dataclasses.replace(current, dagger_split="val")
                    if current.dagger_root
                    else dataclasses.replace(current, episodes=current.val_episodes)
                )
            if mode == "fast" and current.dagger_root is None:
                report["components"].append(
                    {"source": source, "repo": current.repo_id, "split": split, "chunks": None,
                     "note": "fast mode: old dataset validated at training start"}
                )
                if split == "train":
                    train_sizes.append(None)
                continue
            dataset = data_loader.build_torch_dataset(current, config.model, config.model.action_horizon)
            if split == "train":
                train_sizes.append(len(dataset))
            probe = {0} if mode == "fast" else {0, len(dataset) // 2, len(dataset) - 1}
            for index in sorted(probe):
                sample = dataset[index]
                if sample["actions"].shape != (config.model.action_horizon, config.model.action_dim):
                    raise ValueError("Incorrect transformed action shape")
                if not np.all(np.isfinite(sample["state"])) or not np.all(np.isfinite(sample["actions"])):
                    raise ValueError("Nonfinite transformed data")
            report["components"].append(
                {"source": source, "repo": current.repo_id, "split": split, "chunks": len(dataset), "weight": weight}
            )
    if mode == "full":
        catalog = data_loader.WeightedMixtureDataset(
            [range(size) for size in train_sizes],
            [weight for _, weight in components],
            length=config.data.mixture_length or None,
            seed=config.seed,
        )
        draws = [catalog.locate(index) for index in range(min(10_000, len(catalog)))]
        report["sampling_audit"] = {
            "kind": "first virtual catalog indices, not training-order counters",
            "draws": len(draws),
            "source_draws": np.bincount([source for source, _ in draws], minlength=len(components)).tolist(),
            "unique_source_chunks": [
                len({inner for source, inner in draws if source == i}) for i in range(len(components))
            ],
        }
        # Exercise the actual mixture path, including its shared-normalization gate.
        loader = data_loader.create_data_loader(config, num_batches=1)
        next(iter(loader))
        if config.val_interval > 0:
            validation = data_loader.create_mixture_torch_data_loader(config, validation=True, num_batches=1)
            next(iter(validation))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this same experiment, not initialize a new round")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--preflight", choices=("full", "fast", "skip"), default=None,
        help="Check depth before training (default: fast with --train, full standalone)",
    )
    args = parser.parse_args()
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    logging.basicConfig(level=logging.INFO)
    config = build_config(load_plan(args.plan.resolve()))
    if args.train:
        validate_resume(config, resume=args.resume)
    mode = args.preflight or ("fast" if args.train else "full")
    report = preflight(config, mode=mode)
    if args.report:
        # Reports are diagnostics, not export artifacts: overwrite atomically
        # rather than fail-closed (a stale report from an interrupted run must
        # not crash a preflight that has already passed every check).
        tmp = args.report.with_suffix(args.report.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, args.report)
    logging.info("DAgger preflight passed: %s", json.dumps(report))
    if args.train:
        train.main(dataclasses.replace(config, resume=args.resume))


if __name__ == "__main__":
    main()
