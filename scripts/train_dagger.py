"""Validate an explicit DAgger plan; train only when --train is supplied."""

import argparse
import dataclasses
import importlib
import json
import logging
from pathlib import Path

import numpy as np

from openpi.training import data_loader
from openpi.training.dagger_config import build_config
from openpi.training.dagger_config import load_plan
from openpi.training.dagger_dataset import read_json
from openpi.training.dagger_dataset import write_json

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
    if saved != json.loads(json.dumps(expected)):
        raise ValueError("DAgger plan/data/normalization changed; use a new experiment, not --resume")


def preflight(config) -> dict:
    """Decode samples from every source/split before allocating model weights."""
    components = config.data.create_components(config.assets_dirs, config.model)
    report = {"components": [], "initial_weights": dataclasses.asdict(config.weight_loader)}
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
            dataset = data_loader.build_torch_dataset(current, config.model, config.model.action_horizon)
            if split == "train":
                train_sizes.append(len(dataset))
            for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
                sample = dataset[index]
                if sample["actions"].shape != (config.model.action_horizon, config.model.action_dim):
                    raise ValueError("Incorrect transformed action shape")
                if not np.all(np.isfinite(sample["state"])) or not np.all(np.isfinite(sample["actions"])):
                    raise ValueError("Nonfinite transformed data")
            report["components"].append(
                {"source": source, "repo": current.repo_id, "split": split, "chunks": len(dataset), "weight": weight}
            )
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
    args = parser.parse_args()
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    logging.basicConfig(level=logging.INFO)
    config = build_config(load_plan(args.plan.resolve()))
    if args.train:
        validate_resume(config, resume=args.resume)
    report = preflight(config)
    if args.report:
        write_json(args.report, report)
    logging.info("DAgger preflight passed: %s", json.dumps(report))
    if args.train:
        train.main(dataclasses.replace(config, resume=args.resume))


if __name__ == "__main__":
    main()
