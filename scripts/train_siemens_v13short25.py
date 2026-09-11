#!/usr/bin/env python3
"""Fast, explicit entry point for the two Siemens v13-short25 configs.

The generic training CLI builds Tyro subcommands for the repository's entire
config registry. On the large industrial-packing branch that can take several
minutes before JAX initialization. This wrapper selects one of exactly two
existing configs and passes the resulting TrainConfig directly to train.main.
"""

import argparse
import dataclasses

import train as train_script

from openpi.training import config as training_config
from openpi.training import weight_loaders


CONFIGS = {
    "cc": "pi05_siemens_simple_d405_v13short25cc_bs128",
    "tc": "pi05_siemens_simple_d405_v13short25tc_bs128",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(CONFIGS))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--base-params-uri", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--s3-checkpoint-path", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_params_uri.endswith("/params"):
        raise ValueError("--base-params-uri must end in /params")

    config = training_config.get_config(CONFIGS[args.variant])
    expected_repo = f"siemens_simple_d405_v13short25{args.variant}"
    if config.data.repo_id != expected_repo:
        raise ValueError(f"unexpected repo_id: {config.data.repo_id!r} != {expected_repo!r}")
    if config.batch_size != 128 or config.num_train_steps != 20_000:
        raise ValueError(
            "v13-short25 invariant failed: expected batch_size=128, "
            "num_train_steps=20000"
        )
    if args.fsdp_devices not in (2, 4, 8) or 8 % args.fsdp_devices:
        raise ValueError("--fsdp-devices must be one of 2, 4, or 8 on an eight-GPU node")

    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        weight_loader=weight_loaders.CheckpointWeightLoader(args.base_params_uri),
        checkpoint_base_dir=args.checkpoint_base_dir,
        s3_checkpoint_path=args.s3_checkpoint_path,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    train_script.main(config)


if __name__ == "__main__":
    main()
