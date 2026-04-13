#!/usr/bin/env python3
"""
Streamlined OpenPI Training Launcher

This script automates the entire training pipeline given a config name:
1. Upload dataset + norm stats to S3
2. Generate/update SkyPilot configuration
3. Launch training job

Usage:
    uv run sky/launch_training.py --config-name pi0_xmi_rby_low_mem_finetune

    ensure you are running this from the openpi directory (not the sky directory)
"""

import sys
import tempfile
import yaml
from pathlib import Path
from typing import Optional, List
import tyro
from dataclasses import dataclass, field

import openpi.training.config as _config
import lerobot.datasets.lerobot_dataset as lerobot_dataset
from datetime import datetime

from openpi.utils.sky_utils import (
    check_prerequisites,
    upload_dataset_to_s3,
    generate_sky_config,
    launch_training,
    query_sky_accelerators,
)

@dataclass
class SkyPilotTrainingConfig:
    config_name: str
    exp_name: Optional[str] = None
    service_provider: Optional[List[str]] = field(default_factory=lambda: ["lambda"])
    s3_bucket: str = "s3://xdof-internal-research"
    s3_checkpoint_base: str = "s3://xdof-internal-research/model_ckpts"
    accelerators: str = "A100-80GB:8"
    region: str = "us-west-2"
    cluster_name: Optional[str] = None
    disable_wandb: bool = False
    dry_run: bool = False
    managed: bool = True  # Use sky jobs launch (auto-teardown) instead of sky launch
    idle_minutes: int = 10  # Autostop idle timeout before teardown
    xla_mem_fraction: float = 0.95


def main(cfg: SkyPilotTrainingConfig):

    # Load minimal dataset info to check that it exists in the first place
    config = _config.get_config(cfg.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
    )
    dataset_path = dataset.root

    # Check for norm stats
    if data_config.norm_stats is None:
        raise FileNotFoundError(
            "Normalization stats not found. "
            f"Make sure to run: \nuv run scripts/compute_norm_stats.py --config-name={cfg.config_name}"
        )

    # Generate experiment name if not provided
    if not cfg.exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.exp_name = f"sky_{dataset_path.name}_{timestamp}"

    print("[INFO] OpenPI Training Launcher")
    print(f"  Dataset: {dataset_path}")
    print(f"  Config: {cfg.config_name}")
    print(f"  Experiment: {cfg.exp_name}")
    print(f"  S3 Bucket: {cfg.s3_bucket}")
    print(f"  Mode: {'managed job (auto-teardown)' if cfg.managed else 'cluster (manual teardown)'}")
    print()

    wandb_api_key = None
    if not cfg.disable_wandb:
        print("[INFO] Wandb is enabled")
        import wandb
        wandb_api_key = wandb.Api().api_key
        if not wandb_api_key:
            print("[WARN] Wandb API key not found. Disabling wandb.")
            wandb_api_key = None
            cfg.disable_wandb = True
    else:
        print("[INFO] Wandb is disabled")

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    if len(cfg.service_provider) > 1:
        cheapest_service_provider = query_sky_accelerators(cfg.accelerators, cfg.region, cfg.service_provider)
        print(f"[INFO] Cheapest provider for {cfg.accelerators}: {cheapest_service_provider['cheapest_option']['CLOUD']}")
        service_provider = cheapest_service_provider['cheapest_option']['CLOUD'].lower()
    else:
        service_provider = cfg.service_provider[0]
        print(f"[INFO] Using {service_provider} as the service provider")

    # Generate dataset name and S3 paths
    s3_dataset_prefix = dataset_path.parent.name
    repo_id = f"{s3_dataset_prefix}/{dataset_path.name}"

    # Upload dataset to S3
    dataset_s3_path = upload_dataset_to_s3(
        dataset_path,
        cfg.s3_bucket,
        repo_id,
        config.assets_dirs,
    )

    # Generate SkyPilot configuration (single function handles all providers)
    print(f"[INFO] Generating SkyPilot configuration for {service_provider}...")
    sky_config = generate_sky_config(
        cloud=service_provider,
        dataset_s3_path=dataset_s3_path,
        config_name=cfg.config_name,
        exp_name=cfg.exp_name,
        repo_id=repo_id,
        s3_checkpoint_base=cfg.s3_checkpoint_base,
        wandb_api_key=wandb_api_key,
        accelerators=cfg.accelerators,
        region=cfg.region,
        idle_minutes=cfg.idle_minutes,
        xla_mem_fraction=cfg.xla_mem_fraction,
    )

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sky_config, f, default_flow_style=False, sort_keys=False)
        config_file = Path(f.name)

    print(f"[OK] Configuration saved to: {config_file}")

    if cfg.dry_run:
        print("[INFO] Dry run complete - not launching training")
        print(f"  To launch manually, run: sky launch {config_file}")
        return

    # Launch training
    try:
        launch_training(config_file, cfg.cluster_name, managed=cfg.managed)
        print(f"[OK] Training checkpoints will be saved to {cfg.s3_checkpoint_base}/{cfg.config_name}/{cfg.exp_name}")
    except Exception as e:
        print(f"[ERROR] Training launch failed: {e}")
        sys.exit(1)
    finally:
        if config_file.exists():
            config_file.unlink()

if __name__ == "__main__":
    cfg = tyro.cli(SkyPilotTrainingConfig)
    main(cfg)
