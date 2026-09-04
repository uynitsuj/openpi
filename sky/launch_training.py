#!/usr/bin/env python3
"""
Streamlined OpenPI Training Launcher

This script automates the entire training pipeline given a config name:
1. Upload dataset + norm stats to S3
2. Generate a single SkyPilot config with multi-cloud auto-failover
3. Launch one job — SkyPilot handles failover across clouds, regions, and GPU types

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
from lerobot.utils.constants import HF_LEROBOT_HOME
from datetime import datetime

from openpi.utils.sky_utils import (
    check_prerequisites,
    upload_dataset_to_s3,
    generate_sky_config,
    launch_training,
)

@dataclass
class SkyPilotTrainingConfig:
    config_name: str
    exp_name: Optional[str] = None
    service_provider: Optional[List[str]] = field(default_factory=lambda: ["aws", "lambda"])
    s3_bucket: str = "s3://xdof-internal-research"
    s3_checkpoint_base: str = "s3://xdof-internal-research/model_ckpts"
    # Full S3 URI for the dataset (overrides the {s3_bucket}/{parent}/{name} layout),
    # e.g. s3://xdof-internal-research/siemens/datasets/industrial_packing_yam
    s3_dataset_path: Optional[str] = None
    accelerators: List[str] = field(default_factory=lambda: ["A100-80GB:8", "A100-80GB:4", "H100:8", "H100:4", "H200:8", "H200:4", "B200:4"])
    provider_regions: dict[str, str] = field(default_factory=lambda: {
        "aws": "us-west-2",
        #"aws": "us-east-1"
    })  # Pin specific providers to a region; unpinned providers failover across all regions
    aws_image_ids: dict[str, str] = field(default_factory=lambda: {
        "us-west-2": "ami-067cc81f948e50e06",
        "us-east-1": "ami-0365bff494b18bf93",
    })
    cluster_name: Optional[str] = None
    disable_wandb: bool = False
    dry_run: bool = False
    managed: bool = True  # Use sky jobs launch (auto-teardown) instead of sky launch
    use_spot: bool = False  # Request spot instances. Default on-demand (no mid-train preemptions).
    idle_minutes: int = 10  # Autostop idle timeout before teardown
    xla_mem_fraction: float = 0.95
    disk_size: int = 512  # Worker disk size in GiB. Sized for dataset + downloaded prior checkpoints + new checkpoint write + jax/orbax tmp.
    resume: bool = False  # Resume from checkpoints already in S3 at s3_checkpoint_base/config/exp_name. Requires --exp-name.
    num_train_steps: Optional[int] = None  # Override TrainConfig.num_train_steps (e.g., extend a resumed run).
    save_interval: Optional[int] = None  # Override TrainConfig.save_interval to control checkpoint save frequency.
    keep_period: Optional[int] = None  # Override TrainConfig.keep_period (orbax keeps checkpoints at step%keep_period==0 forever).


def main(cfg: SkyPilotTrainingConfig):

    # Resolve dataset path without loading the full dataset (avoids hang)
    config = _config.get_config(cfg.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset_path = HF_LEROBOT_HOME / data_config.repo_id
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Make sure the dataset for repo_id={data_config.repo_id} is downloaded."
        )

    # Norm stats: if not present locally, write a tiny stub to the asset
    # dir so the rest of the pipeline (config build, S3 upload step) can run.
    # The worker run-script detects the stub via file-size and runs
    # compute_norm_stats.py with --max-frames 50000 before training, then
    # uploads the real norm_stats back to S3 for parallel/future jobs to
    # reuse. Skipping local compute is required when videos haven't been
    # downloaded locally (torchcodec can't decode placeholders).
    if data_config.norm_stats is None:
        import json
        asset_id_for_stub = data_config.asset_id or data_config.repo_id
        stub_dir = Path(config.assets_dirs) / asset_id_for_stub
        stub_dir.mkdir(parents=True, exist_ok=True)
        stub_path = stub_dir / "norm_stats.json"
        if not stub_path.exists() or stub_path.stat().st_size < 200:
            stub_path.write_text(json.dumps({"norm_stats": {}}))
            print(f"[WARN] Local norm_stats missing; wrote stub at {stub_path}.")
            print("[WARN] Worker will compute real norm_stats with --max-frames=50000 before training.")

    # Generate experiment name if not provided. Resume mode requires the user
    # to specify the original exp_name explicitly so we point at the same
    # checkpoint directory in S3.
    if cfg.resume and not cfg.exp_name:
        raise ValueError("--resume requires --exp-name to be set to the original experiment name.")
    if not cfg.exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.exp_name = f"sky_{cfg.config_name}_{dataset_path.name}_{timestamp}"

    print("[INFO] OpenPI Training Launcher")
    print(f"  Dataset: {dataset_path}")
    print(f"  Config: {cfg.config_name}")
    print(f"  Experiment: {cfg.exp_name}")
    print(f"  S3 Bucket: {cfg.s3_bucket}")
    print(f"  Providers: {cfg.service_provider}")
    print(f"  Accelerators: {cfg.accelerators}")
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

    # S3 upload path mirrors the local cache layout (parent dir + name).
    # The worker REPO_ID, however, must equal data_config.repo_id so that
    # the dataset/norm-stats land at HF_LEROBOT_HOME / data_config.repo_id
    # (where training reads them). When data_config.repo_id has no username
    # prefix (parent is the literal lerobot home), reusing the s3 prefix as
    # REPO_ID would produce a doubled `lerobot/lerobot/...` path.
    s3_dataset_prefix = dataset_path.parent.name
    s3_repo_id = f"{s3_dataset_prefix}/{dataset_path.name}"
    repo_id = data_config.repo_id

    # Resolve norm stats directory
    asset_id = data_config.asset_id or data_config.repo_id
    norm_stats_dir = Path(config.assets_dirs) / asset_id

    # Upload dataset to S3
    dataset_s3_path = upload_dataset_to_s3(
        dataset_path,
        cfg.s3_bucket,
        s3_repo_id,
        norm_stats_dir,
        s3_path_override=cfg.s3_dataset_path,
        norm_stats_subdir=f"norm_stats/{cfg.config_name}",
    )

    # Generate a single SkyPilot config with auto-failover across all
    # providers, regions, and GPU types. SkyPilot's resources.any_of
    # handles failover natively — no need for manual race logic.
    print(f"[INFO] Generating SkyPilot config with auto-failover...")
    for provider in cfg.service_provider:
        region = cfg.provider_regions.get(provider, "any")
        print(f"  - {provider}/{region}: {cfg.accelerators}")

    sky_config = generate_sky_config(
        providers=cfg.service_provider,
        accelerators=cfg.accelerators,
        dataset_s3_path=dataset_s3_path,
        config_name=cfg.config_name,
        exp_name=cfg.exp_name,
        repo_id=repo_id,
        s3_checkpoint_base=cfg.s3_checkpoint_base,
        wandb_api_key=wandb_api_key,
        provider_regions=cfg.provider_regions,
        aws_image_ids=cfg.aws_image_ids,
        idle_minutes=cfg.idle_minutes,
        xla_mem_fraction=cfg.xla_mem_fraction,
        managed=cfg.managed,
        disk_size=cfg.disk_size,
        resume=cfg.resume,
        num_train_steps_override=cfg.num_train_steps,
        save_interval_override=cfg.save_interval,
        keep_period_override=cfg.keep_period,
        use_spot=cfg.use_spot,
    )

    config_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sky_config, f, default_flow_style=False, sort_keys=False)
            config_file = Path(f.name)

        if cfg.dry_run:
            print("\n[INFO] Dry run — generated config:")
            print(f"  {config_file}")
            print(yaml.dump(sky_config, default_flow_style=False, sort_keys=False))
            return

        # Default the sky job name to the exp_name so `sky jobs queue` shows
        # which config each job is for, instead of auto-generated sky-XXXX names.
        job_name = cfg.cluster_name or cfg.exp_name
        launch_training(config_file, cluster_name=job_name, managed=cfg.managed)
        print(f"[OK] Checkpoints: {cfg.s3_checkpoint_base}/{cfg.config_name}/{cfg.exp_name}")
    except Exception as e:
        print(f"[ERROR] Launch failed: {e}")
        sys.exit(1)
    finally:
        if config_file:
            try:
                config_file.unlink(missing_ok=True)
            except Exception:
                pass

if __name__ == "__main__":
    cfg = tyro.cli(SkyPilotTrainingConfig)
    main(cfg)
