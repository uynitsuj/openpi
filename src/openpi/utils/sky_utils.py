#!/usr/bin/env python3
"""
Utility functions for OpenPI SkyPilot training launcher.
"""

import subprocess
from pathlib import Path
from typing import Optional, List


def run_command(cmd: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with proper error handling."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=capture_output,
        text=True,
    )
    if capture_output:
        print(f"[OUTPUT] {result.stdout.strip()}")
    return result


def check_prerequisites():
    """Check if required tools are available."""
    required_tools = ['aws', 'sky']

    for tool in required_tools:
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            print(f"[OK] {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[ERROR] {tool} is not available. Please install it first.")
            return False

    # Check AWS credentials
    try:
        subprocess.run(['aws', 'sts', 'get-caller-identity'], capture_output=True, check=True)
        print("[OK] AWS credentials are configured")
    except subprocess.CalledProcessError:
        print("[ERROR] AWS credentials not configured. Run 'aws configure' first.")
        return False

    return True



def upload_dataset_to_s3(dataset_path: Path, s3_bucket: str, repo_id: str, norm_stats_dir: str) -> str:
    """Upload dataset and norm stats to S3 and return the full S3 path.

    Args:
        dataset_path: Local path to the dataset.
        s3_bucket: S3 bucket URL (e.g. s3://bucket-name).
        repo_id: Dataset repo id used as the S3 key prefix (e.g. lerobot/dataset_name).
        norm_stats_dir: Local directory containing the norm_stats.json file.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    s3_path = f"{s3_bucket}/{repo_id}"

    print(f"[INFO] Uploading dataset from {dataset_path} to {s3_path}")

    upload_cmd = f"aws s3 sync {dataset_path} {s3_path} --exclude 'dp_dataset/*' --exclude 'jpg/*'"
    run_command(upload_cmd)

    print("[INFO] Verifying upload...")
    verify_cmd = f"aws s3 ls '{s3_path}/' --recursive | head -10"
    result = run_command(verify_cmd, capture_output=True)

    if not result.stdout.strip():
        raise RuntimeError(f"Upload verification failed - no files found at {s3_path}")

    print(f"[OK] Dataset successfully uploaded to {s3_path}")

    norm_stats_dir = Path(norm_stats_dir)
    if not norm_stats_dir.exists():
        raise FileNotFoundError(f"Norm stats directory does not exist: {norm_stats_dir}")

    upload_cmd = f"aws s3 sync '{norm_stats_dir}' '{s3_path}/norm_stats' --delete"
    run_command(upload_cmd)

    print("[INFO] Verifying norm stats upload...")
    verify_cmd = f"aws s3 ls '{s3_path}/norm_stats' --recursive | head -10"
    result = run_command(verify_cmd, capture_output=True)

    if not result.stdout.strip():
        raise RuntimeError(f"Upload verification failed - no files found at {s3_path}/norm_stats")

    return s3_path


def _build_setup_script() -> str:
    """Build the cloud-agnostic setup script.

    Installs FFmpeg 7 dev libraries required by both av>=14 (PyAV, needs
    FFmpeg 7 headers to compile from sdist) and torchcodec (runtime linking).
    Also installs awscli system-wide so ``aws`` is available before the
    venv is activated in the run script.
    """
    lines = [
        'echo "[SETUP] Setting up openpi"',
        'conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; true',
        '',
        '# System dependencies',
        'apt-get update && apt-get install -y git curl pkg-config software-properties-common awscli',
        '',
        '# Install FFmpeg 7 dev libs (required by av>=14 sdist build and torchcodec)',
        '# Ubuntu default repos only have FFmpeg 6; use the PPA for FFmpeg 7.',
        'add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7',
        'apt-get update',
        'apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev',
        '',
        '# Install uv and set up Python environment',
        'curl -LsSf https://astral.sh/uv/install.sh | sh',
        'source $HOME/.local/bin/env 2>/dev/null || true',
        'uv venv --python 3.11',
        'source .venv/bin/activate',
        'GIT_LFS_SKIP_SMUDGE=1 uv sync',
        'GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .',
        'echo "[SETUP] Complete"',
    ]
    return '\n'.join(lines)


def _build_run_script() -> str:
    """Build the run script with SkyPilot metadata logging and wandb tagging.

    Cloud-agnostic: the AWS Deep Learning AMI activation line is always
    included but uses ``|| true`` so it's a silent no-op on other providers.
    """
    lines = [
        'echo "############################"',
        'echo "## Starting training job  ##"',
        'echo "############################"',
        '',
        '# Log SkyPilot instance metadata',
        'echo "[INFO] Task ID: $SKYPILOT_TASK_ID"',
        'echo "[INFO] Cluster info: $SKYPILOT_CLUSTER_INFO"',
        'echo "[INFO] Node rank: $SKYPILOT_NODE_RANK / $SKYPILOT_NUM_NODES"',
        'echo "[INFO] GPUs per node: $SKYPILOT_NUM_GPUS_PER_NODE"',
        '',
        '# Activate environment first so aws/uv are on PATH',
        'conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; true',
        'source .venv/bin/activate',
        '# AWS Deep Learning AMI PyTorch env (silent no-op on other clouds)',
        'source /opt/pytorch/bin/activate 2>/dev/null || true',
    ]

    lines += [
        '',
        '# Sync dataset from S3',
        'echo "[INFO] Syncing dataset from $DATASET_PATH"',
        'mkdir -p ~/.cache/huggingface/lerobot/',
        'aws s3 sync "$DATASET_PATH" "$HOME/.cache/huggingface/lerobot/${REPO_ID}"',
        '',
        '# Sync norm stats from S3',
        'echo "[INFO] Syncing norm stats"',
        'aws s3 sync "$DATASET_PATH/norm_stats" "$HOME/sky_workdir/assets/$CONFIG_NAME/${REPO_ID}"',
        '',
        '# Export SkyPilot metadata for wandb tagging',
        'export SKYPILOT_CLUSTER_NAME=$(echo $SKYPILOT_CLUSTER_INFO | python3 -c "import sys,json; print(json.load(sys.stdin)[\'cluster_name\'])" 2>/dev/null || echo "unknown")',
        'export WANDB_TAGS="skypilot,$SKYPILOT_CLUSTER_NAME"',
        'export WANDB_NOTES="task_id=$SKYPILOT_TASK_ID cluster=$SKYPILOT_CLUSTER_NAME"',
        '',
        '# Run training',
        'echo "[INFO] Running training: $CONFIG_NAME exp=$EXP_NAME"',
        'export XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION',
        '${WANDB_MODE_ARG}uv run scripts/train.py $CONFIG_NAME \\',
        '  --exp-name=$EXP_NAME \\',
        '  --overwrite \\',
        '  --checkpoint_base_dir $CHECKPOINT_BASE_DIR \\',
        '  --s3_checkpoint_path $S3_CHECKPOINT_PATH',
        '',
        '# Final sync after training completes',
        'echo "[INFO] Final checkpoint sync to S3"',
        'aws s3 sync ./tmp_checkpoints "$S3_CHECKPOINT_PATH"',
        'echo "[OK] Training complete"',
    ]
    return '\n'.join(lines)


def generate_sky_config(
    providers: List[str],
    accelerators: List[str],
    dataset_s3_path: str,
    config_name: str,
    exp_name: str,
    repo_id: str,
    s3_checkpoint_base: str,
    wandb_api_key: Optional[str] = None,
    provider_regions: Optional[dict[str, str]] = None,
    aws_image_ids: Optional[dict[str, str]] = None,
    idle_minutes: int = 10,
    xla_mem_fraction: float = 0.95,
    managed: bool = True,
) -> dict:
    """Generate a single SkyPilot YAML with multi-cloud auto-failover.

    Instead of generating separate configs per provider and racing them,
    this uses SkyPilot's native ``resources.any_of`` to let the scheduler
    automatically fail over across clouds, regions, and GPU types.

    Args:
        providers: Cloud providers to try, e.g. ["aws", "lambda"].
        accelerators: GPU specs in preference order, e.g. ["H200:8", "H100:8", "A100-80GB:8"].
        provider_regions: Optional map pinning providers to a region,
            e.g. {"aws": "us-west-2"}. Unpinned providers let SkyPilot
            choose the cheapest available region.
        aws_image_ids: Map of AWS region to AMI ID for the Deep Learning AMI.
        managed: If True, config is for ``sky jobs launch`` (auto-teardown).
    """
    provider_regions = provider_regions or {}
    checkpoint_path = f"{s3_checkpoint_base}/{config_name}/{exp_name}"
    wandb_mode_arg = "" if wandb_api_key else "WANDB_MODE=disabled "

    secrets = {}
    if wandb_api_key:
        secrets['WANDB_API_KEY'] = wandb_api_key

    # Build one resource candidate per provider. SkyPilot's any_of will
    # try each and auto-failover across regions within each cloud.
    candidates = []
    for provider in providers:
        region = provider_regions.get(provider)
        infra = f"{provider}/{region}" if region else provider

        entry = {
            'infra': infra,
            'accelerators': accelerators,
        }

        if provider == 'aws' and aws_image_ids:
            if region and region in aws_image_ids:
                entry['image_id'] = aws_image_ids[region]
            elif not region:
                entry['image_id'] = aws_image_ids

        candidates.append(entry)

    if len(candidates) == 1:
        resources = candidates[0]
    else:
        resources = {'any_of': candidates}

    config = {
        'workdir': '.',
        'envs': {
            'DATASET_PATH': dataset_s3_path,
            'CONFIG_NAME': config_name,
            'REPO_ID': repo_id,
            'EXP_NAME': exp_name,
            'CHECKPOINT_BASE_DIR': './tmp_checkpoints',
            'S3_CHECKPOINT_PATH': checkpoint_path,
            'WANDB_MODE_ARG': wandb_mode_arg,
            'XLA_MEM_FRACTION': str(xla_mem_fraction),
        },
        'resources': resources,
        'num_nodes': 1,
        'setup': _build_setup_script(),
        'run': _build_run_script(),
    }

    if secrets:
        config['secrets'] = secrets

    return config


def _get_latest_job_id() -> Optional[str]:
    """Get the job ID of the most recently submitted managed job."""
    try:
        result = subprocess.run(
            "sky jobs queue | tail -n +2 | head -1 | awk '{print $1}'",
            shell=True, capture_output=True, text=True, check=True,
        )
        job_id = result.stdout.strip()
        return job_id if job_id.isdigit() else None
    except subprocess.CalledProcessError:
        return None


def launch_training(config_file: Path, cluster_name: Optional[str] = None, managed: bool = False):
    """Launch the training job using SkyPilot.

    Args:
        config_file: Path to the SkyPilot YAML config.
        cluster_name: Optional cluster name.
        managed: If True, use ``sky jobs launch`` for guaranteed auto-teardown
                 on completion. Recommended to prevent leaked GPU instances.
    """
    if managed:
        print(f"[INFO] Launching managed job with config: {config_file}")
        launch_cmd = f"sky jobs launch '{config_file}' --yes"
        if cluster_name:
            launch_cmd += f" -n {cluster_name}"
    else:
        print(f"[INFO] Launching cluster job with config: {config_file}")
        launch_cmd = f"sky launch '{config_file}' --retry-until-up --yes"
        if cluster_name:
            launch_cmd += f" --cluster-name {cluster_name}"

    run_command(launch_cmd)
    print("[OK] Training job launched successfully!")

    # For managed jobs, tail the job logs so they stream to this terminal.
    # This blocks until the job finishes (Ctrl+C detaches without cancelling).
    if managed:
        job_id = _get_latest_job_id()
        if job_id:
            print(f"[INFO] Streaming logs for managed job {job_id} (Ctrl+C to detach)...")
            # Use check=False because sky jobs logs returns non-zero if the
            # job fails, but we still want to see the output.
            run_command(f"sky jobs logs {job_id}", check=False)


