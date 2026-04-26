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

    upload_cmd = f"aws s3 sync {dataset_path} {s3_path} --exclude 'dp_dataset/*' --exclude 'jpg/*' --exclude 'norm_stats/*' --delete"
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
        '# Use sudo when available but still work on root-runner images.',
        'if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi',
        'export DEBIAN_FRONTEND=noninteractive',
        '',
        '# System dependencies',
        '$SUDO apt-get update && $SUDO apt-get install -y git curl pkg-config software-properties-common awscli',
        '',
        '# Install FFmpeg runtime/dev libs for torchcodec. Prefer FFmpeg 7, but',
        '# fall back to the distro packages if the PPA is unavailable.',
        'if $SUDO add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7; then',
        '  $SUDO apt-get update',
        'else',
        '  echo "[SETUP] FFmpeg 7 PPA unavailable; falling back to distro FFmpeg packages"',
        'fi',
        '$SUDO apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev',
        '$SUDO ldconfig || true',
        'ffmpeg -version',
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
        '# Resume mode: pull existing checkpoints from S3 into the local checkpoint dir',
        '# BEFORE training so train.py can pick up from where it stopped.',
        'LOCAL_CKPT_DIR="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"',
        'if [ "$RESUME" = "true" ]; then',
        '  echo "[INFO] Resume mode: syncing prior checkpoints from $S3_CHECKPOINT_PATH"',
        '  mkdir -p "$LOCAL_CKPT_DIR"',
        '  aws s3 sync "$S3_CHECKPOINT_PATH" "$LOCAL_CKPT_DIR"',
        '  RESUME_ARG="--resume --no-overwrite"',
        'else',
        '  RESUME_ARG="--overwrite"',
        'fi',
        '',
        '# Optional CLI overrides forwarded to train.py only when set.',
        'EXTRA_ARGS=""',
        'if [ -n "$NUM_TRAIN_STEPS_OVERRIDE" ]; then EXTRA_ARGS="$EXTRA_ARGS --num-train-steps=$NUM_TRAIN_STEPS_OVERRIDE"; fi',
        'if [ -n "$SAVE_INTERVAL_OVERRIDE" ]; then EXTRA_ARGS="$EXTRA_ARGS --save-interval=$SAVE_INTERVAL_OVERRIDE"; fi',
        'if [ -n "$KEEP_PERIOD_OVERRIDE" ]; then EXTRA_ARGS="$EXTRA_ARGS --keep-period=$KEEP_PERIOD_OVERRIDE"; fi',
        '',
        '# Run training',
        'echo "[INFO] Running training: $CONFIG_NAME exp=$EXP_NAME (resume=$RESUME, extras=$EXTRA_ARGS)"',
        'export XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION',
        '${WANDB_MODE_ARG}uv run scripts/train.py $CONFIG_NAME \\',
        '  --exp-name=$EXP_NAME \\',
        '  $RESUME_ARG \\',
        '  $EXTRA_ARGS \\',
        '  --checkpoint_base_dir $CHECKPOINT_BASE_DIR \\',
        '  --s3_checkpoint_path $S3_CHECKPOINT_PATH',
        'TRAIN_EXIT=$?',
        '',
        '# Final sync after training (also runs on crash, since no set -e).',
        '# IMPORTANT: source must be the per-experiment dir, NOT ./tmp_checkpoints,',
        '# otherwise S3 ends up with a nested <exp>/<config>/<exp>/... duplicate.',
        'echo "[INFO] Final checkpoint sync to S3 (train exit=$TRAIN_EXIT)"',
        'aws s3 sync "$LOCAL_CKPT_DIR" "$S3_CHECKPOINT_PATH"',
        'echo "[OK] Training complete (exit $TRAIN_EXIT)"',
        'exit $TRAIN_EXIT',
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
    disk_size: int = 256,
    resume: bool = False,
    num_train_steps_override: Optional[int] = None,
    save_interval_override: Optional[int] = None,
    keep_period_override: Optional[int] = None,
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

    # Build one resource candidate per (provider, accelerator) pair.
    # Within any_of entries, accelerators must be a single string.
    # SkyPilot picks the cheapest available and auto-fails-over.
    candidates = []
    for provider in providers:
        region = provider_regions.get(provider)
        infra = f"{provider}/{region}" if region else provider

        for accel in accelerators:
            entry = {
                'infra': infra,
                'accelerators': accel,
                'disk_size': disk_size,
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
        'workdir': '/home/justinyu/openpi',
        'envs': {
            'DATASET_PATH': dataset_s3_path,
            'CONFIG_NAME': config_name,
            'REPO_ID': repo_id,
            'EXP_NAME': exp_name,
            'CHECKPOINT_BASE_DIR': './tmp_checkpoints',
            'S3_CHECKPOINT_PATH': checkpoint_path,
            'WANDB_MODE_ARG': wandb_mode_arg,
            'XLA_MEM_FRACTION': str(xla_mem_fraction),
            'RESUME': 'true' if resume else 'false',
            'NUM_TRAIN_STEPS_OVERRIDE': str(num_train_steps_override) if num_train_steps_override is not None else '',
            'SAVE_INTERVAL_OVERRIDE': str(save_interval_override) if save_interval_override is not None else '',
            'KEEP_PERIOD_OVERRIDE': str(keep_period_override) if keep_period_override is not None else '',
        },
        'resources': resources,
        'num_nodes': 1,
        'setup': _build_setup_script(),
        'run': _build_run_script(),
    }

    if secrets:
        config['secrets'] = secrets

    return config


def _parse_job_id_from_output(output: str) -> Optional[str]:
    """Extract the managed job ID from ``sky jobs launch`` output.

    Looks for patterns like ``sky jobs logs 219`` or ``Job ID: 219`` in the
    output to reliably capture the ID without a separate queue query.
    """
    import re
    # sky jobs launch prints hints like: "sky jobs logs <id>"
    match = re.search(r'sky jobs logs\s+(?:--controller\s+)?(\d+)', output)
    if match:
        return match.group(1)
    # Fallback: "Job ID: <id>" or "Managed job ID: <id>"
    match = re.search(r'[Jj]ob ID:\s*(\d+)', output)
    if match:
        return match.group(1)
    return None


def launch_training(config_file: Path, cluster_name: Optional[str] = None, managed: bool = False):
    """Launch the training job using SkyPilot.

    For managed jobs, streams the launch output (provisioning/setup) to the
    terminal, parses the job ID, then tails the actual task logs so the user
    sees training output without having to run ``sky jobs logs`` manually.
    """
    if managed:
        print(f"[INFO] Launching managed job with config: {config_file}")
        # --detach-run: return as soon as the job is submitted to the controller
        # instead of streaming worker logs. Lets batch launchers advance
        # through multiple configs without blocking on the actual training run.
        launch_cmd = f"sky jobs launch '{config_file}' --yes --detach-run"
        if cluster_name:
            launch_cmd += f" -n {cluster_name}"
    else:
        print(f"[INFO] Launching cluster job with config: {config_file}")
        launch_cmd = f"sky launch '{config_file}' --retry-until-up --yes"
        if cluster_name:
            launch_cmd += f" --cluster-name {cluster_name}"

    if not managed:
        run_command(launch_cmd)
        print("[OK] Training job launched successfully!")
        return

    # For managed jobs: tee the launch output to the terminal while
    # capturing it so we can parse the job ID for log tailing.
    print(f"[RUN] {launch_cmd}")
    captured_lines = []
    proc = subprocess.Popen(
        launch_cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
        captured_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, launch_cmd)

    print("[OK] Training job launched successfully!")

    # Parse job ID from the launch output. Skip log tailing so batch
    # launches can advance through multiple configs without blocking.
    output = ''.join(captured_lines)
    job_id = _parse_job_id_from_output(output)
    if job_id:
        print(f"[INFO] Submitted job {job_id}. Tail with: sky jobs logs {job_id}")
    else:
        print("[WARN] Could not parse job ID from launch output. "
              "Run 'sky jobs queue' to find it.")

