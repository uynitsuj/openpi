#!/usr/bin/env python3
"""
Utility functions for OpenPI SkyPilot training launcher.
"""

import subprocess
import pandas as pd
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


def parse_sky_gpu_output(output: str) -> pd.DataFrame:
    """Parse tabular output from sky show-gpus into a pandas DataFrame."""
    lines = output.strip().split('\n')

    data_rows = []
    current_headers = None
    header_positions = None

    for line in lines:
        if not line.strip():
            continue

        if line.startswith('GPU'):
            current_headers = []
            header_positions = []

            words = line.split()
            search_pos = 0
            for word in words:
                pos = line.find(word, search_pos)
                current_headers.append(word)
                header_positions.append(pos)
                search_pos = pos + len(word)

        elif current_headers and header_positions and not line.startswith('-') and len(line.strip()) > 0:
            row_data = {}

            for i, header in enumerate(current_headers):
                if i < len(header_positions):
                    start_pos = header_positions[i]

                    if i + 1 < len(header_positions):
                        end_pos = header_positions[i + 1]
                        value = line[start_pos:end_pos].strip()
                    else:
                        value = line[start_pos:].strip()

                    if 'PRICE' in header and value.startswith('$'):
                        value = value.replace('$', '').replace(',', '').strip()

                    if value == '-':
                        value = None

                    row_data[header] = value

            if row_data:
                data_rows.append(row_data)

    return pd.DataFrame(data_rows)


def query_sky_accelerators(accelerators: str, region: str, service_providers: List[str]) -> dict:
    """Query SkyPilot for the cheapest service provider for the requested accelerators."""
    print(f"[INFO] Checking available providers: {service_providers}")

    query_cmd = f"sky check {' '.join(service_providers)}"
    result = run_command(query_cmd, capture_output=True)

    available_providers = []
    for provider in service_providers:
        if f"{provider}: enabled" in result.stdout.lower():
            available_providers.append(provider)

    if not available_providers:
        raise RuntimeError(f"No providers available from {service_providers}")

    print(f"[OK] Available providers: {available_providers}")

    all_dataframes = []
    for provider in available_providers:
        print(f"[INFO] Querying {provider} for {accelerators}...")
        query_cmd = f"sky show-gpus {accelerators} --infra {provider}/{region}"
        try:
            result = run_command(query_cmd, capture_output=True)
            if result.stdout.strip():
                provider_df = parse_sky_gpu_output(result.stdout)
                if not provider_df.empty:
                    all_dataframes.append(provider_df)
                    print(f"[OK] Found {len(provider_df)} options from {provider}/{region}")
                else:
                    print(f"[WARN] No options found from {provider}/{region}")
            else:
                print(f"[WARN] No output from {provider}/{region}")
        except Exception as e:
            print(f"[ERROR] Error querying {provider}/{region}: {e}")

    if all_dataframes:
        df = pd.concat(all_dataframes, ignore_index=True)
    else:
        df = pd.DataFrame()

    if df.empty:
        raise RuntimeError(f"No GPU options found for {accelerators}")

    if region:
        df_filtered = df[df['REGION'] == region]
        if df_filtered.empty:
            print(f"[WARN] No options found in region {region}, showing all regions")
            df_filtered = df
    else:
        df_filtered = df

    df_filtered = df_filtered.copy()
    if 'HOURLY_SPOT_PRICE' in df_filtered.columns:
        df_filtered['HOURLY_SPOT_PRICE'] = pd.to_numeric(df_filtered['HOURLY_SPOT_PRICE'], errors='coerce')
        cheapest_idx = df_filtered['HOURLY_SPOT_PRICE'].idxmin()
        cheapest_option = df_filtered.loc[cheapest_idx].to_dict()
        print(f"[INFO] Cheapest (spot): {cheapest_option['CLOUD']} in {cheapest_option['REGION']} at ${cheapest_option['HOURLY_SPOT_PRICE']}/hour")
    else:
        df_filtered['HOURLY_PRICE'] = pd.to_numeric(df_filtered['HOURLY_PRICE'], errors='coerce')
        cheapest_idx = df_filtered['HOURLY_PRICE'].idxmin()
        cheapest_option = df_filtered.loc[cheapest_idx].to_dict()
        print(f"[INFO] Cheapest (on-demand): {cheapest_option['CLOUD']} in {cheapest_option['REGION']} at ${cheapest_option['HOURLY_PRICE']}/hour")

    return {
        'cheapest_option': cheapest_option,
        'all_options': list(df_filtered.to_dict('records')),
        'dataframe': df_filtered,
    }


def upload_dataset_to_s3(dataset_path: Path, s3_bucket: str, repo_id: str, norm_stats_path: str) -> str:
    """Upload dataset to S3 and return the full S3 path."""
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

    upload_cmd = f"aws s3 sync '{norm_stats_path}/{repo_id}' '{s3_path}/norm_stats' --delete"
    run_command(upload_cmd)

    print("[INFO] Verifying norm stats upload...")
    verify_cmd = f"aws s3 ls '{s3_path}/norm_stats' --recursive | head -10"
    result = run_command(verify_cmd, capture_output=True)

    if not result.stdout.strip():
        raise RuntimeError(f"Upload verification failed - no files found at {s3_path}/norm_stats")

    return s3_path


def _build_setup_script() -> str:
    """Build the cloud-agnostic setup script."""
    lines = [
        'echo "[SETUP] Setting up openpi"',
        'conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; true',
        'apt-get update && apt-get install -y git curl',
        'curl -LsSf https://astral.sh/uv/install.sh | sh',
        'source $HOME/.local/bin/env 2>/dev/null || true',
        'uv venv --python 3.11',
        'source .venv/bin/activate',
        'GIT_LFS_SKIP_SMUDGE=1 uv sync',
        'GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .',
        'echo "[SETUP] Complete"',
    ]
    return '\n'.join(lines)


def _build_run_script(cloud: str) -> str:
    """Build the run script with SkyPilot metadata logging and wandb tagging.

    The ``source /opt/pytorch/bin/activate`` line is only included for AWS,
    where it is provided by the AWS Deep Learning AMI. Lambda and other
    providers do not ship this path and the command would fail.
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
        '# Activate environment',
        'conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; true',
        'source .venv/bin/activate',
    ]

    # AWS Deep Learning AMI exposes a PyTorch env at /opt/pytorch/bin/activate.
    # Lambda (and other providers) do not have this path, so we skip it.
    if cloud == 'aws':
        lines.append('source /opt/pytorch/bin/activate 2>/dev/null || true')

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
    cloud: str,
    dataset_s3_path: str,
    config_name: str,
    exp_name: str,
    repo_id: str,
    s3_checkpoint_base: str,
    wandb_api_key: Optional[str] = None,
    accelerators: str = "A100:8",
    region: str = "us-west-2",
    image_id: Optional[str] = None,
    idle_minutes: int = 10,
    xla_mem_fraction: float = 0.95,
) -> dict:
    """Generate SkyPilot YAML configuration for any supported cloud provider.

    Key differences from per-provider functions:
    - Single function handles aws, lambda, gcp, etc.
    - Uses ``secrets`` for WANDB_API_KEY (redacted in dashboard/logs).
    - Autostop configured with ``wait_for: jobs`` and a checkpoint-sync hook.
    - ``source /opt/pytorch/bin/activate`` only included for AWS.
    - XLA_PYTHON_CLIENT_MEM_FRACTION properly exported (not on a bare line).
    """
    checkpoint_path = f"{s3_checkpoint_base}/{config_name}/{exp_name}"
    wandb_mode_arg = "" if wandb_api_key else "WANDB_MODE=disabled "

    # Secrets block keeps sensitive values redacted in `sky status` / dashboard
    secrets = {}
    if wandb_api_key:
        secrets['WANDB_API_KEY'] = wandb_api_key

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
        'resources': {
            'cloud': cloud,
            'accelerators': accelerators,
            'region': region,
            'autostop': {
                'idle_minutes': idle_minutes,
                'down': True,
                'wait_for': 'jobs',
                'hook': (
                    'echo "[HOOK] Syncing final checkpoints to S3..." && '
                    f'aws s3 sync ./tmp_checkpoints {checkpoint_path} && '
                    'echo "[HOOK] Done."'
                ),
                'hook_timeout': 600,
            },
        },
        'num_nodes': 1,
        'setup': _build_setup_script(),
        'run': _build_run_script(cloud),
    }

    if secrets:
        config['secrets'] = secrets

    if cloud == 'aws' and image_id:
        config['resources']['image_id'] = image_id

    return config


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
