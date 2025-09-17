#!/usr/bin/env python3
"""
Utility functions for OpenPI SkyPilot training launcher.
"""

import json
import subprocess
import pandas as pd
import re
from pathlib import Path
from typing import Optional, List

def run_command(cmd: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with proper error handling."""
    print(f"🔄 Running: \n{cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    if capture_output:
        print(f"✅ Output: \n{result.stdout.strip()}")
    return result


def check_prerequisites():
    """Check if required tools are available."""
    required_tools = ['aws', 'sky']
    
    for tool in required_tools:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, check=True)
            print(f"✅ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {tool} is not available. Please install it first.")
            return False
    
    # Check AWS credentials
    try:
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], capture_output=True, check=True)
        print("✅ AWS credentials are configured")
    except subprocess.CalledProcessError:
        print("❌ AWS credentials not configured. Run 'aws configure' first.")
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
            
        # Check if this is a header line
        if line.startswith('GPU'):
            current_headers = []
            header_positions = []
            
            # Find header positions by looking for column starts
            words = line.split()
            search_pos = 0
            for word in words:
                pos = line.find(word, search_pos)
                current_headers.append(word)
                header_positions.append(pos)
                search_pos = pos + len(word)
            
        elif current_headers and header_positions and not line.startswith('-') and len(line.strip()) > 0:
            # This is a data row - parse using header positions
            row_data = {}
            
            for i, header in enumerate(current_headers):
                if i < len(header_positions):
                    start_pos = header_positions[i]
                    
                    # Find end position (start of next column or end of line)
                    if i + 1 < len(header_positions):
                        end_pos = header_positions[i + 1]
                        value = line[start_pos:end_pos].strip()
                    else:
                        value = line[start_pos:].strip()
                    
                    # Clean up price values
                    if 'PRICE' in header and value.startswith('$'):
                        value = value.replace('$', '').replace(',', '').strip()
                    
                    # Handle empty values
                    if value == '-':
                        value = None
                    
                    row_data[header] = value
            
            if row_data:  # Only add if we got some data
                data_rows.append(row_data)
    
    return pd.DataFrame(data_rows)


def query_sky_accelerators(accelerators: str, region: str, service_providers: List[str]) -> dict:
    """Query SkyPilot for the cheapest service provider for the requested accelerators."""
    print(f"🔍 Checking available providers: {service_providers}")
    
    # Check which providers are available
    query_cmd = f"sky check {' '.join(service_providers)}"
    result = run_command(query_cmd, capture_output=True)

    available_providers = []
    for provider in service_providers:
        if f"{provider}: enabled" in result.stdout.lower():
            available_providers.append(provider)
    
    if not available_providers:
        raise RuntimeError(f"No providers available from {service_providers}")
    
    print(f"✅ Available providers: {available_providers}")
    
    # Query GPU availability for each provider individually
    all_dataframes = []
    for provider in available_providers:
        print(f"🔍 Querying {provider} for {accelerators}...")
        query_cmd = f"sky show-gpus {accelerators} --infra {provider}/{region}"
        try:
            result = run_command(query_cmd, capture_output=True)
            if result.stdout.strip():
                provider_df = parse_sky_gpu_output(result.stdout)
                if not provider_df.empty:
                    all_dataframes.append(provider_df)
                    print(f"✅ Found {len(provider_df)} options from {provider}/{region}")
                else:
                    print(f"⚠️  No options found from {provider}/{region}")
            else:
                print(f"⚠️  No output from {provider}/{region}")
        except Exception as e:
            print(f"❌ Error querying {provider}/{region}: {e}")
    
    # Combine all dataframes
    if all_dataframes:
        df = pd.concat(all_dataframes, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    if df.empty:
        raise RuntimeError(f"No GPU options found for {accelerators}")
    
    # Filter by region if specified
    if region:
        df_filtered = df[df['REGION'] == region]
        if df_filtered.empty:
            print(f"⚠️  No options found in region {region}, showing all regions")
            df_filtered = df
    else:
        df_filtered = df
    
    # Convert price columns to numeric for comparison
    df_filtered = df_filtered.copy()
    if 'HOURLY_SPOT_PRICE' in df_filtered.columns:
        df_filtered['HOURLY_SPOT_PRICE'] = pd.to_numeric(df_filtered['HOURLY_SPOT_PRICE'], errors='coerce')
        # Find cheapest option by spot price
        spot_prices = df_filtered['HOURLY_SPOT_PRICE']
        cheapest_idx = int(spot_prices.idxmin())
        cheapest_option = df_filtered.iloc[cheapest_idx].to_dict()
        print(f"💰 Cheapest option by spot price: {cheapest_option['CLOUD']} in {cheapest_option['REGION']} at ${cheapest_option['HOURLY_SPOT_PRICE']}/hour")
    else:
        # Fallback to regular price
        df_filtered['HOURLY_PRICE'] = pd.to_numeric(df_filtered['HOURLY_PRICE'], errors='coerce') 
        regular_prices = df_filtered['HOURLY_PRICE']
        cheapest_idx = int(regular_prices.idxmin())
        cheapest_option = df_filtered.iloc[cheapest_idx].to_dict()
        print(f"💰 Cheapest option by regular price: {cheapest_option['CLOUD']} in {cheapest_option['REGION']} at ${cheapest_option['HOURLY_PRICE']}/hour")
    
    return {
        'cheapest_option': cheapest_option,
        'all_options': list(df_filtered.to_dict('records')),
        'dataframe': df_filtered
    }

def upload_dataset_to_s3(dataset_path: Path, s3_bucket: str, repo_id: str, norm_stats_path: str) -> str:
    """Upload dataset to S3 and return the full S3 path."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    s3_path = f"{s3_bucket}/{repo_id}"
    
    print(f"📤 Uploading dataset from {dataset_path} to {s3_path}")
    
    # Upload to S3
    upload_cmd = f"aws s3 sync {dataset_path} {s3_path} --exclude 'dp_dataset/*' --exclude 'jpg/*'" # --delete"
    run_command(upload_cmd)
    
    # Verify upload
    print(f"🔍 Verifying upload...")
    verify_cmd = f"aws s3 ls '{s3_path}/' --recursive | head -10"
    result = run_command(verify_cmd, capture_output=True)
    
    if not result.stdout.strip():
        raise RuntimeError(f"Upload verification failed - no files found at {s3_path}")
    
    print(f"✅ Dataset successfully uploaded to {s3_path}")
    # Upload norm stats to S3
    upload_cmd = f"aws s3 sync '{norm_stats_path}/{repo_id}' '{s3_path}/norm_stats' --delete"
    run_command(upload_cmd)

    # Verify upload
    print(f"🔍 Verifying norm stats upload...")
    verify_cmd = f"aws s3 ls '{s3_path}/norm_stats' --recursive | head -10"
    result = run_command(verify_cmd, capture_output=True)
    
    if not result.stdout.strip():
        raise RuntimeError(f"Upload verification failed - no files found at {s3_path}/norm_stats")

    return s3_path


def generate_sky_config_aws(
    dataset_s3_path: str,
    config_name: str,
    exp_name: str,
    repo_id: str,
    s3_checkpoint_base: str,
    wandb_api_key: Optional[str] = None,
    accelerators: str = "A100:8",
    region: str = "us-west-2",
    # image_id: str = "ami-067cc81f948e50e06", # for us-west-2
    image_id: str = "ami-0365bff494b18bf93", # for us-east-1
) -> dict:
    """Generate SkyPilot YAML configuration."""
    
    checkpoint_path = f"{s3_checkpoint_base}/{config_name}/{exp_name}"

    wandb_api_key_arg = f"{wandb_api_key}" if wandb_api_key else ""
    wandb_mode_arg = "" if wandb_api_key else "WANDB_MODE=disabled "
    
    config = {
        'workdir': '.',
        'envs': {
            'DATASET_PATH': dataset_s3_path,
            'CONFIG_NAME': config_name,
            'REPO_ID': repo_id,
            'EXP_NAME': exp_name,
            'CHECKPOINT_BASE_DIR': './tmp_checkpoints',
            'S3_CHECKPOINT_PATH': checkpoint_path,
            'WANDB_API_KEY': wandb_api_key_arg,
            'WANDB_MODE_ARG': wandb_mode_arg
        },
        'resources': {
            'cloud': 'aws',
            'accelerators': accelerators,
            'region': region,
            'image_id': image_id,
            'autostop': {'down': True}
        },
        'num_nodes': 1,
        'setup': """echo "Setting up openpi"
conda deactivate
conda deactivate
apt-get update && apt-get install git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo "Setup complete!\"""",
        'run': f"""echo "######################"
echo "## Starting job ... ##"
echo "######################"

mkdir -p ~/.cache/huggingface/lerobot/
echo "Syncing dataset to S3"
echo "aws s3 sync "$DATASET_PATH" "/home/ubuntu/.cache/huggingface/lerobot/${{REPO_ID}}""
aws s3 sync "$DATASET_PATH" "/home/ubuntu/.cache/huggingface/lerobot/${{REPO_ID}}"
conda deactivate
conda deactivate
source .venv/bin/activate
source /opt/pytorch/bin/activate
echo "Syncing norm stats to S3"
echo "aws s3 sync "$DATASET_PATH"/norm_stats "/home/ubuntu/sky_workdir/assets/$CONFIG_NAME/${{REPO_ID}}""
aws s3 sync "$DATASET_PATH"/norm_stats "/home/ubuntu/sky_workdir/assets/$CONFIG_NAME/${{REPO_ID}}"
echo "Running training script with command: "
echo "${{WANDB_MODE_ARG}}XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME --overwrite --checkpoint_base_dir $CHECKPOINT_BASE_DIR --s3_checkpoint_path $S3_CHECKPOINT_PATH"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 
${{WANDB_MODE_ARG}}uv run scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME --overwrite --checkpoint_base_dir $CHECKPOINT_BASE_DIR --s3_checkpoint_path $S3_CHECKPOINT_PATH"""
    }
    
    return config



def generate_sky_config_lambda(
    dataset_s3_path: str,
    config_name: str,
    exp_name: str,
    repo_id: str,
    s3_checkpoint_base: str,
    wandb_api_key: Optional[str] = None,
    accelerators: str = "A100:8",
    region: str = "us-west-2",
) -> dict:
    """Generate SkyPilot YAML configuration."""
    
    checkpoint_path = f"{s3_checkpoint_base}/{config_name}/{exp_name}"

    wandb_api_key_arg = f"{wandb_api_key}" if wandb_api_key else ""
    wandb_mode_arg = "" if wandb_api_key else "WANDB_MODE=disabled "
    
    config = {
        'workdir': '.',
        'envs': {
            'DATASET_PATH': dataset_s3_path,
            'CONFIG_NAME': config_name,
            'REPO_ID': repo_id,
            'EXP_NAME': exp_name,
            'CHECKPOINT_BASE_DIR': './tmp_checkpoints',
            'S3_CHECKPOINT_PATH': checkpoint_path,
            'WANDB_API_KEY': wandb_api_key_arg,
            'WANDB_MODE_ARG': wandb_mode_arg
        },
        'resources': {
            'cloud': 'lambda',
            'accelerators': accelerators,
            'region': region,
            'autostop': {'down': True}
        },
        'num_nodes': 1,
        'setup': """echo "Setting up openpi"
conda deactivate
conda deactivate
apt-get update && apt-get install git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo "Setup complete!\"""",
        'run': f"""echo "######################"
echo "## Starting job ... ##"
echo "######################"

mkdir -p ~/.cache/huggingface/lerobot/
echo "Syncing dataset to S3"
echo "aws s3 sync "$DATASET_PATH" "/home/ubuntu/.cache/huggingface/lerobot/${{REPO_ID}}""
aws s3 sync "$DATASET_PATH" "/home/ubuntu/.cache/huggingface/lerobot/${{REPO_ID}}"
conda deactivate
conda deactivate
source .venv/bin/activate
source /opt/pytorch/bin/activate
echo "Syncing norm stats to S3"
echo "aws s3 sync "$DATASET_PATH"/norm_stats "/home/ubuntu/sky_workdir/assets/$CONFIG_NAME/${{REPO_ID}}""
aws s3 sync "$DATASET_PATH"/norm_stats "/home/ubuntu/sky_workdir/assets/$CONFIG_NAME/${{REPO_ID}}"
echo "Running training script with command: "
echo "${{WANDB_MODE_ARG}}XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME --overwrite --checkpoint_base_dir $CHECKPOINT_BASE_DIR --s3_checkpoint_path $S3_CHECKPOINT_PATH"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 
${{WANDB_MODE_ARG}}uv run scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME --overwrite --checkpoint_base_dir $CHECKPOINT_BASE_DIR --s3_checkpoint_path $S3_CHECKPOINT_PATH"""
    }
    
    return config

def launch_training(config_file: Path, cluster_name: Optional[str] = None):
    """Launch the training job using SkyPilot."""
    print(f"🚀 Launching training job with config: {config_file}")
    
    launch_cmd = f"sky launch '{config_file}' --retry-until-up"
    if cluster_name:
        launch_cmd += f" --cluster-name {cluster_name}"
    
    # Add interactive flag to handle user prompts
    launch_cmd += " --yes"
    
    run_command(launch_cmd)
    print("✅ Training job launched successfully!") 