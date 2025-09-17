# Finetune `pi0` for I2RT YAMs

## Step 1: Clone and Set Up the Repository

```bash
git clone git@github.com:xdofai/openpi.git
cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Step 3: Convert Data (You may have to write your own depending on source format)
- **TODO BY USER**: Open `scripts/yam_data/convert_yam_data.py` and update:
  - `yam_data_path`
  - `repo_name`
  - `language_instruction`
  - `fps`

```bash
uv run scripts/yam_data/convert_yam_data.py
```

## Step 4: Compute Normalization Statistics
- **TODO BY USER**: Edit `src/openpi/training/config.py` to add your own `TrainConfig`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats_jax.py --config-name pi0_yam_low_mem_finetune --epsilon 1e-2
```

## Step 5: Launch Training
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_yam_low_mem_finetune --exp-name=<task_name> --overwrite
```
