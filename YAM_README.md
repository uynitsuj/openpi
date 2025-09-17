# Finetune `pi0`

## Step 1: Clone and Set Up the Repository

```bash
git clone git@github.com:xdofai/openpi.git
cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Step 2: Check Video Info

```bash
python scripts/sim_video_info_check.py --yam_data_path <dpath_to_dataset>
```

- **TODO BY USER**: Note down the `video_mean_fps` from the output.

## Step 3: Convert Simulation Data
- **TODO BY USER**: Open `scripts/yam_data/convert_yam_data.py` and update:
  - `yam_data_path`
  - `repo_name`
  - `language_instruction`
  - `fps` (use the `video_mean_fps`)

```bash
python scripts/yam_data/convert_yam_data.py
```

## Step 4: Compute Normalization Statistics
- **TODO BY USER**: Edit `src/openpi/training/config.py` to add your own `TrainConfig`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name <your_training_configuration> --epsilon 1e-2
```

## Step 5: Launch Training
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <your_training_configuration> --exp-name=<task_name> --overwrite
```

# Setting Up AWS Training
## Step 1: Dataset preperation
- **TODO BY USER**: Edit `src/openpi/training/config.py` to add your own `TrainConfig`.
```bash
# input the env variable for your specific situation
sky launch sky/prep_data.yaml
```

## Step 2: Compute Normalization Statistics
- **TODO BY USER**: Edit `src/openpi/training/config.py` to add your own `TrainConfig`.
```bash
# input the env variable for your specific situation
sky launch sky/train_pi0.yaml
```

## minimal working example

aws bucket: `s3://xdof-internal-research/`
upload to the bucket (already done): `aws s3 sync /nfs_us/justinyu/xmi_lerobot_datasets/xmi_rby_coffee_cup_on_dish_subsampled_and_gripper_action_cleaned s3://xdof-internal-research/xmi_lerobot_datasets/xmi_rby_coffee_cup_on_dish_subsampled_and_gripper_action_cleaned`
confirm the data is there: `aws s3 ls s3://xdof-internal-research/xmi_lerobot_datasets/xmi_rby_coffee_cup_on_dish_subsampled_and_gripper_action_cleaned/`
launch the training: `sky launch sky/train_pi0.yaml`
