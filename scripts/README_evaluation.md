# Model Evaluation Script

This evaluation script allows you to load a trained OpenPI model and compare ground-truth vs predicted action chunks to debug model training performance.

## Features

- **Model Loading**: Loads trained models from checkpoints with proper EMA parameter handling
- **Batch Evaluation**: Evaluates on multiple batches of data for robust statistics
- **Comprehensive Metrics**: Computes MSE, MAE, RMSE, and R² scores
- **Rich Visualizations**: 
  - Action trajectory comparisons (ground-truth vs predicted)
  - Error heatmaps across time steps and action dimensions
  - Error distribution histograms and box plots
  - Summary metrics bar charts
- **Robot-Specific Action Names**: Automatically detects robot type and uses meaningful action names
- **Detailed Results**: Saves all results to JSON for further analysis

## Usage

### Basic Usage

```bash
cd openpi
python scripts/evaluate.py --config-name pi0_yam --num-eval-batches 20
```

### Advanced Usage

```bash
# Evaluate specific checkpoint step
python scripts/evaluate.py --config-name pi0_yam --checkpoint-step 15000 --num-eval-batches 50

# Evaluate XMI RBY model
python scripts/evaluate.py --config-name pi0_xmi_rby --num-eval-batches 30

# Evaluate LoRA fine-tuned model
python scripts/evaluate.py --config-name pi0_yam_low_mem_finetune --num-eval-batches 10
```

## Available Configurations

### YAM (Your Autonomous Manipulator) Configs
- `pi0_yam`: Full fine-tuning on YAM bimanual dataset (14D actions)
- `pi0_yam_low_mem_finetune`: LoRA fine-tuning for memory efficiency

### XMI RBY Configs  
- `pi0_xmi_rby`: Full fine-tuning on XMI RBY bimanual dataset (20D actions)
- `pi0_xmi_rby_low_mem_finetune`: LoRA fine-tuning version
- `pi0_fast_xmi_rby`: Using PI0-FAST architecture

### ALOHA Configs
- `pi0_aloha`: Inference config for ALOHA robot
- `pi0_aloha_pen_uncap`: Fine-tuned for pen uncapping task
- `pi0_aloha_sim`: Simulated ALOHA environment

### Libero Configs
- `pi0_libero`: Full fine-tuning on Libero dataset
- `pi0_libero_low_mem_finetune`: LoRA fine-tuning version
- `pi0_fast_libero`: Using PI0-FAST architecture
- `pi0_fast_libero_low_mem_finetune`: PI0-FAST with LoRA

### Debug Configs
- `debug`: Minimal config for testing

## Output Files

The script creates an `evaluation/` directory in your checkpoint folder with:

### Visualization Files
- `action_comparison.png`: Line plots comparing ground-truth vs predicted actions
- `error_heatmap.png`: Heatmap showing errors across time steps and action dimensions  
- `error_distribution.png`: Histogram and box plots of prediction errors
- `summary_metrics.png`: Bar chart of evaluation metrics with error bars

### Data Files
- `evaluation_results.json`: Complete results including all metrics and configuration

## Action Dimension Mapping

The script automatically detects the robot type and uses meaningful action names:

### YAM (14 dimensions)
```
L_Joint_0, L_Joint_1, L_Joint_2, L_Joint_3, L_Joint_4, L_Joint_5, L_Gripper,
R_Joint_0, R_Joint_1, R_Joint_2, R_Joint_3, R_Joint_4, R_Joint_5, R_Gripper
```

### XMI RBY (20 dimensions)  
```
L_Rot_0, L_Rot_1, L_Rot_2, L_Rot_3, L_Rot_4, L_Rot_5,
L_Pos_X, L_Pos_Y, L_Pos_Z, L_Gripper,
R_Rot_0, R_Rot_1, R_Rot_2, R_Rot_3, R_Rot_4, R_Rot_5,
R_Pos_X, R_Pos_Y, R_Pos_Z, R_Gripper
```

### ALOHA (14 dimensions)
```
L_Joint_0, L_Joint_1, L_Joint_2, L_Joint_3, L_Joint_4, L_Joint_5, L_Gripper,
R_Joint_0, R_Joint_1, R_Joint_2, R_Joint_3, R_Joint_4, R_Joint_5, R_Gripper
```

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Average of squared differences between predictions and ground truth
- **MAE (Mean Absolute Error)**: Average of absolute differences  
- **RMSE (Root Mean Squared Error)**: Square root of MSE, same units as original data
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model (1.0 = perfect)

## Interpreting Results

### Good Performance Indicators
- **Low MSE/MAE/RMSE**: Predictions are close to ground truth
- **High R²** (close to 1.0): Model explains most of the variance
- **Consistent errors across action dimensions**: No systematic bias
- **Errors centered around 0**: No systematic over/under-prediction

### Potential Issues to Look For
- **High errors in specific action dimensions**: May indicate insufficient training data or model capacity
- **Increasing errors over time steps**: Potential issue with action chunk prediction
- **Bimodal error distributions**: May indicate model uncertainty or conflicting training examples
- **Very low R²**: Model may not be learning meaningful patterns

## Troubleshooting

### Common Issues

1. **"No checkpoints found"**: Ensure your training has saved at least one checkpoint
2. **"Config not found"**: Use `--config-name` that matches exactly with available configs
3. **Memory issues**: Reduce `--num-eval-batches` or use a model with smaller batch size
4. **Missing dependencies**: Install required packages: `matplotlib`, `seaborn`, `scikit-learn`

### Checkpoint Location
Checkpoints are expected at: `/home/justinyu/checkpoints/{config_name}/{exp_name}/`

If your checkpoints are elsewhere, modify the `checkpoint_base_dir` in the config or create a custom config.

## Example Workflow

1. **Train a model**:
   ```bash
   python scripts/train.py pi0_yam --exp-name my_yam_experiment
   ```

2. **Evaluate the trained model**:
   ```bash  
   python scripts/evaluate.py --config-name pi0_yam --num-eval-batches 25
   ```

3. **Check results**:
   - Look at plots in `checkpoints/pi0_yam/my_yam_experiment/evaluation/`
   - Review metrics in the console output
   - Analyze detailed results in `evaluation_results.json`

4. **Iterate**:
   - If errors are high, consider more training steps, different learning rates, or data augmentation
   - If certain action dimensions perform poorly, investigate data quality or model architecture
   - If errors increase over time steps, consider adjusting action horizon or model capacity

## Tips for Better Evaluation

- **Use enough batches** (20-50) for reliable statistics
- **Evaluate on different checkpoint steps** to see training progression  
- **Compare different model architectures** (PI0 vs PI0-FAST)
- **Check both full and LoRA fine-tuning** results
- **Look at both aggregate metrics and per-dimension performance** 