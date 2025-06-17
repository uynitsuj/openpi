import dataclasses
import functools
import logging
import platform
from typing import Any
import os

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def load_trained_model(
    config: _config.TrainConfig, 
    checkpoint_step: int | None = None
) -> tuple[_model.BaseModel, training_utils.TrainState]:
    """Load a trained model from checkpoints."""
    
    # Initialize checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    
    if not resuming:
        raise ValueError(f"No checkpoints found in {config.checkpoint_dir}")
    
    # Initialize model architecture using the same method as training
    rng = jax.random.key(config.seed)
    init_rng, _ = jax.random.split(rng)
    
    mesh = sharding.make_mesh(config.fsdp_devices)
    
    # Use the same initialization logic as training
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # Create train state structure for loading
    train_state_shape = jax.eval_shape(init, init_rng)
    
    # Load from checkpoint
    train_state = _checkpoints.restore_state(
        checkpoint_manager, 
        train_state_shape, 
        None,  # We don't need data loader for this
        checkpoint_step
    )
    
    # Reconstruct model with loaded parameters
    # Use EMA params if available, otherwise use regular params
    params_to_use = train_state.ema_params if train_state.ema_params is not None else train_state.params
    model = nnx.merge(train_state.model_def, params_to_use)
    model.eval()  # Set to evaluation mode
    
    return model, train_state


@at.typecheck
def evaluate_model(
    model: _model.BaseModel,
    observation: _model.Observation,
    actions_gt: _model.Actions,
    rng: at.KeyArrayLike,
) -> tuple[_model.Actions, dict[str, float]]:
    """Evaluate model on a single batch and compute metrics."""
    
    # Get model predictions
    actions_pred = model.sample_actions(rng, observation)
    
    # Compute metrics
    actions_gt_np = np.array(actions_gt)
    actions_pred_np = np.array(actions_pred)
    
    # Flatten for metric computation
    gt_flat = actions_gt_np.reshape(-1)
    pred_flat = actions_pred_np.reshape(-1)
    
    metrics = {
        "mse": float(mean_squared_error(gt_flat, pred_flat)),
        "mae": float(mean_absolute_error(gt_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(gt_flat, pred_flat))),
        "r2": float(1 - np.sum((gt_flat - pred_flat) ** 2) / np.sum((gt_flat - np.mean(gt_flat)) ** 2)),
    }
    
    return actions_pred, metrics


def plot_action_comparison(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
    sample_indices: list[int] | None = None,
):
    """Create comprehensive plots comparing ground truth vs predicted actions."""
    
    batch_size, action_horizon, action_dim = actions_gt.shape
    
    if action_names is None:
        action_names = [f"Action_{i}" for i in range(action_dim)]
    
    if sample_indices is None:
        # Select a few random samples to plot
        sample_indices = np.random.choice(batch_size, min(4, batch_size), replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(sample_indices), action_dim, figsize=(4*action_dim, 3*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    if action_dim == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample_idx in enumerate(sample_indices):
        for j in range(action_dim):
            ax = axes[i, j]
            
            # Plot ground truth and predicted actions
            time_steps = np.arange(action_horizon)
            ax.plot(time_steps, actions_gt[sample_idx, :, j], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(time_steps, actions_pred[sample_idx, :, j], 'r--', label='Predicted', linewidth=2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action Value')
            ax.set_title(f'Sample {sample_idx} - {action_names[j]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_heatmap(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
):
    """Create heatmap of prediction errors across time and action dimensions."""
    
    batch_size, action_horizon, action_dim = actions_gt.shape
    
    if action_names is None:
        action_names = [f"Action_{i}" for i in range(action_dim)]
    
    # Compute absolute errors
    errors = np.abs(actions_gt - actions_pred)
    
    # Average errors across batch
    avg_errors = np.mean(errors, axis=0)  # Shape: (action_horizon, action_dim)
    
    # Create heatmap
    plt.figure(figsize=(max(8, action_dim), max(6, action_horizon // 2)))
    sns.heatmap(
        avg_errors.T,  # Transpose so actions are on y-axis
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f'T{i}' for i in range(action_horizon)],
        yticklabels=action_names,
        cbar_kws={'label': 'Mean Absolute Error'}
    )
    plt.title('Prediction Error Heatmap\n(Averaged across batch)')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
):
    """Plot distribution of prediction errors."""
    
    # Compute errors
    errors = (actions_pred - actions_gt).flatten()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of errors
    ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot of absolute errors by action dimension
    action_dim = actions_gt.shape[-1]
    abs_errors_by_dim = []
    for i in range(action_dim):
        abs_errors_by_dim.append(np.abs(actions_pred[:, :, i] - actions_gt[:, :, i]).flatten())
    
    ax2.boxplot(abs_errors_by_dim, positions=range(action_dim))
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Distribution by Action Dimension')
    ax2.set_xticks(range(action_dim))
    ax2.set_xticklabels([f'A{i}' for i in range(action_dim)])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_metrics_plot(
    all_metrics: list[dict[str, float]],
    save_path: str,
):
    """Create a summary plot of all evaluation metrics."""
    
    # Aggregate metrics
    metric_names = list(all_metrics[0].keys())
    aggregated = {}
    
    for metric in metric_names:
        values = [batch_metrics[metric] for batch_metrics in all_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(metric_names))
    means = [aggregated[metric]['mean'] for metric in metric_names]
    stds = [aggregated[metric]['std'] for metric in metric_names]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Model Evaluation Metrics Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return aggregated


def main(config: _config.TrainConfig, num_eval_batches: int = 10, checkpoint_step: int | None = None):
    """Main evaluation function."""
    
    init_logging()
    logging.info(f"Running evaluation on: {platform.node()}")
    
    # Create output directory
    eval_output_dir = config.checkpoint_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    logging.info("Loading trained model...")
    model, train_state = load_trained_model(config, checkpoint_step)
    logging.info(f"Loaded model from step {train_state.step}")
    
    # Create data loader
    logging.info("Creating data loader...")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=False,  # Don't shuffle for evaluation
        num_batches=num_eval_batches,
    )
    
    # Setup evaluation
    rng = jax.random.key(config.seed)
    all_metrics = []
    all_actions_gt = []
    all_actions_pred = []
    
    # Run evaluation
    logging.info(f"Running evaluation on {num_eval_batches} batches...")
    data_iter = iter(data_loader)
    
    for batch_idx in tqdm.tqdm(range(num_eval_batches), desc="Evaluating"):
        # Get batch
        observation, actions_gt = next(data_iter)
        
        # Run inference
        eval_rng = jax.random.fold_in(rng, batch_idx)
        actions_pred, metrics = evaluate_model(model, observation, actions_gt, eval_rng)
        
        # Store results
        all_metrics.append(metrics)
        all_actions_gt.append(np.array(actions_gt))
        all_actions_pred.append(np.array(actions_pred))
        
        # Log progress
        if batch_idx % 5 == 0:
            logging.info(f"Batch {batch_idx}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Concatenate all results
    all_actions_gt = np.concatenate(all_actions_gt, axis=0)
    all_actions_pred = np.concatenate(all_actions_pred, axis=0)
    
    logging.info(f"Evaluation complete. Total samples: {all_actions_gt.shape[0]}")
    logging.info(f"Action shape: {all_actions_gt.shape}")
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Determine action names based on the dataset
    action_names = None
    data_config = config.data.create(config.assets_dirs, config.model)
    if hasattr(config.data, 'repo_id') and config.data.repo_id:
        if 'yam' in config.data.repo_id.lower():
            action_names = [
                'L_Joint_0', 'L_Joint_1', 'L_Joint_2', 'L_Joint_3', 'L_Joint_4', 'L_Joint_5', 'L_Gripper',
                'R_Joint_0', 'R_Joint_1', 'R_Joint_2', 'R_Joint_3', 'R_Joint_4', 'R_Joint_5', 'R_Gripper'
            ]
        elif 'xmi' in config.data.repo_id.lower():
            action_names = [
                'L_Rot_0', 'L_Rot_1', 'L_Rot_2', 'L_Rot_3', 'L_Rot_4', 'L_Rot_5',
                'L_Pos_X', 'L_Pos_Y', 'L_Pos_Z', 'L_Gripper',
                'R_Rot_0', 'R_Rot_1', 'R_Rot_2', 'R_Rot_3', 'R_Rot_4', 'R_Rot_5',
                'R_Pos_X', 'R_Pos_Y', 'R_Pos_Z', 'R_Gripper'
            ]
        elif 'aloha' in config.data.repo_id.lower():
            action_names = [
                'L_Joint_0', 'L_Joint_1', 'L_Joint_2', 'L_Joint_3', 'L_Joint_4', 'L_Joint_5', 'L_Gripper',
                'R_Joint_0', 'R_Joint_1', 'R_Joint_2', 'R_Joint_3', 'R_Joint_4', 'R_Joint_5', 'R_Gripper'
            ]
    
    # Action comparison plots
    plot_action_comparison(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "action_comparison.png"),
        action_names=action_names,
    )
    
    # Error heatmap
    plot_error_heatmap(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_heatmap.png"),
        action_names=action_names,
    )
    
    # Error distribution
    plot_error_distribution(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_distribution.png"),
    )
    
    # Summary metrics
    summary_metrics = create_summary_metrics_plot(
        all_metrics,
        str(eval_output_dir / "summary_metrics.png"),
    )
    
    # Save detailed results
    results = {
        'summary_metrics': summary_metrics,
        'all_metrics': all_metrics,
        'config': dataclasses.asdict(config),
        'evaluation_info': {
            'num_batches': num_eval_batches,
            'total_samples': all_actions_gt.shape[0],
            'action_shape': all_actions_gt.shape,
            'checkpoint_step': int(train_state.step),
        }
    }
    
    import json
    with open(eval_output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    for metric_name, metric_data in summary_metrics.items():
        logging.info(f"{metric_name.upper()}: {metric_data['mean']:.6f} ± {metric_data['std']:.6f}")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {eval_output_dir}")
    logging.info("Generated plots:")
    logging.info(f"  - Action comparison: {eval_output_dir / 'action_comparison.png'}")
    logging.info(f"  - Error heatmap: {eval_output_dir / 'error_heatmap.png'}")
    logging.info(f"  - Error distribution: {eval_output_dir / 'error_distribution.png'}")
    logging.info(f"  - Summary metrics: {eval_output_dir / 'summary_metrics.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenPI model")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the training config to use")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (required if not set in config)")
    parser.add_argument("--num-eval-batches", type=int, default=10, help="Number of batches to evaluate on")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Specific checkpoint step to load (default: latest)")
    
    args = parser.parse_args()
    
    # Load config
    config = _config.get_config(args.config_name)
    
    # Set experiment name if provided
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    
    # Run evaluation
    main(config, args.num_eval_batches, args.checkpoint_step) 