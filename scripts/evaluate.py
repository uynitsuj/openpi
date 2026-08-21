"""Evaluate a trained OpenPI model on its training dataset.

Loads a checkpoint, runs inference on evaluation batches, and produces
action-prediction metrics and visualizations.

Usage:
    uv run python scripts/evaluate.py pi0_yam_tshirt_sarm_rabc \
        --exp-name sarm_rabc_dense_progress_20260510 \
        --checkpoint-base-dir /home/kavishk/checkpoints \
        --num-eval-batches 50
"""

import dataclasses
import json
import logging
import pathlib
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_logging():
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
    checkpoint_step: int | None = None,
) -> tuple[_model.BaseModel, training_utils.TrainState]:
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise ValueError(f"No checkpoints found in {config.checkpoint_dir}")

    rng = jax.random.key(config.seed)
    init_rng, _ = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
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

    train_state_shape = jax.eval_shape(init, init_rng)
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, checkpoint_step)

    params_to_use = train_state.ema_params if train_state.ema_params is not None else train_state.params
    model = nnx.merge(train_state.model_def, params_to_use)
    model.eval()

    return model, train_state


@at.typecheck
def evaluate_batch(
    model: _model.BaseModel,
    observation: _model.Observation,
    actions_gt: _model.Actions,
    rng: at.KeyArrayLike,
) -> tuple[np.ndarray, dict[str, float]]:
    actions_pred = model.sample_actions(rng, observation)

    gt = np.asarray(jax.device_get(actions_gt))
    pred = np.asarray(jax.device_get(actions_pred))

    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)

    metrics = {
        "mse": float(mean_squared_error(gt_flat, pred_flat)),
        "mae": float(mean_absolute_error(gt_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(gt_flat, pred_flat))),
    }
    denom = float(np.sum((gt_flat - np.mean(gt_flat)) ** 2))
    metrics["r2"] = float(1 - np.sum((gt_flat - pred_flat) ** 2) / denom) if denom > 0 else 0.0

    return pred, metrics


def plot_action_comparison(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
    sample_indices: list[int] | None = None,
):
    batch_size, action_horizon, action_dim = actions_gt.shape

    if action_names is None:
        action_names = [f"Action_{i}" for i in range(action_dim)]

    if sample_indices is None:
        sample_indices = list(np.random.choice(batch_size, min(4, batch_size), replace=False))

    n_samples = len(sample_indices)
    n_dims = min(action_dim, 14)
    fig, axes = plt.subplots(n_samples, n_dims, figsize=(3 * n_dims, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample_idx in enumerate(sample_indices):
        for j in range(n_dims):
            ax = axes[i, j]
            ts = np.arange(action_horizon)
            ax.plot(ts, actions_gt[sample_idx, :, j], "b-", label="GT", linewidth=1.5)
            ax.plot(ts, actions_pred[sample_idx, :, j], "r--", label="Pred", linewidth=1.5)
            ax.set_title(f"S{sample_idx} {action_names[j]}", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_heatmap(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
):
    _, action_horizon, action_dim = actions_gt.shape

    if action_names is None:
        action_names = [f"A{i}" for i in range(action_dim)]

    avg_errors = np.mean(np.abs(actions_gt - actions_pred), axis=0)

    plt.figure(figsize=(max(8, action_horizon // 2), max(6, action_dim // 2)))
    sns.heatmap(
        avg_errors.T,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=5,
        yticklabels=action_names,
        cbar_kws={"label": "Mean Absolute Error"},
    )
    plt.title("Prediction Error Heatmap (averaged across batch)")
    plt.xlabel("Time Step")
    plt.ylabel("Action Dimension")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
):
    errors = (actions_pred - actions_gt).flatten()
    action_dim = actions_gt.shape[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(errors, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Prediction Error")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Error Distribution")
    ax1.grid(True, alpha=0.3)

    abs_errors_by_dim = [
        np.abs(actions_pred[:, :, i] - actions_gt[:, :, i]).flatten()
        for i in range(action_dim)
    ]
    ax2.boxplot(abs_errors_by_dim, positions=range(action_dim))
    ax2.set_xlabel("Action Dimension")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Error by Action Dimension")
    ax2.set_xticks(range(action_dim))
    ax2.set_xticklabels([f"A{i}" for i in range(action_dim)], fontsize=7, rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_metrics(
    all_metrics: list[dict[str, float]],
    save_path: str,
) -> dict[str, dict[str, float]]:
    metric_names = list(all_metrics[0].keys())
    aggregated = {}
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(metric_names))
    means = [aggregated[m]["mean"] for m in metric_names]
    stds = [aggregated[m]["std"] for m in metric_names]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color="skyblue", edgecolor="black")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std,
            f"{mean:.4f}±{std:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Value")
    ax.set_title("Evaluation Metrics Summary")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return aggregated


YAM_ACTION_NAMES = [
    "L_J0", "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_Grip",
    "R_J0", "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_Grip",
]


def main(config: _config.TrainConfig, num_eval_batches: int = 10, checkpoint_step: int | None = None):
    init_logging()
    logging.info(f"Running evaluation on: {platform.node()}")
    logging.info(f"Config: {config.name}, exp: {config.exp_name}")

    eval_output_dir = pathlib.Path(str(config.checkpoint_dir)) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading trained model...")
    model, train_state = load_trained_model(config, checkpoint_step)
    logging.info(f"Loaded checkpoint at step {train_state.step}")

    logging.info("Creating data loader...")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=num_eval_batches,
    )

    rng = jax.random.key(config.seed + 1)
    all_metrics: list[dict[str, float]] = []
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    logging.info(f"Evaluating on {num_eval_batches} batches...")
    data_iter = iter(data_loader)
    for batch_idx in tqdm.tqdm(range(num_eval_batches), desc="Evaluating"):
        try:
            observation, actions_gt = next(data_iter)
        except StopIteration:
            logging.warning(f"Data exhausted after {batch_idx} batches")
            break

        eval_rng = jax.random.fold_in(rng, batch_idx)
        pred, metrics = evaluate_batch(model, observation, actions_gt, eval_rng)

        all_metrics.append(metrics)
        all_gt.append(np.asarray(jax.device_get(actions_gt)))
        all_pred.append(pred)
        del observation, actions_gt
        jax.clear_caches()

        if batch_idx % 10 == 0:
            logging.info(f"Batch {batch_idx}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    all_gt_arr = np.concatenate(all_gt, axis=0)
    all_pred_arr = np.concatenate(all_pred, axis=0)
    logging.info(f"Total samples: {all_gt_arr.shape[0]}, action shape: {all_gt_arr.shape}")

    action_names = None
    repo_id = getattr(config.data, "repo_id", "") or ""
    if "yam" in repo_id.lower() or "sarm" in repo_id.lower() or "tshirt" in repo_id.lower():
        action_names = YAM_ACTION_NAMES

    logging.info("Generating plots...")

    plot_action_comparison(
        all_gt_arr, all_pred_arr,
        str(eval_output_dir / "action_comparison.png"),
        action_names=action_names,
    )

    plot_error_heatmap(
        all_gt_arr, all_pred_arr,
        str(eval_output_dir / "error_heatmap.png"),
        action_names=action_names,
    )

    plot_error_distribution(
        all_gt_arr, all_pred_arr,
        str(eval_output_dir / "error_distribution.png"),
    )

    summary = plot_summary_metrics(all_metrics, str(eval_output_dir / "summary_metrics.png"))

    results = {
        "summary_metrics": summary,
        "per_batch_metrics": all_metrics,
        "evaluation_info": {
            "config_name": config.name,
            "exp_name": config.exp_name,
            "num_batches": len(all_metrics),
            "total_samples": int(all_gt_arr.shape[0]),
            "action_shape": list(all_gt_arr.shape),
            "checkpoint_step": int(train_state.step),
        },
    }
    with open(eval_output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    for name, data in summary.items():
        logging.info(f"  {name.upper():6s}: {data['mean']:.6f} ± {data['std']:.6f}")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {eval_output_dir}")


if __name__ == "__main__":
    import sys

    num_eval_batches = 50
    checkpoint_step = None
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--num-eval-batches" and i + 1 < len(sys.argv):
            num_eval_batches = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--checkpoint-step" and i + 1 < len(sys.argv):
            checkpoint_step = int(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = filtered_argv

    config = _config.cli()
    main(config, num_eval_batches, checkpoint_step)
