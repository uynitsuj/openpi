"""
World Model Training Script

This script provides the training infrastructure for world models, specifically
adapted for VJEPA-2 style video understanding and prediction tasks.
"""

import dataclasses
import functools
import logging
import platform
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import common_utils
import tqdm_loggable.auto as tqdm
import wandb

from openpi.models.world_model import WorldModelInput, WorldModelOutput
from openpi.models.vjepa2_world_model import VJEPA2WorldModel, create_vjepa2_model
from openpi.shared import array_typing as at
from openpi.training import optimizer as _optimizer
from openpi.training import checkpoints as _checkpoints
from openpi.training.world_model_training.config import WorldModelTrainConfig
from openpi.training.world_model_training.data_loader import create_world_model_data_loader
from openpi.training import sharding

logger = logging.getLogger("openpi")


@dataclass
class WorldModelTrainState:
    """Training state for world models."""
    
    step: int
    model: VJEPA2WorldModel
    params: Any
    optimizer: optax.OptState
    tx: optax.GradientTransformation
    ema_model: VJEPA2WorldModel | None = None
    ema_decay: float | None = None
    progressive_masking_schedule: Any = None
    target_encoder_momentum: float = 0.99


def init_logging():
    """Initialize logging configuration."""
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


def init_wandb(config: WorldModelTrainConfig, resuming: bool = False):
    """Initialize Weights & Biases logging."""
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resuming:
        run_id_file = ckpt_dir / "wandb_id.txt"
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            logger.warning("Resume requested but no wandb ID found. Starting new run.")
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def init_train_state(config: WorldModelTrainConfig, init_rng: jax.Array) -> WorldModelTrainState:
    """Initialize the training state."""
    logger.info("Initializing world model...")
    
    model = create_vjepa2_model(config.model_config)
    
    tx = _optimizer.create_optimizer(
        config.optimizer,
        config.lr_schedule,
        weight_decay_mask=None,
    )
    
    dummy_input = WorldModelInput(
        video_frames=jnp.zeros((
            config.batch_size,
            config.data_config.num_frames,
            config.data_config.image_size[0],
            config.data_config.image_size[1] * len(config.data_config.image_keys),
            3
        )),
        mask=jnp.zeros((
            config.batch_size,
            config.data_config.num_frames,
            config.data_config.image_size[0] // 16,
            config.data_config.image_size[1] * len(config.data_config.image_keys) // 16,
        ), dtype=bool),
        camera_names=list(config.data_config.image_keys),
    )
    
    import torch
    
    torch_params = dict(model.named_parameters())
    
    def torch_to_jax(tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    
    params = {name: torch_to_jax(param) for name, param in torch_params.items()}
    
    opt_state = tx.init(params)
    
    ema_model = None
    if hasattr(config, 'ema_decay') and config.ema_decay is not None:
        ema_model = create_vjepa2_model(config.model_config)
    
    progressive_masking_schedule = None
    if config.data_config.use_progressive_masking:
        from openpi.training.world_model_training.data_loader import ProgressiveMaskingSchedule
        progressive_masking_schedule = ProgressiveMaskingSchedule(total_steps=config.num_train_steps)
    
    return WorldModelTrainState(
        step=0,
        model=model,
        params=params,
        optimizer=opt_state,
        tx=tx,
        ema_model=ema_model,
        ema_decay=getattr(config, 'ema_decay', None),
        progressive_masking_schedule=progressive_masking_schedule,
        target_encoder_momentum=config.target_encoder_momentum,
    )


def compute_loss(
    model: VJEPA2WorldModel,
    params: Any,
    batch: Tuple[WorldModelInput, WorldModelOutput],
    rng: jax.Array,
    train: bool = True,
) -> Tuple[jax.Array, dict]:
    """Compute the training loss."""
    model_input, model_output = batch
    
    import torch
    
    def jax_to_torch(arr):
        return torch.from_numpy(np.array(arr)).float()
    
    def update_model_params(model, jax_params):
        for name, param in model.named_parameters():
            if name in jax_params:
                param.data = jax_to_torch(jax_params[name])
    
    model.train(train)
    
    video_frames = jax_to_torch(model_input.video_frames)
    mask = jax_to_torch(model_input.mask).bool()
    
    with torch.enable_grad() if train else torch.no_grad():
        loss = model.compute_loss(video_frames, mask)
    
    total_loss_jax = jnp.array(loss.detach().cpu().numpy())
    
    metrics = {
        'reconstruction_loss': total_loss_jax,
        'mask_ratio': mask.float().mean(),
        'num_masked_patches': mask.float().sum(),
    }
    
    return total_loss_jax, metrics


def train_step(
    state: WorldModelTrainState,
    batch: Tuple[WorldModelInput, WorldModelOutput],
    rng: jax.Array,
) -> Tuple[WorldModelTrainState, dict]:
    """Execute a single training step using PyTorch native training."""
    
    import torch
    
    def jax_to_torch(arr):
        return torch.from_numpy(np.array(arr)).float()
    
    def update_model_params(model, jax_params):
        for name, param in model.named_parameters():
            if name in jax_params:
                param.data = jax_to_torch(jax_params[name])
    
    update_model_params(state.model, state.params)
    
    state.model.train()
    
    video_frames = jax_to_torch(batch[0].video_frames)
    mask = jax_to_torch(batch[0].mask).bool()
    
    state.model.zero_grad()
    
    with torch.cuda.amp.autocast():
        loss = state.model.compute_loss(video_frames, mask)
    
    if not hasattr(state, 'torch_optimizer'):
        state.torch_optimizer = torch.optim.AdamW(
            state.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        state.grad_scaler = torch.cuda.amp.GradScaler()
    
    state.grad_scaler.scale(loss).backward()
    
    state.grad_scaler.step(state.torch_optimizer)
    state.grad_scaler.update()
    
    if hasattr(state.model, 'update_target_encoder'):
        momentum = getattr(state, 'target_encoder_momentum', 0.99)
        state.model.update_target_encoder(momentum)
    
    grad_norm = 0.0
    for param in state.model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    def torch_to_jax(tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    
    new_params = {}
    for name, param in state.model.named_parameters():
        new_params[name] = torch_to_jax(param.data)
    
    param_norm = 0.0
    for param in new_params.values():
        param_norm += jnp.sum(param ** 2)
    param_norm = jnp.sqrt(param_norm)
    
    new_ema_model = state.ema_model
    if state.ema_model is not None and state.ema_decay is not None:
        pass
    
    new_state = WorldModelTrainState(
        step=state.step + 1,
        model=state.model,
        params=new_params,
        optimizer=state.optimizer,
        tx=state.tx,
        ema_model=new_ema_model,
        ema_decay=state.ema_decay,
        target_encoder_momentum=state.target_encoder_momentum,
    )
    
    metrics = {
        'reconstruction_loss': jnp.array(loss.detach().cpu().numpy()),
        'mask_ratio': jnp.array(mask.float().mean().cpu().numpy()),
        'num_masked_patches': jnp.array(mask.float().sum().cpu().numpy()),
        'grad_norm': jnp.array(grad_norm),
        'param_norm': jnp.array(param_norm),
    }
    
    if hasattr(state, 'progressive_masking_schedule') and state.progressive_masking_schedule is not None:
        masking_params = state.progressive_masking_schedule.get_masking_params(state.step)
        metrics.update({
            'masking_progress': masking_params['phase_progress'],
            'target_mask_ratio': masking_params['mask_ratio'],
            'target_num_masked_patches': masking_params['num_masked_patches'],
        })
        metrics['_masking_phase'] = masking_params['phase']
    
    return new_state, metrics


def validation_step(
    state: WorldModelTrainState,
    batch: Tuple[WorldModelInput, WorldModelOutput],
    rng: jax.Array,
) -> dict:
    """Execute a validation step."""
    model = state.model
    
    import torch
    
    def jax_to_torch(arr):
        return torch.from_numpy(np.array(arr)).float()
    
    def update_model_params(model, jax_params):
        for name, param in model.named_parameters():
            if name in jax_params:
                param.data = jax_to_torch(jax_params[name])
    
    update_model_params(model, state.params)
    
    model.eval()
    
    video_frames = jax_to_torch(batch[0].video_frames)
    mask = jax_to_torch(batch[0].mask).bool()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            loss = model.compute_loss(video_frames, mask)
    
    total_loss_jax = jnp.array(loss.detach().cpu().numpy())
    
    metrics = {
        'reconstruction_loss': total_loss_jax,
        'mask_ratio': jnp.array(mask.float().mean().cpu().numpy()),
        'num_masked_patches': jnp.array(mask.float().sum().cpu().numpy()),
    }
    
    return {f"val_{k}": v for k, v in metrics.items()}


def save_checkpoint(
    state: WorldModelTrainState,
    config: WorldModelTrainConfig,
    step: int,
):
    """Save a checkpoint."""
    checkpoint_dir = config.checkpoint_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    params_path = checkpoint_dir / "model_params.pkl"
    with open(params_path, 'wb') as f:
        import pickle
        pickle.dump(state.model.state_dict(), f)
    
    if state.ema_model is not None and hasattr(state.ema_model, 'state_dict'):
        ema_params_path = checkpoint_dir / "ema_params.pkl"
        with open(ema_params_path, 'wb') as f:
            pickle.dump(state.ema_model.state_dict(), f)
    
    opt_state_path = checkpoint_dir / "optimizer_state.pkl"
    with open(opt_state_path, 'wb') as f:
        pickle.dump(state.optimizer, f)
    
    step_info_path = checkpoint_dir / "step_info.txt"
    with open(step_info_path, 'w') as f:
        f.write(f"step: {step}\n")
        f.write(f"config: {config.name}\n")
    
    logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def main(config: WorldModelTrainConfig):
    """Main training loop."""
    init_logging()
    logger.info(f"Starting world model training on {platform.node()}")
    logger.info(f"Configuration: {config.name}")
    
    rng = jax.random.PRNGKey(config.seed)
    train_rng, init_rng, data_rng = jax.random.split(rng, 3)
    
    state = init_train_state(config, init_rng)
    logger.info(f"Initialized model with {sum(jnp.size(x) for x in jax.tree_leaves(state.params)):,} parameters")
    
    train_loader = create_world_model_data_loader(
        config.data_config,
        batch_size=config.batch_size,
        split="train",
        shuffle=True,
        num_workers=config.num_workers,
        fake_data=(config.data_config.repo_id is None),
        current_step=state.step,
        chunk_size=config.data_config.chunk_size,
    )
    
    val_loader = create_world_model_data_loader(
        config.data_config,
        batch_size=config.batch_size,
        split="validation",
        shuffle=False,
        num_workers=config.num_workers,
        fake_data=(config.data_config.repo_id is None),
        current_step=state.step,
        chunk_size=config.data_config.chunk_size,
    )
    
    init_wandb(config, resuming=config.resume)
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    metrics_history = []
    
    pbar = tqdm.tqdm(
        range(config.num_train_steps),
        desc="Training",
        dynamic_ncols=True,
    )
    
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            try:
                batch = next(train_iter)
            except StopIteration:
                logger.warning(f"No data available at step {step}, skipping...")
                continue
        
        if isinstance(batch[0].video_frames, torch.Tensor):
            batch = (
                WorldModelInput(
                    video_frames=jnp.array(batch[0].video_frames.numpy()),
                    mask=jnp.array(batch[0].mask.numpy()),
                    camera_names=batch[0].camera_names,
                ),
                WorldModelOutput(
                    predicted_features=jnp.array(batch[1].predicted_features.numpy()),
                    reconstruction_loss=jnp.array(batch[1].reconstruction_loss.numpy()),
                    mask_ratio=jnp.array(batch[1].mask_ratio.numpy()),
                )
            )
        
        train_loader.update_step(step)
        
        step_rng = jax.random.fold_in(train_rng, step)
        state, metrics = train_step(state, batch, step_rng)
        
        metrics_history.append(metrics)
        
        if step % config.log_interval == 0:
            avg_metrics = {}
            for key in metrics_history[0].keys():
                if key.startswith('_'):
                    continue
                try:
                    avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in metrics_history]))
                except (TypeError, ValueError):
                    continue
            
            wandb.log(avg_metrics, step=step)
            
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items()])
            pbar.set_postfix_str(metrics_str)
            
            metrics_history = []
        
        if step % config.validation_interval == 0 and step > 0:
            val_metrics = []
            for _ in range(config.validation_steps):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        logger.warning("No validation data available, skipping validation...")
                        break
                
                if isinstance(val_batch[0].video_frames, torch.Tensor):
                    val_batch = (
                        WorldModelInput(
                            video_frames=jnp.array(val_batch[0].video_frames.numpy()),
                            mask=jnp.array(val_batch[0].mask.numpy()),
                            camera_names=val_batch[0].camera_names,
                        ),
                        WorldModelOutput(
                            predicted_features=jnp.array(val_batch[1].predicted_features.numpy()),
                            reconstruction_loss=jnp.array(val_batch[1].reconstruction_loss.numpy()),
                            mask_ratio=jnp.array(val_batch[1].mask_ratio.numpy()),
                        )
                    )
                
                val_step_rng = jax.random.fold_in(train_rng, step * 1000 + len(val_metrics))
                val_metrics.append(validation_step(state, val_batch, val_step_rng))
            
            avg_val_metrics = {}
            for key in val_metrics[0].keys():
                if key.startswith('_'):
                    continue
                try:
                    avg_val_metrics[key] = jnp.mean(jnp.array([m[key] for m in val_metrics]))
                except (TypeError, ValueError):
                    continue
            
            wandb.log(avg_val_metrics, step=step)
            logger.info(f"Validation at step {step}: {avg_val_metrics}")
        
        if step % config.save_interval == 0 and step > 0:
            save_checkpoint(state, config, step)
    
    save_checkpoint(state, config, config.num_train_steps)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    from openpi.training.world_model_training.config import create_world_model_config_cli
    
    config = create_world_model_config_cli()
    main(config) 