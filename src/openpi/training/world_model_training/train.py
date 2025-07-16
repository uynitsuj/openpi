"""
World Model Training Script

This script provides the training infrastructure for world models, specifically
adapted for VJEPA-2 style video understanding and prediction tasks.
"""

# Set multiprocessing start method immediately to avoid JAX fork warnings
import multiprocessing
import os
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Set PyTorch CUDA memory allocation for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import dataclasses
import functools
import logging
import platform
from dataclasses import dataclass
from typing import Any, Tuple
import pathlib

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
from openpi.training.world_model_training.wandb_utils import log_training_metrics, log_model_info, log_debug_visualization
from openpi.training import sharding

logger = logging.getLogger("openpi")

try:
    from openpi.training.world_model_training.optimized_data_loader import create_optimized_world_model_data_loader
    OPTIMIZED_DATALOADER_AVAILABLE = True
    logger.info("Optimized dataloader available")
except ImportError as e:
    OPTIMIZED_DATALOADER_AVAILABLE = False
    logger.warning(f"Optimized dataloader not available, using original: {e}")

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, debug visualizations will be disabled")


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
    gradient_clip_norm: float = 1.0


def save_debug_images(
    batch: Tuple[WorldModelInput, WorldModelOutput],
    step: int,
    checkpoint_dir: pathlib.Path,
    config: WorldModelTrainConfig,
):
    """Save sample images showing original vs masked frames for debugging."""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    debug_dir = checkpoint_dir / "debug_images"
    debug_dir.mkdir(exist_ok=True)
    
    model_input, model_output = batch
    
    # Convert JAX arrays to numpy
    video_frames = np.array(model_input.video_frames)
    mask = np.array(model_input.mask)
    
    # Save first sample from the batch
    sample_idx = 0
    step_dir = debug_dir / f"step_{step:06d}"
    step_dir.mkdir(exist_ok=True)
    
    if sample_idx >= video_frames.shape[0]:
        return
    
    # Get the first sample
    frames = video_frames[sample_idx]  # Shape: (T, H, W, C)
    sample_mask = mask[sample_idx]     # Shape: (T, H, W)
    
    # Create visualization
    num_frames = frames.shape[0]
    num_cameras = len(model_input.camera_names)
    
    # Create figure with subplots for each frame
    fig, axes = plt.subplots(num_frames, 2, figsize=(12, 4 * num_frames))
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    for t in range(num_frames):
        # Original frame
        original_frame = frames[t]
        if original_frame.dtype == np.float32:
            # Convert from float32 to uint8 if needed
            original_frame = (np.clip(original_frame, -1, 1) * 127.5 + 127.5).astype(np.uint8)
        
        axes[t, 0].imshow(original_frame)
        axes[t, 0].set_title(f"Original Frame {t}")
        axes[t, 0].axis('off')
        
        # Masked frame
        masked_frame = original_frame.copy()
        frame_mask = sample_mask[t]
        
        # Apply mask (set masked regions to black or gray)
        masked_frame[frame_mask] = 128  # Gray out masked regions
        
        axes[t, 1].imshow(masked_frame)
        axes[t, 1].set_title(f"Masked Frame {t} (Mask Ratio: {frame_mask.mean():.2f})")
        axes[t, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = step_dir / f"step_{step:06d}_frames_vs_masked.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual frame data
    for t in range(num_frames):
        frame_data = {
            'original_frame': frames[t],
            'mask': sample_mask[t],
            'mask_ratio': sample_mask[t].mean(),
        }
        np.save(step_dir / f"frame_{t:02d}_data.npy", frame_data)
    
    # Save stats
    stats_path = step_dir / f"step_{step:06d}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Step: {step}\n")
        f.write(f"Sample index: {sample_idx}\n")
        f.write(f"Camera names: {model_input.camera_names}\n")
        f.write(f"Video shape: {frames.shape}\n")
        f.write(f"Mask shape: {sample_mask.shape}\n")
        f.write(f"Overall mask ratio: {sample_mask.mean():.4f}\n")
        f.write(f"Frame-by-frame mask ratios: {[sample_mask[t].mean() for t in range(num_frames)]}\n")
        f.write(f"Video min/max: {frames.min():.4f}/{frames.max():.4f}\n")
        f.write(f"Video mean/std: {frames.mean():.4f}/{frames.std():.4f}\n")
    
    logger.info(f"Saved debug images for step {step} to {step_dir}")
    
    return viz_path


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
    
    # Move model to GPU if available
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    
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
        gradient_clip_norm=getattr(config, 'gradient_clip_norm', 1.0),
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
        tensor = torch.from_numpy(np.array(arr)).float()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
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
        tensor = torch.from_numpy(np.array(arr)).float()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
    def update_model_params(model, jax_params):
        for name, param in model.named_parameters():
            if name in jax_params:
                param.data = jax_to_torch(jax_params[name])
    
    update_model_params(state.model, state.params)
    
    state.model.train()
    
    video_frames = jax_to_torch(batch[0].video_frames)
    mask = jax_to_torch(batch[0].mask).bool()
    
    state.model.zero_grad()
    
    with torch.amp.autocast('cuda'):
        loss = state.model.compute_loss(video_frames, mask)

    if not hasattr(state, 'torch_optimizer'):
        state.torch_optimizer = torch.optim.AdamW(
            state.model.parameters(),
            lr=1e-5,  # Reduced from 1e-4 to slow down learning
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        state.grad_scaler = torch.cuda.amp.GradScaler()
    
    state.grad_scaler.scale(loss).backward()
    
    # Apply gradient clipping for stability
    if hasattr(state, 'gradient_clip_norm'):
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.gradient_clip_norm)
    
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
        'learning_rate': jnp.array(state.torch_optimizer.param_groups[0]['lr']),
        'loss_std': jnp.array(loss.detach().cpu().numpy()),  # For tracking loss variance
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
        tensor = torch.from_numpy(np.array(arr)).float()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
    def update_model_params(model, jax_params):
        for name, param in model.named_parameters():
            if name in jax_params:
                param.data = jax_to_torch(jax_params[name])
    
    update_model_params(model, state.params)
    
    model.eval()
    
    video_frames = jax_to_torch(batch[0].video_frames)
    mask = jax_to_torch(batch[0].mask).bool()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
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


def create_data_loader_with_fallback(
    data_config,
    batch_size: int,
    split: str,
    shuffle: bool,
    num_workers: int,
    fake_data: bool,
    current_step: int,
    chunk_size: int,
    use_optimized: bool = True,
    train_config: WorldModelTrainConfig = None,
):
    """Create dataloader with fallback to original if optimized fails."""
    
    # Try optimized dataloader first if available and requested
    # if use_optimized and OPTIMIZED_DATALOADER_AVAILABLE:
    #     try:
    logger.info(f"Creating optimized dataloader for {split} split...")
    
    # Use config optimization parameters or calculate defaults
    if train_config:
        cache_size = getattr(train_config, 'dataloader_cache_size', max(100, batch_size * 4))
        max_workers = getattr(train_config, 'dataloader_max_workers', min(2, num_workers // 4))
        prefetch_factor = getattr(train_config, 'dataloader_prefetch_factor', 4)
    else:
        cache_size = max(100, batch_size * 4)  # Dynamic cache size
        max_workers = min(2, num_workers // 4)  # Parallel loading workers
        prefetch_factor = 4  # Enhanced prefetching
    
    # Disable DataLoader multiprocessing to avoid pickle errors with spawn
    optimized_num_workers = 0  # Use single-threaded DataLoader
    
    # Adjust prefetch_factor for single-threaded mode
    actual_prefetch_factor = prefetch_factor if optimized_num_workers > 0 else None
    
    loader = create_world_model_data_loader(
        config=data_config,
        batch_size=batch_size,
        split=split,
        shuffle=shuffle,
        num_workers=optimized_num_workers,
        chunk_size=chunk_size,
        cache_size=cache_size,
        max_workers=max_workers,
        prefetch_factor=actual_prefetch_factor,
        pin_memory=False,  # Disable for stability
    )
    
    logger.info(f"✅ Successfully created optimized {split} dataloader "
                f"(workers: {optimized_num_workers}, cache: {cache_size}, "
                f"parallel_workers: {max_workers})")
    
    return loader
    
    # except Exception as e:
    #     logger.warning(f"Failed to create optimized dataloader for {split}: {e}")
        # logger.info("Falling back to original dataloader...")
    
    # Fallback to original dataloader
    # logger.info(f"Creating original dataloader for {split} split...")
    # loader = create_world_model_data_loader(
    #     data_config,
    #     batch_size=batch_size,
    #     split=split,
    #     shuffle=shuffle,
    #     num_workers=min(num_workers, 8),  # Match optimized worker count
    #     # fake_data=fake_data,
    #     current_step=current_step,
    #     chunk_size=chunk_size,
    # )
    
    # logger.info(f"✅ Successfully created original {split} dataloader")
    # return loader


def main(config: WorldModelTrainConfig):
    """Main training loop."""
    init_logging()
    logger.info(f"Starting world model training on {platform.node()}")
    logger.info(f"Configuration: {config.name}")
    
    # Always use optimized dataloader (superior performance)
    use_optimized = True
    logger.info("Using optimized dataloader with smart image resizing")
    
    # Multiprocessing start method already set at module import
    
    # Initialize JAX (will use GPU if available)
    logger.info("Initializing JAX...")
    import jax
    import jax.numpy as jnp
    
    # Check JAX GPU availability
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX default backend: {jax.default_backend()}")
    
    # Force GPU usage if available
    if jax.devices('gpu'):
        logger.info("GPU detected, forcing JAX to use GPU")
        # Test GPU with a simple operation
        test_array = jnp.ones((1000, 1000))
        test_result = jnp.sum(test_array)
        logger.info(f"GPU test successful: {test_result}")
    else:
        logger.warning("No GPU detected by JAX - running on CPU")
    
    rng = jax.random.PRNGKey(config.seed)
    train_rng, init_rng, data_rng = jax.random.split(rng, 3)
    
    state = init_train_state(config, init_rng)
    logger.info(f"Initialized model with {sum(jnp.size(x) for x in jax.tree_leaves(state.params)):,} parameters")
    
    # Create data loaders with fallback
    train_loader = create_data_loader_with_fallback(
        data_config=config.data_config,
        batch_size=config.batch_size,
        split="train",
        shuffle=True,
        num_workers=config.num_workers,
        fake_data=(config.data_config.repo_id is None),
        current_step=state.step,
        chunk_size=getattr(config.data_config, 'chunk_size', 500),
        use_optimized=use_optimized,
        train_config=config,
    )
    
    val_loader = create_data_loader_with_fallback(
        data_config=config.data_config,
        batch_size=config.batch_size,
        split="validation", 
        shuffle=False,
        num_workers=config.num_workers,
        fake_data=(config.data_config.repo_id is None),
        current_step=state.step,
        chunk_size=getattr(config.data_config, 'chunk_size', 500),
        use_optimized=use_optimized,
        train_config=config,
    )
    
    init_wandb(config, resuming=config.resume)
    
    # Log model architecture info to wandb
    log_model_info(state.model, state.model.config if hasattr(state.model, 'config') else config.model_config)
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    metrics_history = []
    loss_smoothing_factor = 0.9  # Exponential moving average factor
    smoothed_loss = None
    
    pbar = tqdm.tqdm(
        range(config.num_train_steps),
        desc="Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
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
        
        # Update step if the loader supports it
        if hasattr(train_loader, 'update_step'):
            train_loader.update_step(step)
        elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'update_step'):
            train_loader.dataset.update_step(step)
        
        step_rng = jax.random.fold_in(train_rng, step)
        state, metrics = train_step(state, batch, step_rng)
        
        # Apply loss smoothing for more stable training signals
        current_loss = metrics['reconstruction_loss']
        if smoothed_loss is None:
            smoothed_loss = current_loss
        else:
            smoothed_loss = loss_smoothing_factor * smoothed_loss + (1 - loss_smoothing_factor) * current_loss
        
        metrics['smoothed_loss'] = smoothed_loss
        metrics_history.append(metrics)
        
        # Update progress bar every step to prevent line clearing
        current_loss = metrics['reconstruction_loss']
        current_mask_ratio = metrics['mask_ratio']
        current_grad_norm = metrics['grad_norm']
        
        # Always update to prevent tqdm from clearing the line
        if hasattr(config, 'progress_bar_verbose') and config.progress_bar_verbose:
            current_metrics_str = f"loss={current_loss:.4f}, mask={current_mask_ratio:.3f}, grad={current_grad_norm:.3f}"
        else:
            current_metrics_str = f"loss={current_loss:.4f}"
        pbar.set_postfix_str(current_metrics_str)
        
        if step % config.log_interval == 0:
            avg_metrics = {}
            for key in metrics_history[0].keys():
                if key.startswith('_'):
                    continue
                try:
                    values = [m[key] for m in metrics_history if m[key] is not None]
                    if values:
                        avg_metrics[key] = jnp.mean(jnp.array(values))
                except (TypeError, ValueError):
                    continue
            
            # Use improved wandb logging with better organization
            log_training_metrics(
                step=step,
                train_loss=avg_metrics.get('reconstruction_loss', 0.0),
                learning_rate=getattr(state.torch_optimizer.param_groups[0], 'lr', None) if hasattr(state, 'torch_optimizer') else None,
                grad_norm=avg_metrics.get('grad_norm', 0.0),
                momentum=getattr(state.model.config, 'momentum', None) if hasattr(state.model, 'config') else None,
            )
            
            # Log debug visualization every 100 steps
            if step % 100 == 0 and step > 0:
                # try:
                # Get a fresh batch for visualization
                viz_batch = next(iter(train_loader))
                if isinstance(viz_batch, tuple) and len(viz_batch) >= 2:
                    video_frames = viz_batch[0].video_frames
                    mask = viz_batch[0].mask
                    
                    # Get model outputs for visualization
                    # with torch.no_grad():
                    #     outputs = state.model.forward(video_frames, mask)
                    
                    log_debug_visualization(
                        video_frames=video_frames,
                        mask=mask,
                        # outputs=outputs,
                        step=step,
                        prefix="train"
                    )
                # except Exception as e:
                #     logger.warning(f"Failed to create debug visualization at step {step}: {e}")
            
            # Don't override the current metrics display with averaged ones
            # The current metrics are more useful for real-time monitoring
            # pbar.set_postfix_str(metrics_str)  # Removed this line
            
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
                    values = [m[key] for m in val_metrics if m[key] is not None]
                    if values:
                        avg_val_metrics[key] = jnp.mean(jnp.array(values))
                except (TypeError, ValueError):
                    continue
            
            # Use improved validation logging
            log_training_metrics(
                step=step,
                train_loss=None,  # No train loss for validation step
                val_metrics=avg_val_metrics,
            )
            logger.info(f"Validation at step {step}: {avg_val_metrics}")
        
        if step % config.save_interval == 0 and step > 0:
            save_checkpoint(state, config, step)
            
            # Save debug images every save_interval
            if MATPLOTLIB_AVAILABLE:
                try:
                    viz_path = save_debug_images(batch, step, config.checkpoint_dir, config)
                    if viz_path and viz_path.exists():
                        # Log to wandb
                        wandb.log({
                            "debug_frames_vs_masked": wandb.Image(str(viz_path)),
                        }, step=step)
                except Exception as e:
                    logger.warning(f"Failed to save debug images for step {step}: {e}")
    
    save_checkpoint(state, config, config.num_train_steps)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    from openpi.training.world_model_training.config import create_world_model_config_cli
    
    config = create_world_model_config_cli()
    main(config) 