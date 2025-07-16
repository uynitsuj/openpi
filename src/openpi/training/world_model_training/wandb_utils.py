"""
Wandb utilities for VJEPA-2 training visualization and logging.
"""

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger("openpi")

# Compatibility functions for JAX/PyTorch interop
def safe_detach(tensor):
    """Safely detach tensor, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'detach'):
        return tensor.detach()
    elif hasattr(tensor, '__array__'):
        # JAX array - already detached by nature
        return tensor
    else:
        return tensor

def safe_float(tensor):
    """Safely convert to float, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'float'):
        return tensor.float()
    elif hasattr(tensor, 'astype'):
        # JAX array
        import jax.numpy as jnp
        return tensor.astype(jnp.float32)
    else:
        return tensor

def safe_cpu(tensor):
    """Safely move to CPU, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'cpu'):
        return tensor.cpu()
    else:
        # JAX arrays are already on CPU by default
        return tensor

def safe_numpy(tensor):
    """Safely convert to numpy, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'detach'):
        # PyTorch tensor
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, '__array__'):
        # JAX array
        return np.array(tensor)
    else:
        return np.array(tensor)

def safe_mean(tensor):
    """Safely compute mean, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'mean'):
        return tensor.mean()
    else:
        import jax.numpy as jnp
        return jnp.mean(tensor)

def safe_sum(tensor):
    """Safely compute sum, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'sum'):
        return tensor.sum()
    else:
        import jax.numpy as jnp
        return jnp.sum(tensor)

def safe_item(tensor):
    """Safely extract scalar value, handling both PyTorch and JAX arrays."""
    if hasattr(tensor, 'item'):
        return tensor.item()
    else:
        # JAX array
        return float(tensor)

def denormalize_minus_one_to_one(tensor):
    """
    Denormalize images from [-1, 1] range back to [0, 1] range.
    
    This matches the normalization used in the data loader:
    - Original: [0, 1] (after dividing by 255)
    - Normalized: image * 2.0 - 1.0 → [-1, 1]
    - Denormalized: (image + 1.0) / 2.0 → [0, 1]
    
    Args:
        tensor: Normalized tensor in [-1, 1] range
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    # Convert to numpy if needed
    if not isinstance(tensor, np.ndarray):
        tensor = safe_numpy(tensor)
    
    # Denormalize: x = (x_normalized + 1.0) / 2.0
    denormalized = (tensor + 1.0) / 2.0
    
    # Clip to [0, 1] range to handle any numerical errors
    denormalized = np.clip(denormalized, 0, 1)
    
    return denormalized


def create_mask_visualization(
    video_frames: torch.Tensor,
    mask: torch.Tensor,
    # predicted_features: Optional[torch.Tensor] = None,
    max_frames_to_show: int = 10,
    max_batch_items: int = 2,
) -> wandb.Image:
    """
    Create a visualization showing original frames, masked patches, and predictions.
    
    Args:
        video_frames: Input video frames (B, T, H, W, C)
        mask: Boolean mask (B, num_patches) where True = masked
        predicted_features: Optional predicted features for reconstruction
        max_frames_to_show: Maximum number of frames to display
        max_batch_items: Maximum number of batch items to show
        
    Returns:
        wandb.Image object for logging
    """
    try:
        B, T, H, W, C = video_frames.shape
        
        # Limit batch size and frames for visualization
        B = min(B, max_batch_items)
        T = min(T, max_frames_to_show)
        
        # Convert to numpy and denormalize from [-1, 1] to [0, 1] range
        frames = denormalize_minus_one_to_one(video_frames[:B, :T])
        # Frames should already be in [0, 1] range after denormalization
        frames = np.clip(frames, 0, 1)
        
        # Calculate grid layout - 8 elements per row (4 original + 4 masked frames)
        frames_per_row = 4  # 4 pairs of (original, masked) per row
        total_pairs = T  # Total number of frame pairs
        
        # Calculate rows needed for each batch
        rows_per_batch = (total_pairs + frames_per_row - 1) // frames_per_row
        total_rows = B * rows_per_batch
        
        # Create figure with dynamic size
        fig, axes = plt.subplots(total_rows, frames_per_row * 2, figsize=(frames_per_row * 4, total_rows * 2))
        
        # Ensure axes is always 2D
        if total_rows == 1:
            axes = axes.reshape(1, -1)
        elif frames_per_row * 2 == 1:
            axes = axes.reshape(-1, 1)
        
        # Handle single subplot case
        if not hasattr(axes, 'shape'):
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            if total_rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(-1, 1)
        
        # Turn off all axes first
        for i in range(total_rows):
            for j in range(frames_per_row * 2):
                axes[i, j].axis('off')
            
        for b in range(B):
            for t in range(T):
                # Calculate which row and column this frame should go in
                row_in_batch = t // frames_per_row
                col_in_row = t % frames_per_row
                
                # Calculate absolute row position
                abs_row = b * rows_per_batch + row_in_batch
                
                # Original frame
                ax_orig = axes[abs_row, col_in_row * 2]
                ax_orig.imshow(frames[b, t])
                # Add a hash of the frame to help identify if samples are changing
                frame_hash = hash(frames[b, t].tobytes()) % 10000
                ax_orig.set_title(f"Batch {b}, Frame {t} ({frame_hash})", fontsize=8)
                ax_orig.axis('off')
                
                # Masked frame
                ax_masked = axes[abs_row, col_in_row * 2 + 1]
                masked_frame = frames[b, t].copy()
                
                # Apply mask visualization (spatiotemporal masking)
                if mask is not None and len(mask.shape) >= 2:
                    # Convert spatiotemporal patch mask to spatial mask for this time step
                    patch_size = 16  # Default 16x16 patches (matches VJEPA2WorldModelConfig default)
                    patches_per_dim = H // patch_size
                    temporal_patch_size = 2  # Default temporal patch size (matches VJEPA2WorldModelConfig default)
                    
                    # Calculate temporal patch index for this frame
                    temporal_patch_idx = t // temporal_patch_size
                    
                    # Get the full mask for this batch item
                    full_mask = safe_numpy(mask[b])
                    
                    # Extract mask for this time step
                    for i in range(patches_per_dim):
                        for j in range(patches_per_dim):
                            # Calculate the 3D patch index (t, h, w)
                            # The mask is flattened as: t * (H_patches * W_patches) + h * W_patches + w
                            patch_3d_idx = temporal_patch_idx * (patches_per_dim * patches_per_dim) + i * patches_per_dim + j
                            
                            if patch_3d_idx < len(full_mask) and full_mask[patch_3d_idx]:
                                # Mask this patch (make it gray)
                                y_start, y_end = i * patch_size, (i + 1) * patch_size
                                x_start, x_end = j * patch_size, (j + 1) * patch_size
                                masked_frame[y_start:y_end, x_start:x_end] = 0.5
                
                ax_masked.imshow(masked_frame)
                ax_masked.set_title(f"Masked Frame {t}", fontsize=8)
                ax_masked.axis('off')
        
        plt.tight_layout()
        
        # Convert to wandb image
        wandb_image = wandb.Image(fig)
        plt.close(fig)
        
        return wandb_image
        
    except Exception as e:
        logger.warning(f"Failed to create mask visualization: {e}")
        # Return a simple placeholder
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, f"Visualization failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        wandb_image = wandb.Image(fig)
        plt.close(fig)
        return wandb_image


def log_training_metrics(
    step: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]] = None,
    learning_rate: Optional[float] = None,
    grad_norm: Optional[float] = None,
    momentum: Optional[float] = None,
) -> None:
    """
    Log training metrics to wandb with proper organization.
    
    Args:
        step: Training step
        train_loss: Training loss value
        val_metrics: Optional validation metrics
        learning_rate: Current learning rate
        grad_norm: Gradient norm
        momentum: EMA momentum value
    """
    metrics = {"train/loss": train_loss, "train/step": step}
    
    if learning_rate is not None:
        metrics["train/learning_rate"] = learning_rate
        
    if grad_norm is not None:
        metrics["train/grad_norm"] = grad_norm
        
    if momentum is not None:
        metrics["train/momentum"] = momentum
    
    if val_metrics:
        for key, value in val_metrics.items():
            # Organize validation metrics under val/ prefix
            clean_key = key.replace("val_", "").replace("validation_", "")
            metrics[f"val/{clean_key}"] = value
    
    wandb.log(metrics, step=step)


def log_model_info(
    model: torch.nn.Module,
    config: Any,
    step: int = 0,
) -> None:
    """
    Log model architecture and configuration info.
    
    Args:
        model: The model to analyze
        config: Model configuration
        step: Current step
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log model info
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/encoder_hidden_size": getattr(config, 'encoder_hidden_size', 'unknown'),
            "model/encoder_layers": getattr(config, 'encoder_num_layers', 'unknown'),
            "model/predictor_hidden_size": getattr(config, 'predictor_hidden_size', 'unknown'),
            "model/predictor_layers": getattr(config, 'predictor_num_layers', 'unknown'),
        }, step=step)
        
        logger.info(f"Model info logged - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
    except Exception as e:
        logger.warning(f"Failed to log model info: {e}")


def log_debug_visualization(
    video_frames: torch.Tensor,
    mask: torch.Tensor,
    # outputs: Dict[str, torch.Tensor],
    step: int,
    prefix: str = "debug",
) -> None:
    """
    Log debug visualizations every N steps.
    
    Args:
        video_frames: Input video frames
        mask: Mask used for training
        outputs: Model outputs containing predictions
        step: Current training step
        prefix: Prefix for wandb logging
    """
    try:
        # Create mask visualization
        viz_image = create_mask_visualization(
            video_frames=video_frames,
            mask=mask,
            # predicted_features=outputs.get('predicted_features'),
            max_frames_to_show=10,
            max_batch_items=2
        )
        
        # Log to wandb
        wandb.log({
            f"{prefix}/mask_visualization": viz_image,
            f"{prefix}/mask_ratio": safe_item(safe_mean(safe_float(mask))) if mask is not None else 0.0,
            f"{prefix}/num_masked_patches": safe_item(safe_sum(mask)) if mask is not None else 0,
        }, step=step)
        
        logger.info(f"Debug visualization logged at step {step}")
        
    except Exception as e:
        logger.warning(f"Failed to log debug visualization at step {step}: {e}")