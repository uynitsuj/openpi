"""
Wandb utilities for VJEPA-2 training visualization and logging.
"""

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("openpi")


def create_mask_visualization(
    video_frames: torch.Tensor,
    mask: torch.Tensor,
    predicted_features: Optional[torch.Tensor] = None,
    max_frames_to_show: int = 4,
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
        
        # Convert to numpy and normalize to [0, 1]
        frames = video_frames[:B, :T].detach().cpu().numpy()
        if frames.max() <= 1.0:
            frames = np.clip(frames, 0, 1)
        else:
            frames = np.clip(frames / 255.0, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(B, T * 2, figsize=(T * 4, B * 2))
        if B == 1:
            axes = axes.reshape(1, -1)
        if T == 1:
            axes = axes.reshape(B, 2)
            
        for b in range(B):
            for t in range(T):
                # Original frame
                ax_orig = axes[b, t * 2]
                ax_orig.imshow(frames[b, t])
                ax_orig.set_title(f"Batch {b}, Frame {t}")
                ax_orig.axis('off')
                
                # Masked frame
                ax_masked = axes[b, t * 2 + 1]
                masked_frame = frames[b, t].copy()
                
                # Apply mask visualization (simplified - assumes patch-based masking)
                if mask is not None and len(mask.shape) >= 2:
                    # Convert patch mask to spatial mask (simplified)
                    patch_size = 16  # Assume 16x16 patches
                    patches_per_dim = H // patch_size
                    
                    if mask.shape[1] >= patches_per_dim ** 2:
                        mask_2d = mask[b].detach().cpu().numpy()
                        for i in range(patches_per_dim):
                            for j in range(patches_per_dim):
                                patch_idx = i * patches_per_dim + j
                                if patch_idx < len(mask_2d) and mask_2d[patch_idx]:
                                    # Mask this patch (make it gray)
                                    y_start, y_end = i * patch_size, (i + 1) * patch_size
                                    x_start, x_end = j * patch_size, (j + 1) * patch_size
                                    masked_frame[y_start:y_end, x_start:x_end] = 0.5
                
                ax_masked.imshow(masked_frame)
                ax_masked.set_title(f"Masked Frame {t}")
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
    outputs: Dict[str, torch.Tensor],
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
            predicted_features=outputs.get('predicted_features')
        )
        
        # Log to wandb
        wandb.log({
            f"{prefix}/mask_visualization": viz_image,
            f"{prefix}/mask_ratio": mask.float().mean().item() if mask is not None else 0.0,
            f"{prefix}/num_masked_patches": mask.sum().item() if mask is not None else 0,
        }, step=step)
        
        logger.info(f"Debug visualization logged at step {step}")
        
    except Exception as e:
        logger.warning(f"Failed to log debug visualization at step {step}: {e}")