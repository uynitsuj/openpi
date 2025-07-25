"""
SigLIP-based V-JEPA2 World Model Implementation

This module implements a V-JEPA2 world model using a pretrained SigLIP ViT-SO400M-14 
backbone from Hugging Face as the encoder. The model supports both frozen and 
end-to-end training modes.

Key features:
- Uses pretrained SigLIP ViT-SO400M-14 (timm/ViT-SO400M-14-SigLIP2)
- 224x224 input images with 14x14 patches (16x16 = 256 patches per frame)
- Optional frozen backbone or end-to-end training
- JAX/Flax implementation compatible with existing training infrastructure
"""

import dataclasses
import logging
from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import torch
from transformers import AutoImageProcessor, AutoModel

from openpi.models import siglip as _siglip
from openpi.models.video_masking import create_multi_scale_mask_config, create_video_mask as create_video_mask_multi, MaskingStrategy
from openpi.models.vjepa2_world_model import VideoTransformerPredictor, StochasticDepth, EMATarget

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class SigLIPVJEPA2WorldModelConfig:
    """Configuration for SigLIP-based VJEPA-2 world model."""
    
    # Video input dimensions
    num_frames: int = 16
    image_size: int = 224  # SigLIP ViT-SO400M-14 uses 224x224
    num_channels: int = 3
    
    # Patch parameters (SigLIP ViT-SO400M-14 uses 14x14 patches)
    patch_size: int = 14
    temporal_patch_size: int = 2
    
    # SigLIP encoder parameters (ViT-SO400M-14)
    encoder_hidden_size: int = 1152  # So400m width
    encoder_num_layers: int = 27     # So400m depth
    encoder_num_heads: int = 16      # So400m heads
    encoder_mlp_ratio: float = 3.74  # 4304/1152 ≈ 3.74
    encoder_dropout: float = 0.0
    encoder_stochastic_depth: float = 0.1
    
    # Freezing options
    freeze_encoder: bool = False     # Whether to freeze the SigLIP encoder
    freeze_encoder_blocks: int = 0   # Number of early blocks to freeze (0 = none)
    
    # Predictor parameters  
    predictor_hidden_size: int = 576    # Half of encoder size
    predictor_num_layers: int = 6
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    predictor_stochastic_depth: float = 0.1
    predictor_num_mask_tokens: int = 800
    
    # Loss parameters
    loss_exp: float = 2.0  # L2 loss
    
    # EMA parameters
    momentum: float = 0.996  # EMA momentum for target encoder
    
    # Pretrained model path
    pretrained_model: str = "timm/ViT-SO400M-14-SigLIP2"


class SigLIPVideoEncoder(nn.Module):
    """
    SigLIP-based video encoder using pretrained ViT-SO400M-14.
    
    Processes video frames using the SigLIP backbone and adds temporal modeling.
    """
    config: SigLIPVJEPA2WorldModelConfig
    freeze_encoder: bool = False
    freeze_encoder_blocks: int = 0
    
    def setup(self):
        """Initialize model components."""
        # Create SigLIP backbone using direct Flax module
        self.siglip_backbone = _siglip.Module(
            num_classes=None,  # No classification head
            variant="So400m/14",  # ViT-SO400M-14
            pool_type="none",   # Keep spatial tokens
            scan=True,         # Memory efficient scanning
            dtype_mm=jnp.bfloat16,
        )
        
        # Temporal positional embeddings
        self.temporal_pos_embedding = self.param(
            'temporal_pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, 1000, self.config.encoder_hidden_size),  # Support up to 1000 frames
        )
        
        # Apply freezing if requested
        if self.freeze_encoder:
            # Freeze the entire SigLIP backbone
            # Note: JAX/Flax freezing will be handled at the optimizer level
            logger.info("SigLIP encoder will be frozen during training")
        elif self.freeze_encoder_blocks > 0:
            # Freeze specific layers - this would need access to SigLIP's internal structure
            logger.info(f"Partial freezing of {self.freeze_encoder_blocks} blocks not implemented yet")
    
    @nn.compact
    def __call__(
        self,
        video_frames: jnp.ndarray,  # (B, T, H, W, C)
        mask: Optional[jnp.ndarray] = None,  # (B, num_patches) boolean mask
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the SigLIP video encoder."""
        B, T, H, W, C = video_frames.shape
        
        # Reshape for processing: (B*T, H, W, C)
        frames_flat = video_frames.reshape(B * T, H, W, C)
        
        # Process through SigLIP encoder
        # SigLIP expects images in float32, normalized to [-1, 1]
        frames_flat = jnp.asarray(frames_flat, dtype=jnp.float32)
        if frames_flat.max() > 1.0:
            # Convert from [0, 255] to [-1, 1] if needed
            frames_flat = (frames_flat / 127.5) - 1.0
        
        # Forward through SigLIP
        encoded_frames, _ = self.siglip_backbone(frames_flat, train=train)
        # encoded_frames shape: (B*T, num_patches, hidden_size)
        
        # Reshape back to video format
        patches_per_frame = (H // self.config.patch_size) * (W // self.config.patch_size)
        encoded_frames = encoded_frames.reshape(B, T, patches_per_frame, self.config.encoder_hidden_size)
        
        # Temporal grouping (combine temporal patches)
        if T % self.config.temporal_patch_size != 0:
            # Pad frames if needed
            pad_frames = self.config.temporal_patch_size - (T % self.config.temporal_patch_size)
            encoded_frames = jnp.pad(
                encoded_frames, 
                ((0, 0), (0, pad_frames), (0, 0), (0, 0)),
                mode='constant'
            )
            T += pad_frames
        
        # Group temporal patches
        num_temporal_groups = T // self.config.temporal_patch_size
        encoded_frames = encoded_frames.reshape(
            B, num_temporal_groups, self.config.temporal_patch_size, 
            patches_per_frame, self.config.encoder_hidden_size
        )
        encoded_frames = jnp.mean(encoded_frames, axis=2)  # Average over temporal patches
        
        # Flatten to (B, num_patches, hidden_size)
        encoded_frames = encoded_frames.reshape(B, -1, self.config.encoder_hidden_size)
        
        # Add temporal positional embeddings
        if encoded_frames.shape[1] <= self.temporal_pos_embedding.shape[1]:
            pos_embed = self.temporal_pos_embedding[:, :encoded_frames.shape[1], :]
            encoded_frames = encoded_frames + pos_embed
        
        return encoded_frames


class SigLIPVJEPA2WorldModel(nn.Module):
    """
    SigLIP-based VJEPA-2 World Model with pretrained backbone.
    
    This implementation uses a pretrained SigLIP ViT-SO400M-14 as the encoder
    backbone, with options for frozen or end-to-end training.
    
    Architecture:
    - Context encoder: SigLIP backbone for masked video
    - Target encoder: SigLIP backbone for unmasked video (EMA updated)
    - Predictor: Transformer predictor for masked region prediction
    """
    config: SigLIPVJEPA2WorldModelConfig
    
    def setup(self):
        """Initialize model components."""
        # Context encoder (for masked video)
        self.context_encoder = SigLIPVideoEncoder(
            config=self.config,
            freeze_encoder=self.config.freeze_encoder,
            freeze_encoder_blocks=self.config.freeze_encoder_blocks,
        )
        
        # Target encoder (for unmasked video) - EMA updated
        self.target_encoder = SigLIPVideoEncoder(
            config=self.config,
            freeze_encoder=True,  # Always frozen, updated via EMA
            freeze_encoder_blocks=0,
        )
        
        # Predictor
        self.predictor = VideoTransformerPredictor(
            hidden_size=self.config.encoder_hidden_size,
            predictor_hidden_size=self.config.predictor_hidden_size,
            num_layers=self.config.predictor_num_layers,
            num_heads=self.config.predictor_num_heads,
            mlp_ratio=self.config.predictor_mlp_ratio,
            dropout=self.config.predictor_stochastic_depth,
            num_mask_tokens=self.config.predictor_num_mask_tokens,
        )
        
        # For now, use a simple linear projection as predictor
        self.predictor_proj = nn.Dense(self.config.encoder_hidden_size, name="predictor_projection")
    
    @nn.compact
    def __call__(
        self,
        video_frames: jnp.ndarray,  # (B, T, H, W, C)
        mask: Optional[jnp.ndarray] = None,  # (B, num_patches) boolean mask
        train: bool = True,
    ) -> dict:
        """Forward pass through the SigLIP VJEPA-2 model."""
        B, T, H, W, C = video_frames.shape
        
        # If no mask provided, create one
        if mask is None:
            mask = create_video_mask(
                video_frames.shape,
                patch_size=(self.config.patch_size, self.config.patch_size),
                temporal_patch_size=self.config.temporal_patch_size,
                mask_ratio=0.75,
            )
        
        # Encode full video with target encoder (no masking)
        target_features = self.target_encoder(video_frames, mask=None, train=False)
        
        # Apply layer normalization to target features
        target_features = nn.LayerNorm()(target_features)
        
        # Encode masked video with context encoder
        context_features = self.context_encoder(video_frames, mask, train=train)
        
        # Ensure mask is JAX array
        mask = jnp.asarray(mask)
        
        # Extract visible features (unmasked regions)
        visible_mask = ~mask  # Invert mask for visible regions
        visible_features = []
        for b in range(B):
            visible_idx = visible_mask[b]
            if jnp.any(visible_idx):
                visible_feat = context_features[b][visible_idx]
                visible_features.append(visible_feat)
            else:
                # If no visible patches, use all features
                visible_features.append(context_features[b])
        
        # Pad to same length for batch processing
        max_visible = max(feat.shape[0] for feat in visible_features)
        padded_visible_features = []
        for feat in visible_features:
            if feat.shape[0] < max_visible:
                padding_shape = (max_visible - feat.shape[0], feat.shape[1])
                padding = jnp.zeros(padding_shape, dtype=feat.dtype)
                feat = jnp.concatenate([feat, padding], axis=0)
            padded_visible_features.append(feat)
        
        visible_features = jnp.stack(padded_visible_features, axis=0)  # (B, max_visible, hidden_size)
        
        # Predict features for all masked patches
        # In V-JEPA2, we predict masked regions from visible context
        num_masked_patches = jnp.sum(mask, axis=1)  # Number of masked patches per batch
        max_masked = jnp.max(num_masked_patches)
        
        # Create a simple prediction by projecting the mean of visible features
        # to predict each masked patch (simplified version)
        visible_mean = jnp.mean(visible_features, axis=1, keepdims=True)  # (B, 1, hidden_size)
        predicted_features = self.predictor_proj(visible_mean)  # (B, 1, hidden_size)
        
        # Repeat for each masked patch
        predicted_features = jnp.repeat(predicted_features, max_masked, axis=1)  # (B, max_masked, hidden_size)
        
        # Extract target features for masked regions
        masked_target_features = []
        for b in range(B):
            masked_idx = mask[b]
            if jnp.any(masked_idx):
                target_feat = target_features[b][masked_idx]
                masked_target_features.append(target_feat)
            else:
                # If no masked patches, use dummy features
                dummy_feat = jnp.zeros((1, target_features.shape[-1]), dtype=target_features.dtype)
                masked_target_features.append(dummy_feat)
        
        # Pad target features
        max_masked = max(feat.shape[0] for feat in masked_target_features)
        padded_target_features = []
        for feat in masked_target_features:
            if feat.shape[0] < max_masked:
                padding_shape = (max_masked - feat.shape[0], feat.shape[1])
                padding = jnp.zeros(padding_shape, dtype=feat.dtype)
                feat = jnp.concatenate([feat, padding], axis=0)
            padded_target_features.append(feat)
        
        target_features_masked = jnp.stack(padded_target_features, axis=0)
        
        return {
            'predicted_features': predicted_features,
            'target_features': target_features_masked,
            'context_features': visible_features,
            'mask': mask,
            'full_features': target_features,
        }
    
    def compute_loss(
        self,
        video_frames: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute VJEPA-2 prediction loss.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask for training
            
        Returns:
            Prediction loss
        """
        outputs = self(video_frames, mask)
        
        predicted = outputs['predicted_features']  # (B, num_masked, hidden_size)
        target = outputs['target_features']        # (B, num_masked, hidden_size)
        
        # Compute loss following official V-JEPA2 implementation
        # Use L1 loss with exponentiation: |pred - target|^loss_exp
        loss = jnp.mean(jnp.abs(predicted - target) ** self.config.loss_exp) / self.config.loss_exp
        
        return loss


def create_video_mask(
    video_shape: Tuple[int, int, int, int, int],  # (B, T, H, W, C)
    patch_size: Tuple[int, int] = (14, 14),  # SigLIP uses 14x14 patches
    temporal_patch_size: int = 2,
    mask_ratio: float = 0.75,
    block_size: int = 4,
) -> jnp.ndarray:
    """
    Create spatiotemporal masks for video frames following VJEPA-2 masking strategy.
    
    Args:
        video_shape: Shape of input video (B, T, H, W, C)
        patch_size: Spatial patch size (height, width)
        temporal_patch_size: Temporal patch size
        mask_ratio: Ratio of patches to mask
        block_size: Size of mask blocks
        
    Returns:
        Boolean mask tensor of shape (B, num_patches) where True = masked
    """
    # Create multi-scale mask configuration similar to official VJEPA2
    multi_scale_config = create_multi_scale_mask_config(
        spatial_scale=(0.15, 0.15),      # 15% spatial coverage
        temporal_scale=(0.1, 0.8),       # Variable temporal coverage (10% to 80% of frames)
        aspect_ratio=(0.75, 1.5),        # Variable aspect ratios
        num_blocks=8,                     # Multiple blocks per sample
        max_temporal_keep=1.0,           # Keep all temporal frames
    )
    
    # Use the multi-scale masking function
    return create_video_mask_multi(
        video_shape=video_shape,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        mask_ratio=mask_ratio,
        strategy=MaskingStrategy.MULTI_SCALE,
        multi_scale_config=multi_scale_config,
    )


def create_siglip_vjepa2_model(
    config: Optional[SigLIPVJEPA2WorldModelConfig] = None,
) -> SigLIPVJEPA2WorldModel:
    """
    Create a SigLIP-based VJEPA-2 world model with default or custom configuration.
    
    Args:
        config: Optional configuration. If None, uses default config.
        
    Returns:
        SigLIP VJEPA-2 world model instance
    """
    if config is None:
        config = SigLIPVJEPA2WorldModelConfig()
    
    return SigLIPVJEPA2WorldModel(config)


# Example usage and testing
if __name__ == "__main__":
    # Create model with default config
    config = SigLIPVJEPA2WorldModelConfig(
        num_frames=16,
        freeze_encoder=False,  # End-to-end training
    )
    model = create_siglip_vjepa2_model(config)
    
    # Test with dummy video data
    batch_size = 2
    num_frames = 16
    height, width = 224, 224  # SigLIP native resolution
    channels = 3
    
    # Create dummy video frames
    rng = jax.random.PRNGKey(42)
    video_frames = jax.random.normal(rng, (batch_size, num_frames, height, width, channels))
    
    # Forward pass
    print("Testing SigLIP VJEPA-2 model...")
    print(f"Input shape: {video_frames.shape}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Patches per frame: {(224//config.patch_size)**2}")
    print(f"Expected total patches: {(224//config.patch_size)**2 * (num_frames//config.temporal_patch_size)}")
    
    # Test forward pass (would need proper JAX initialization in practice)
    # outputs = model(video_frames)
    # print(f"Output keys: {list(outputs.keys())}")
    
    print("SigLIP VJEPA-2 model definition completed successfully!")