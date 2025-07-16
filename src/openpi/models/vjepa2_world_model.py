"""
VJEPA-2 World Model Implementation

This module implements a VJEPA-2 inspired world model for video understanding and prediction.
Based on the Video Joint-Embedding Predictive Architecture (V-JEPA) from Meta AI.

The model uses a vision encoder + predictor architecture for masked video modeling,
where visible patches are encoded and used to predict masked regions in representation space.
"""

import dataclasses
import logging
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers.models.vit import ViTModel, ViTConfig
import math
from openpi.models.video_masking import create_multi_scale_mask_config, create_video_mask as create_video_mask_multi, MaskingStrategy

logger = logging.getLogger("openpi")


class StochasticDepth(nn.Module):
    """
    Stochastic Depth implementation for transformer layers.
    Based on official VJEPA-2 implementation with layer-wise decay.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training or self.drop_prob == 0.0:
            return x
            
        # Sample keep probability
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with arbitrary tensor shapes
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for video transformers.
    Based on official VJEPA-2 implementation.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute frequency scales
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute sin/cos values
        t = torch.arange(max_seq_len).type_as(freqs)
        freqs = torch.outer(t, freqs)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor (B, seq_len, dim)
            seq_len: Sequence length
            
        Returns:
            Tensor with RoPE applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        cos = self.cos_cached[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin_cached[:seq_len].view(1, seq_len, 1, -1)
        
        # Split x into pairs for rotation
        x1, x2 = x[..., 0::2], x[..., 1::2]
        
        # Apply rotation
        x_rope = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return x_rope


def create_video_mask(
    video_shape: Tuple[int, int, int, int, int],  # (B, T, H, W, C)
    patch_size: Tuple[int, int] = (16, 16),
    temporal_patch_size: int = 2,
    mask_ratio: float = 0.75,
    block_size: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create spatiotemporal masks for video frames following VJEPA-2 masking strategy.
    
    Args:
        video_shape: Shape of input video (B, T, H, W, C)
        patch_size: Spatial patch size (height, width)
        temporal_patch_size: Temporal patch size
        mask_ratio: Ratio of patches to mask
        block_size: Size of mask blocks
        device: Device to create masks on
        
    Returns:
        Boolean mask tensor of shape (B, num_patches) where True = masked
    """
    # Create multi-scale mask configuration similar to official VJEPA2
    multi_scale_config = create_multi_scale_mask_config(
        spatial_scale=(0.15, 0.15),      # 15% spatial coverage
        temporal_scale=(1.0, 1.0),       # Full temporal coverage
        aspect_ratio=(0.75, 1.5),        # Variable aspect ratios
        num_blocks=8,                     # Multiple blocks per sample
        max_temporal_keep=1.0,           # Keep all temporal frames
    )
    
    # Use the new multi-scale masking function
    return create_video_mask_multi(
        video_shape=video_shape,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        mask_ratio=mask_ratio,
        strategy=MaskingStrategy.MULTI_SCALE,
        device=device,
        multi_scale_config=multi_scale_config,
    )


class VideoTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for video sequences.
    
    Takes video frames and produces patch embeddings using a ViT-based architecture.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        num_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_pretrained: bool = True,
        pretrained_model: str = "google/vit-base-patch16-224",
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        # Calculate patch dimensions
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        
        if use_pretrained:
            # Load pretrained ViT model
            self.vit_config = AutoConfig.from_pretrained(pretrained_model)
            self.vit_config.hidden_size = hidden_size
            self.vit_config.num_hidden_layers = num_layers
            self.vit_config.num_attention_heads = num_heads
            self.vit_config.intermediate_size = int(hidden_size * mlp_ratio)
            self.vit_config.hidden_dropout_prob = dropout
            self.vit_config.attention_probs_dropout_prob = dropout
            
            self.vit_model = ViTModel(self.vit_config)
            # Enable gradient checkpointing to save memory
            self.vit_model.gradient_checkpointing_enable()
        else:
            # Create ViT from scratch
            self.vit_config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=int(hidden_size * mlp_ratio),
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
            self.vit_model = ViTModel(self.vit_config)
            # Enable gradient checkpointing to save memory
            self.vit_model.gradient_checkpointing_enable()
        
        # Temporal positional embeddings
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, 1000, hidden_size) * 0.02  # Support up to 1000 frames
        )
        
    def forward(
        self,
        video_frames: torch.Tensor,  # (B, T, H, W, C)
        mask: Optional[torch.Tensor] = None,  # (B, num_patches) boolean mask
    ) -> torch.Tensor:
        """
        Forward pass through the video encoder.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask indicating which patches to mask (True = masked)
            
        Returns:
            Encoded video features (B, num_patches, hidden_size)
        """
        B, T, H, W, C = video_frames.shape
        
        # Reshape for processing: (B*T, H, W, C)
        frames_flat = video_frames.view(B * T, H, W, C)
        
        # Convert to (B*T, C, H, W) for ViT
        frames_flat = frames_flat.permute(0, 3, 1, 2)
        
        # Process through ViT encoder
        # ViT expects (batch_size, num_channels, height, width)
        vit_outputs = self.vit_model(pixel_values=frames_flat)
        patch_embeddings = vit_outputs.last_hidden_state  # (B*T, num_patches+1, hidden_size)
        
        # Remove CLS token
        patch_embeddings = patch_embeddings[:, 1:, :]  # (B*T, num_patches, hidden_size)
        
        # Reshape back to video format
        patches_per_frame = self.num_patches_per_frame
        patch_embeddings = patch_embeddings.view(B, T, patches_per_frame, self.hidden_size)
        
        # Temporal grouping (combine temporal patches)
        if T % self.temporal_patch_size != 0:
            # Pad frames if needed
            pad_frames = self.temporal_patch_size - (T % self.temporal_patch_size)
            patch_embeddings = F.pad(patch_embeddings, (0, 0, 0, 0, 0, pad_frames))
            T += pad_frames
        
        # Group temporal patches
        num_temporal_groups = T // self.temporal_patch_size
        patch_embeddings = patch_embeddings.view(B, num_temporal_groups, self.temporal_patch_size, patches_per_frame, self.hidden_size)
        patch_embeddings = patch_embeddings.mean(dim=2)  # Average over temporal patches
        
        # Flatten to (B, num_patches, hidden_size)
        patch_embeddings = patch_embeddings.view(B, -1, self.hidden_size)
        
        # Add temporal positional embeddings
        if patch_embeddings.size(1) <= self.temporal_pos_embedding.size(1):
            pos_embed = self.temporal_pos_embedding[:, :patch_embeddings.size(1), :]
            patch_embeddings = patch_embeddings + pos_embed
        
        return patch_embeddings


class VideoTransformerPredictor(nn.Module):
    """
    Vision Transformer predictor for masked video modeling.
    
    Takes context features and predicts masked regions.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        predictor_hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_mask_tokens: int = 100,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.predictor_hidden_size = predictor_hidden_size
        self.num_mask_tokens = num_mask_tokens
        
        # Project context features to predictor dimension
        self.context_proj = nn.Linear(hidden_size, predictor_hidden_size)
        
        # Mask token for prediction (single learnable token like official VJEPA-2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_hidden_size))
        
        # Transformer blocks for prediction with stochastic depth
        stoch_depth_probs = [0.0] + [dropout * (i / (num_layers - 1)) for i in range(1, num_layers)]
        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_hidden_size,
                nhead=num_heads,
                dim_feedforward=int(predictor_hidden_size * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Stochastic depth for each layer
        self.stochastic_depths = nn.ModuleList([
            StochasticDepth(prob) for prob in stoch_depth_probs
        ])
        
        # Final projection back to encoder dimension
        self.output_proj = nn.Linear(predictor_hidden_size, hidden_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(predictor_hidden_size)
        
    def forward(
        self,
        context_features: torch.Tensor,  # (B, num_context_patches, hidden_size)
        mask: torch.Tensor,  # (B, total_patches) boolean mask
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.
        
        Args:
            context_features: Features from visible patches
            mask: Boolean mask indicating which patches to predict
            
        Returns:
            Predicted features for masked regions (B, num_masked_patches, hidden_size)
        """
        B, num_context_patches, _ = context_features.shape
        
        # Project context features
        context_features = self.context_proj(context_features)
        
        # Create mask tokens for prediction
        num_masked = mask.sum(dim=1)  # (B,)
        max_masked = num_masked.max().item()
        
        # Use the single mask token (like official VJEPA-2)
        mask_tokens = self.mask_token.expand(B, max_masked, -1)  # (B, max_masked, predictor_hidden_size)
        
        # Concatenate context and mask tokens
        combined_features = torch.cat([context_features, mask_tokens], dim=1)
        
        # Apply transformer blocks with stochastic depth
        for block, stoch_depth in zip(self.predictor_blocks, self.stochastic_depths):
            residual = combined_features
            x = block(combined_features)
            # Apply stochastic depth to the residual connection
            combined_features = residual + stoch_depth(x - residual, training=self.training)
        
        # Normalize
        combined_features = self.norm(combined_features)
        
        # Extract predicted features (mask tokens)
        predicted_features = combined_features[:, num_context_patches:, :]
        
        # Project back to encoder dimension
        predicted_features = self.output_proj(predicted_features)
        
        return predicted_features


@dataclasses.dataclass
class VJEPA2WorldModelConfig:
    """Configuration for VJEPA-2 world model."""
    
    # Video input dimensions
    num_frames: int = 16
    image_size: int = 224
    num_channels: int = 3
    
    # Patch parameters
    patch_size: int = 16
    temporal_patch_size: int = 2
    
    # Encoder parameters
    encoder_hidden_size: int = 768
    encoder_num_layers: int = 12
    encoder_num_heads: int = 12
    encoder_mlp_ratio: float = 4.0
    encoder_dropout: float = 0.0
    encoder_stochastic_depth: float = 0.1  # Stochastic depth for encoder layers
    
    # Predictor parameters
    predictor_hidden_size: int = 384
    predictor_num_layers: int = 6
    predictor_num_heads: int = 6
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    predictor_stochastic_depth: float = 0.1  # Stochastic depth for predictor layers
    predictor_num_mask_tokens: int = 800
    
    # Loss parameters
    loss_exp: float = 2.0  # L2 loss
    
    # EMA parameters (like official VJEPA-2)
    momentum: float = 0.996  # EMA momentum for target encoder
    
    # Pretrained model
    use_pretrained_encoder: bool = True
    pretrained_model: str = "google/vit-base-patch16-224"


class VJEPA2WorldModel(nn.Module):
    """
    VJEPA-2 World Model with separate context and target encoders.
    
    This implementation follows the official V-JEPA2 architecture:
    - Context encoder: Processes masked video
    - Target encoder: Processes unmasked video (frozen/EMA)
    - Predictor: Predicts target features from context
    """
    
    def __init__(self, config: VJEPA2WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Context encoder (for masked video)
        self.context_encoder = VideoTransformerEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            num_channels=config.num_channels,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            mlp_ratio=config.encoder_mlp_ratio,
            dropout=config.encoder_dropout,
            use_pretrained=config.use_pretrained_encoder,
            pretrained_model=config.pretrained_model,
        )
        
        # Target encoder (for unmasked video) - should be frozen/EMA
        self.target_encoder = VideoTransformerEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            num_channels=config.num_channels,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            mlp_ratio=config.encoder_mlp_ratio,
            dropout=config.encoder_dropout,
            use_pretrained=config.use_pretrained_encoder,
            pretrained_model=config.pretrained_model,
        )
        
        # Initialize target encoder with same weights as context encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor
        self.predictor = VideoTransformerPredictor(
            hidden_size=config.encoder_hidden_size,
            predictor_hidden_size=config.predictor_hidden_size,
            num_layers=config.predictor_num_layers,
            num_heads=config.predictor_num_heads,
            mlp_ratio=config.predictor_mlp_ratio,
            dropout=config.predictor_dropout,
            num_mask_tokens=config.predictor_num_mask_tokens,
        )
        
        # Loss exponent
        self.loss_exp = config.loss_exp
        
        # EMA momentum for target encoder updates
        self.momentum = config.momentum
        
    def forward(
        self,
        video_frames: torch.Tensor,  # (B, T, H, W, C)
        mask: Optional[torch.Tensor] = None,  # (B, num_patches) boolean mask
    ) -> dict:
        """
        Forward pass through the VJEPA-2 model.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask indicating which patches to mask (True = masked)
            
        Returns:
            Dictionary containing model outputs
        """
        B, T, H, W, C = video_frames.shape
        
        # If no mask provided, create one
        if mask is None:
            mask = create_video_mask(
                video_frames.shape,
                patch_size=(self.config.patch_size, self.config.patch_size),
                temporal_patch_size=self.config.temporal_patch_size,
                mask_ratio=0.75,
                device=video_frames.device,
            )
        
        # Encode full video with target encoder (no masking)
        with torch.no_grad():
            target_features = self.target_encoder(video_frames, mask=None)
        
        # Apply layer normalization to target features
        target_features = F.layer_norm(target_features, (target_features.size(-1),))
        
        # Encode masked video with context encoder
        context_features = self.context_encoder(video_frames, mask)
        
        # Extract visible features (unmasked regions)
        visible_mask = ~mask  # Invert mask for visible regions
        visible_features = []
        for b in range(B):
            visible_idx = visible_mask[b]
            if visible_idx.any():
                visible_feat = context_features[b][visible_idx]
                visible_features.append(visible_feat)
            else:
                # If no visible patches, use all features
                visible_features.append(context_features[b])
        
        # Pad to same length for batch processing
        max_visible = max(feat.size(0) for feat in visible_features)
        padded_visible_features = []
        for feat in visible_features:
            if feat.size(0) < max_visible:
                padding = torch.zeros(max_visible - feat.size(0), feat.size(1), device=feat.device)
                feat = torch.cat([feat, padding], dim=0)
            padded_visible_features.append(feat)
        
        visible_features = torch.stack(padded_visible_features, dim=0)  # (B, max_visible, hidden_size)
        
        # Predict masked regions
        predicted_features = self.predictor(visible_features, mask)
        
        # Extract target features for masked regions
        masked_target_features = []
        for b in range(B):
            masked_idx = mask[b]
            if masked_idx.any():
                target_feat = target_features[b][masked_idx]
                masked_target_features.append(target_feat)
            else:
                # If no masked patches, use dummy features
                dummy_feat = torch.zeros(1, target_features.size(-1), device=target_features.device)
                masked_target_features.append(dummy_feat)
        
        # Pad target features
        max_masked = max(feat.size(0) for feat in masked_target_features)
        padded_target_features = []
        for feat in masked_target_features:
            if feat.size(0) < max_masked:
                padding = torch.zeros(max_masked - feat.size(0), feat.size(1), device=feat.device)
                feat = torch.cat([feat, padding], dim=0)
            padded_target_features.append(feat)
        
        target_features_masked = torch.stack(padded_target_features, dim=0)
        
        return {
            'predicted_features': predicted_features,
            'target_features': target_features_masked,
            'context_features': visible_features,
            'mask': mask,
            'full_features': target_features,
        }
    
    def compute_loss(
        self,
        video_frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute VJEPA-2 prediction loss following the official implementation.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask for training
            
        Returns:
            Prediction loss
        """
        outputs = self.forward(video_frames, mask)
        
        predicted = outputs['predicted_features']  # (B, num_masked, hidden_size)
        target = outputs['target_features']        # (B, num_masked, hidden_size)
        
        # Compute loss following official V-JEPA2 implementation
        # Use L1 loss with exponentiation: |pred - target|^loss_exp
        loss = torch.mean(torch.abs(predicted - target) ** self.loss_exp) / self.loss_exp
        
        return loss
    
    def update_target_encoder(self, momentum: Optional[float] = None):
        """
        Update target encoder with EMA from context encoder (like official VJEPA-2).
        
        Args:
            momentum: Momentum coefficient for EMA update (uses self.momentum if None)
        """
        if momentum is None:
            momentum = self.momentum
            
        with torch.no_grad():
            for target_param, context_param in zip(
                self.target_encoder.parameters(), 
                self.context_encoder.parameters()
            ):
                # EMA update: target = momentum * target + (1 - momentum) * context
                target_param.data.mul_(momentum).add_(context_param.data, alpha=1 - momentum)
    
    def predict_masked_regions(
        self,
        video_frames: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict features for masked regions.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Boolean mask indicating which patches to predict
            
        Returns:
            Predicted features for masked regions
        """
        outputs = self.forward(video_frames, mask)
        return outputs['predicted_features']
    
    def encode_video(
        self,
        video_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode video frames into patch features.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            
        Returns:
            Encoded video features (B, num_patches, hidden_size)
        """
        return self.context_encoder(video_frames, mask=None)


def create_vjepa2_model(
    config: Optional[VJEPA2WorldModelConfig] = None,
) -> VJEPA2WorldModel:
    """
    Create a VJEPA-2 world model with default or custom configuration.
    
    Args:
        config: Optional configuration. If None, uses default config.
        
    Returns:
        VJEPA-2 world model instance
    """
    if config is None:
        config = VJEPA2WorldModelConfig()
    
    return VJEPA2WorldModel(config)


# Example usage and testing
if __name__ == "__main__":
    # Create model with default config
    model = create_vjepa2_model()
    
    # Test with dummy video data
    batch_size = 2
    num_frames = 16
    height, width = 224, 224
    channels = 3
    
    # Create dummy video frames
    video_frames = torch.randn(batch_size, num_frames, height, width, channels)
    
    # Forward pass
    print("Testing VJEPA-2 model...")
    loss = model.compute_loss(video_frames)
    print(f"Loss: {loss.item():.4f}")
    
    # Test encoding
    features = model.encode_video(video_frames)
    print(f"Encoded features shape: {features.shape}")
    
    # Test prediction
    mask = create_video_mask(
        video_frames.shape,
        mask_ratio=0.75,
        device=video_frames.device
    )
    predicted = model.predict_masked_regions(video_frames, mask)
    print(f"Predicted features shape: {predicted.shape}")
    
    print("VJEPA-2 model test completed successfully!") 