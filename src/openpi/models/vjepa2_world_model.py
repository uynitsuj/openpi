"""
VJEPA-2 World Model Implementation

This module implements a VJEPA-2 inspired world model for video understanding and prediction.
Based on the Video Joint-Embedding Predictive Architecture (V-JEPA) from Meta AI.

The model uses a vision encoder + predictor architecture for masked video modeling,
where visible patches are encoded and used to predict masked regions in representation space.
"""

import dataclasses
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers.models.vit import ViTModel, ViTConfig
import math

logger = logging.getLogger("openpi")


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
    B, T, H, W, C = video_shape
    
    # Calculate number of patches
    num_patches_h = H // patch_size[0]
    num_patches_w = W // patch_size[1]
    num_patches_t = T // temporal_patch_size
    total_patches = num_patches_t * num_patches_h * num_patches_w
    
    # Number of patches to mask
    num_masked = int(mask_ratio * total_patches)
    
    masks = []
    for b in range(B):
        # Create mask for this batch element
        mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
        
        # Randomly select patches to mask in blocks
        masked_patches = 0
        while masked_patches < num_masked:
            # Random starting position
            t_start = torch.randint(0, num_patches_t, (1,)).item()
            h_start = torch.randint(0, num_patches_h, (1,)).item()
            w_start = torch.randint(0, num_patches_w, (1,)).item()
            
            # Block boundaries
            t_end = min(t_start + block_size, num_patches_t)
            h_end = min(h_start + block_size, num_patches_h)  
            w_end = min(w_start + block_size, num_patches_w)
            
            # Mark patches in this block as masked
            for t in range(t_start, t_end):
                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        patch_idx = t * (num_patches_h * num_patches_w) + h * num_patches_w + w
                        if not mask[patch_idx]:
                            mask[patch_idx] = True
                            masked_patches += 1
                            if masked_patches >= num_masked:
                                break
                    if masked_patches >= num_masked:
                        break
                if masked_patches >= num_patches:
                    break
        
        masks.append(mask)
    
    return torch.stack(masks, dim=0)


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
        
        # Temporal positional embeddings
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, 1000, hidden_size) * 0.02  # Support up to 1000 frames
        )
        
        # Learnable mask tokens for masked regions
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
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
            T = T + pad_frames
            
        # Group temporal patches
        num_temporal_patches = T // self.temporal_patch_size
        patch_embeddings = patch_embeddings.view(
            B, num_temporal_patches, self.temporal_patch_size, patches_per_frame, self.hidden_size
        )
        # Average over temporal dimension
        patch_embeddings = patch_embeddings.mean(dim=2)  # (B, num_temporal_patches, patches_per_frame, hidden_size)
        
        # Flatten spatial and temporal dimensions
        total_patches = num_temporal_patches * patches_per_frame
        patch_embeddings = patch_embeddings.view(B, total_patches, self.hidden_size)
        
        # Add temporal positional embeddings
        temporal_pos = self.temporal_pos_embedding[:, :total_patches, :]
        patch_embeddings = patch_embeddings + temporal_pos
        
        # Apply masking if provided
        if mask is not None:
            # Replace masked patches with mask tokens
            mask_tokens = self.mask_token.expand(B, total_patches, self.hidden_size)
            patch_embeddings = torch.where(
                mask.unsqueeze(-1).expand(-1, -1, self.hidden_size),
                mask_tokens,
                patch_embeddings
            )
            
        return patch_embeddings


class VideoTransformerPredictor(nn.Module):
    """
    Transformer predictor for VJEPA-2 world model.
    
    Takes encoded visible patches and predicts features for masked regions.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_mask_tokens: int = 100,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_mask_tokens = num_mask_tokens
        
        # Learnable mask tokens for prediction
        self.mask_tokens = nn.Parameter(torch.randn(1, num_mask_tokens, hidden_size) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        context_features: torch.Tensor,  # (B, num_context_patches, hidden_size)
        mask: torch.Tensor,  # (B, total_patches) boolean mask
    ) -> torch.Tensor:
        """
        Predict features for masked regions.
        
        Args:
            context_features: Features from visible patches
            mask: Boolean mask indicating which patches were masked
            
        Returns:
            Predicted features for masked regions (B, num_masked, hidden_size)
        """
        B, num_context, hidden_size = context_features.shape
        
        # Get number of masked patches
        num_masked = mask.sum(dim=1).max().item()
        
        # Warn if we're exceeding the number of available mask tokens
        if num_masked > self.num_mask_tokens:
            import warnings
            warnings.warn(f"Number of masked patches ({num_masked}) exceeds available mask tokens ({self.num_mask_tokens}). "
                         f"Truncating to {self.num_mask_tokens} masked patches.")
        
        # Ensure we don't exceed the number of available mask tokens
        num_masked = min(num_masked, self.num_mask_tokens)
        
        # Create mask tokens
        mask_tokens = self.mask_tokens[:, :num_masked, :].expand(B, -1, -1)
        
        # Concatenate context and mask tokens
        predictor_input = torch.cat([context_features, mask_tokens], dim=1)
        
        # Create attention mask (context tokens can attend to each other, mask tokens attend to context)
        seq_len = predictor_input.size(1)
        attention_mask = torch.ones(seq_len, seq_len, device=predictor_input.device)
        
        # Mask tokens can only attend to context (not to other mask tokens)
        attention_mask[num_context:, num_context:] = 0
        
        # Apply transformer
        predicted_features = self.transformer(
            predictor_input,
            mask=attention_mask,
        )
        
        # Extract predictions for masked regions
        predicted_masked = predicted_features[:, num_context:, :]
        
        # Apply layer norm
        predicted_masked = self.layer_norm(predicted_masked)
        
        return predicted_masked


@dataclasses.dataclass(frozen=True)
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
    
    # Predictor parameters
    predictor_hidden_size: int = 384
    predictor_num_layers: int = 6
    predictor_num_heads: int = 6
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    predictor_num_mask_tokens: int = 800  # Increased to handle up to 90% masking of large videos
    
    # Pretrained model
    use_pretrained_encoder: bool = True
    pretrained_model: str = "google/vit-base-patch16-224"


class VJEPA2WorldModel(nn.Module):
    """
    VJEPA-2 World Model for video understanding and prediction.
    
    This model implements the Video Joint-Embedding Predictive Architecture (V-JEPA)
    for self-supervised learning from video. It uses a vision encoder + predictor
    architecture to predict masked regions in video sequences.
    """
    
    def __init__(self, config: VJEPA2WorldModelConfig):
        super().__init__()
        
        self.config = config
        
        # Video encoder
        self.encoder = VideoTransformerEncoder(
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
        
        # Predictor
        self.predictor = VideoTransformerPredictor(
            hidden_size=config.predictor_hidden_size,
            num_layers=config.predictor_num_layers,
            num_heads=config.predictor_num_heads,
            mlp_ratio=config.predictor_mlp_ratio,
            dropout=config.predictor_dropout,
            num_mask_tokens=config.predictor_num_mask_tokens,
        )
        
        # Projection layers to align encoder and predictor dimensions
        if config.encoder_hidden_size != config.predictor_hidden_size:
            self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.predictor_hidden_size)
            self.target_proj = nn.Linear(config.encoder_hidden_size, config.predictor_hidden_size)
        else:
            self.encoder_proj = nn.Identity()
            self.target_proj = nn.Identity()
    
    def forward(
        self,
        video_frames: torch.Tensor,  # (B, T, H, W, C)
        mask: Optional[torch.Tensor] = None,  # (B, num_patches) boolean mask
    ) -> dict:
        """
        Forward pass through VJEPA-2 model.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask for training (True = masked)
            
        Returns:
            Dictionary containing model outputs
        """
        B, T, H, W, C = video_frames.shape
        
        # Create mask if not provided
        if mask is None:
            mask = create_video_mask(
                video_frames.shape,
                patch_size=(self.config.patch_size, self.config.patch_size),
                temporal_patch_size=self.config.temporal_patch_size,
                mask_ratio=0.75, # This parameter is removed from config, so it's hardcoded here
                block_size=4, # This parameter is removed from config, so it's hardcoded here
                device=video_frames.device,
            )
        
        # Encode full video (including masked regions for targets)
        full_features = self.encoder(video_frames, mask=None)
        
        # Encode visible regions only (using provided mask)
        visible_features = self.encoder(video_frames, mask=mask)
        
        # Get context features (non-masked patches)
        context_mask = ~mask  # Invert mask for context
        context_features = visible_features[context_mask.unsqueeze(-1).expand(-1, -1, visible_features.size(-1))]
        context_features = context_features.view(B, -1, visible_features.size(-1))
        
        # Project to predictor dimension
        context_features = self.encoder_proj(context_features)
        
        # Predict masked regions
        predicted_features = self.predictor(context_features, mask)
        
        # Get target features for masked regions (using the same mask)
        target_features = full_features[mask.unsqueeze(-1).expand(-1, -1, full_features.size(-1))]
        target_features = target_features.view(B, -1, full_features.size(-1))
        target_features = self.target_proj(target_features)
        
        return {
            'predicted_features': predicted_features,
            'target_features': target_features,
            'context_features': context_features,
            'mask': mask,
            'full_features': full_features,
        }
    
    def compute_loss(
        self,
        video_frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute VJEPA-2 prediction loss.
        
        Args:
            video_frames: Input video frames (B, T, H, W, C)
            mask: Optional mask for training
            
        Returns:
            Prediction loss
        """
        outputs = self.forward(video_frames, mask)
        
        predicted = outputs['predicted_features']
        target = outputs['target_features']
        
        # L2 loss between predicted and target features
        loss = F.mse_loss(predicted, target, reduction='mean')
        
        return loss
    
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
        return self.encoder(video_frames, mask=None)
    
    def predict_future_frames(
        self,
        past_frames: torch.Tensor,  # (B, T_past, H, W, C)
        num_future_frames: int = 8,
    ) -> torch.Tensor:
        """
        Predict future frames causally from past frames.
        
        Args:
            past_frames: Input past frames (B, T_past, H, W, C)
            num_future_frames: Number of future frames to predict
            
        Returns:
            Predicted future frames (B, T_future, H, W, C)
        """
        B, T_past, H, W, C = past_frames.shape
        
        # Encode past frames
        past_features = self.encoder(past_frames, mask=None)
        
        # Create future frame mask (all future patches are masked)
        total_patches = (H // self.config.patch_size) * (W // self.config.patch_size) * (T_past // self.config.temporal_patch_size)
        future_patches = (H // self.config.patch_size) * (W // self.config.patch_size) * (num_future_frames // self.config.temporal_patch_size)
        
        # Create mask where future patches are True (masked)
        mask = torch.zeros(B, total_patches + future_patches, dtype=torch.bool, device=past_frames.device)
        mask[:, total_patches:] = True  # Mask all future patches
        
        # Project past features to predictor dimension
        past_features_proj = self.encoder_proj(past_features)
        
        # Predict future features
        predicted_future_features = self.predictor(past_features_proj, mask)
        
        # Convert features back to image space (you'll need a decoder)
        # This is a simplified version - you'll need to implement proper decoding
        predicted_frames = self._features_to_frames(predicted_future_features, num_future_frames)
        
        return predicted_frames
    
    def _features_to_frames(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Convert predicted features back to image frames.
        This is a placeholder - you'll need to implement proper decoding.
        """
        # This is a simplified implementation
        # In practice, you'd need a proper decoder network
        B, num_patches, hidden_size = features.shape
        
        # Reshape to spatial dimensions
        patches_per_frame = (self.config.image_size // self.config.patch_size) ** 2
        num_temporal_patches = num_patches // patches_per_frame
        
        features = features.view(B, num_temporal_patches, patches_per_frame, hidden_size)
        
        # Simple linear projection to RGB values (simplified)
        rgb_proj = nn.Linear(hidden_size, self.config.patch_size ** 2 * 3).to(features.device)
        rgb_patches = rgb_proj(features)  # (B, T, patches_per_frame, patch_size^2 * 3)
        
        # Reshape to image format
        rgb_patches = rgb_patches.view(B, num_temporal_patches, patches_per_frame, 
                                     self.config.patch_size, self.config.patch_size, 3)
        
        # Reconstruct images (simplified)
        # In practice, you'd use a proper decoder network
        return rgb_patches.mean(dim=2)  # Average over patches (simplified)


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