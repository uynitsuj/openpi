"""
Base classes for world models that predict future or masked video frames.

This module provides abstract base classes for world models that work with sequential video data,
distinct from the action prediction models in the main model.py file.
"""

import abc
import dataclasses
import enum
import logging
from typing import Generic, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.nnx as nnx

from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class WorldModelType(enum.Enum):
    """Supported world model types."""
    VJEPA2 = "vjepa2"


@dataclasses.dataclass
class WorldModelInput:
    """Input data for world models."""
    video_frames: Union[jnp.ndarray, at.Array]  # (B, T, H, W, C)
    mask: Union[jnp.ndarray, at.Array]  # (B, T, H, W) or (B, num_patches)
    camera_names: list[str]


@dataclasses.dataclass
class WorldModelOutput:
    """Output data from world models."""
    predicted_features: Union[jnp.ndarray, at.Array]  # (B, T, H, W, C) or (B, num_patches, D)
    reconstruction_loss: Union[jnp.ndarray, at.Array]  # scalar
    mask_ratio: Union[jnp.ndarray, at.Array]  # scalar


@at.typecheck  
@struct.dataclass
class VideoObservation(Generic[ArrayT]):
    """
    Holds video observations for world models.
    
    Unlike the standard Observation class, this focuses on sequential video data
    without action-specific fields.
    """
    
    # Sequential video frames: [batch, num_frames, height, width, channels]
    video_frames: at.Float[ArrayT, "*b t h w c"]
    
    # Frame masks indicating valid frames: [batch, num_frames]
    frame_masks: at.Bool[ArrayT, "*b t"] | None = None
    
    # Spatial masks for each frame (for masked modeling): [batch, num_frames, num_patches]
    spatial_masks: at.Bool[ArrayT, "*b t p"] | None = None
    
    # Temporal masks across frames: [batch, num_frames]
    temporal_masks: at.Bool[ArrayT, "*b t"] | None = None
    
    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "VideoObservation[ArrayT]":
        """Create VideoObservation from dictionary format."""
        # Convert uint8 images to float32 in [-1, 1] range
        video_frames = data["video_frames"]
        if video_frames.dtype == np.uint8:
            video_frames = video_frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            
        return cls(
            video_frames=video_frames,
            frame_masks=data.get("frame_masks"),
            spatial_masks=data.get("spatial_masks"),
            temporal_masks=data.get("temporal_masks"),
        )


@at.typecheck
@struct.dataclass  
class VideoTarget(Generic[ArrayT]):
    """
    Target video frames for prediction.
    
    This can represent future frames to predict or masked regions to reconstruct.
    """
    
    # Target frames: [batch, num_target_frames, height, width, channels]
    target_frames: at.Float[ArrayT, "*b t h w c"]
    
    # Mask indicating which targets are valid: [batch, num_target_frames]
    target_masks: at.Bool[ArrayT, "*b t"] | None = None
    
    # Indices of target frames in the original sequence: [batch, num_target_frames]
    target_indices: at.Int[ArrayT, "*b t"] | None = None


@dataclasses.dataclass(frozen=True)
class BaseWorldModelConfig(abc.ABC):
    """Base configuration for world models."""
    
    # Video input dimensions
    num_frames: int
    frame_height: int
    frame_width: int
    num_channels: int = 3
    
    # Patch dimensions for tokenization
    patch_height: int = 16
    patch_width: int = 16
    temporal_patch_size: int = 2
    
    # Model dimensions
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    @property
    @abc.abstractmethod
    def model_type(self) -> WorldModelType:
        """The world model type."""
        
    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseWorldModel":
        """Create a new world model instance."""
        
    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[VideoObservation, VideoTarget]:
        """Returns input specifications for the model."""
        
    def fake_video_obs(self, batch_size: int = 1) -> VideoObservation:
        """Create fake video observations for testing."""
        obs_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)
        
    def fake_video_target(self, batch_size: int = 1) -> VideoTarget:
        """Create fake video targets for testing."""
        _, target_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), target_spec)


@dataclasses.dataclass
class BaseWorldModel(nnx.Module, abc.ABC):
    """Base class for world models that predict video sequences."""
    
    num_frames: int
    frame_height: int
    frame_width: int
    num_channels: int
    hidden_size: int
    
    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        video_obs: VideoObservation,
        video_target: VideoTarget,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """
        Compute prediction loss between model output and target.
        
        Args:
            rng: Random key for any stochastic operations
            video_obs: Input video observations
            video_target: Target video frames to predict
            train: Whether in training mode
            
        Returns:
            Loss values for each batch element
        """
        
    @abc.abstractmethod
    def predict_video(
        self,
        rng: at.KeyArrayLike,
        video_obs: VideoObservation,
        *,
        num_pred_frames: int = 1,
    ) -> VideoTarget:
        """
        Predict future video frames given observed frames.
        
        Args:
            rng: Random key
            video_obs: Input video observations
            num_pred_frames: Number of frames to predict
            
        Returns:
            Predicted video frames
        """
        
    @abc.abstractmethod
    def reconstruct_masked(
        self,
        rng: at.KeyArrayLike,
        video_obs: VideoObservation,
    ) -> VideoTarget:
        """
        Reconstruct masked regions in the input video.
        
        Args:
            rng: Random key  
            video_obs: Input video with masks
            
        Returns:
            Reconstructed video frames
        """ 