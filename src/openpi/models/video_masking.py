"""
Video Masking Utilities for World Models

This module provides various masking strategies for video sequences, inspired by VJEPA-2
and other video understanding models. These masking strategies are used for self-supervised
learning from video data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union, Dict, Any
from enum import Enum
import math
import random


class MaskingStrategy(Enum):
    """Different masking strategies for video sequences."""
    RANDOM = "random"
    BLOCK = "block"
    TUBE = "tube"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    RUNNING_CELL = "running_cell"
    MULTI_SCALE = "multi_scale"  # New: Multi-scale masking like official VJEPA2


class MultiScaleMaskConfig:
    """Configuration for multi-scale masking similar to official VJEPA2."""
    
    def __init__(
        self,
        spatial_scale: Tuple[float, float] = (0.15, 0.15),
        temporal_scale: Tuple[float, float] = (0.1, 0.8),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        num_blocks: int = 8,
        max_temporal_keep: float = 1.0,
        max_keep: Optional[int] = None,
        full_complement: bool = False,
        inv_block: bool = False,
    ):
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_temporal_keep = max_temporal_keep
        self.max_keep = max_keep
        self.full_complement = full_complement
        self.inv_block = inv_block


class VideoMaskGenerator:
    """
    Generator for various video masking patterns.
    
    This class implements different masking strategies for video sequences,
    including the block masking strategy used in VJEPA-2.
    """
    
    def __init__(
        self,
        num_frames: int,
        image_size: Tuple[int, int],  # (H, W)
        patch_size: Tuple[int, int] = (16, 16),  # (patch_h, patch_w)
        temporal_patch_size: int = 2,
        mask_ratio: float = 0.75,
        masking_strategy: MaskingStrategy = MaskingStrategy.BLOCK,
        num_masked_patches: int | None = None,
        device: str = "cpu",
        multi_scale_config: Optional[MultiScaleMaskConfig] = None,
    ):
        """
        Initialize the video mask generator.
        
        Args:
            num_frames: Number of frames in the video
            image_size: Size of input video (H, W)
            patch_size: Size of spatial patches (height, width)
            temporal_patch_size: Size of temporal patches
            mask_ratio: Ratio of patches to mask
            masking_strategy: Masking strategy to use
            num_masked_patches: If provided, always mask exactly this many patches
            device: Device to create masks on
            multi_scale_config: Configuration for multi-scale masking
        """
        self.input_size = (num_frames, image_size[0], image_size[1])
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.mask_ratio = mask_ratio
        self.strategy = masking_strategy
        self.device = device
        self.multi_scale_config = multi_scale_config
        
        # Calculate patch dimensions
        self.num_patches_h = self.input_size[1] // patch_size[0]
        self.num_patches_w = self.input_size[2] // patch_size[1]
        self.num_patches_t = self.input_size[0] // temporal_patch_size
        self.total_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        
        # Number of patches to mask
        if num_masked_patches is not None:
            self.num_masked = num_masked_patches
        else:
            self.num_masked = int(mask_ratio * self.total_patches)
        
        # Multi-scale masking parameters
        if self.multi_scale_config is not None:
            self.max_context_duration = max(
                1, int(self.num_patches_t * self.multi_scale_config.max_temporal_keep)
            )
        else:
            self.max_context_duration = self.num_patches_t
        
    def generate_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate video masks for a batch.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Boolean mask tensor of shape (batch_size, total_patches)
            where True indicates masked patches
        """
        if self.strategy == MaskingStrategy.RANDOM:
            return self._generate_random_mask(batch_size)
        elif self.strategy == MaskingStrategy.BLOCK:
            return self._generate_block_mask(batch_size)
        elif self.strategy == MaskingStrategy.TUBE:
            return self._generate_tube_mask(batch_size)
        elif self.strategy == MaskingStrategy.TEMPORAL:
            return self._generate_temporal_mask(batch_size)
        elif self.strategy == MaskingStrategy.SPATIAL:
            return self._generate_spatial_mask(batch_size)
        elif self.strategy == MaskingStrategy.RUNNING_CELL:
            return self._generate_running_cell_mask(batch_size)
        elif self.strategy == MaskingStrategy.MULTI_SCALE:
            return self._generate_multi_scale_mask(batch_size)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")
    
    def _generate_random_mask(self, batch_size: int) -> torch.Tensor:
        """Generate random masks."""
        masks = []
        for _ in range(batch_size):
            # Random selection of patches to mask
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            masked_indices = torch.randperm(self.total_patches, device=self.device)[:self.num_masked]
            mask[masked_indices] = True
            masks.append(mask)
        return torch.stack(masks, dim=0)
    
    def _generate_block_mask(self, batch_size: int, block_size: int = 4) -> torch.Tensor:
        """
        Generate block masks similar to VJEPA-2.
        
        This creates spatiotemporal blocks of masked patches.
        """
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            masked_patches = 0
            
            # Keep trying to place blocks until we reach the target number
            max_attempts = 1000
            attempt = 0
            while masked_patches < self.num_masked and attempt < max_attempts:
                # Random starting position
                t_start = torch.randint(0, self.num_patches_t, (1,)).item()
                h_start = torch.randint(0, self.num_patches_h, (1,)).item()
                w_start = torch.randint(0, self.num_patches_w, (1,)).item()
                
                # Block boundaries
                t_end = min(t_start + block_size, self.num_patches_t)
                h_end = min(h_start + block_size, self.num_patches_h)
                w_end = min(w_start + block_size, self.num_patches_w)
                
                # Mark patches in this block as masked
                for t in range(t_start, t_end):
                    for h in range(h_start, h_end):
                        for w in range(w_start, w_end):
                            if masked_patches >= self.num_masked:
                                break
                            patch_idx = self._get_patch_index(t, h, w)
                            if not mask[patch_idx]:
                                mask[patch_idx] = True
                                masked_patches += 1
                        if masked_patches >= self.num_masked:
                            break
                    if masked_patches >= self.num_masked:
                        break
                
                attempt += 1
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_multi_scale_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate multi-scale masks similar to official VJEPA2.
        
        This creates multiple blocks with variable sizes, aspect ratios, and scales.
        Now respects the mask_ratio parameter by generating blocks until target ratio is achieved.
        """
        if self.multi_scale_config is None:
            # Fallback to default multi-scale config
            self.multi_scale_config = MultiScaleMaskConfig()
        
        masks = []
        for _ in range(batch_size):
            # Create mask for this sample
            mask = torch.ones((self.num_patches_t, self.num_patches_h, self.num_patches_w), 
                            dtype=torch.int32, device=self.device)
            
            # Generate blocks until we reach the desired mask ratio
            target_masked_patches = int(self.mask_ratio * self.total_patches)
            current_masked_patches = 0
            max_attempts = self.multi_scale_config.num_blocks * 2  # Limit attempts to avoid infinite loop
            
            for attempt in range(max_attempts):
                if current_masked_patches >= target_masked_patches:
                    break
                    
                # Sample block size using multi-scale parameters
                block_size = self._sample_block_size()
                
                # Sample block mask
                block_mask = self._sample_block_mask(block_size)
                
                # Apply block mask (multiply to combine blocks)
                new_mask = mask * block_mask
                
                # Count how many new patches would be masked
                new_masked_patches = (new_mask == 0).sum().item()
                
                # If this doesn't exceed our target, apply the mask
                if new_masked_patches <= target_masked_patches:
                    mask = new_mask
                    current_masked_patches = new_masked_patches
                else:
                    # If we would exceed target, stop here
                    break
            
            # Convert to boolean mask and flatten
            mask = (mask == 0).flatten()  # 0 = masked, 1 = visible
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _sample_block_size(self) -> Tuple[int, int, int]:
        """
        Sample block size using multi-scale parameters.
        
        Returns:
            Tuple of (temporal_size, height, width)
        """
        if self.multi_scale_config is None:
            return (4, 4, 4)  # Default block size
        
        # Sample temporal block mask scale
        min_t, max_t = self.multi_scale_config.temporal_scale
        temporal_mask_scale = min_t + random.random() * (max_t - min_t)
        t = max(1, int(self.num_patches_t * temporal_mask_scale))
        
        # Sample spatial block mask scale
        min_s, max_s = self.multi_scale_config.spatial_scale
        spatial_mask_scale = min_s + random.random() * (max_s - min_s)
        spatial_num_keep = int(self.num_patches_h * self.num_patches_w * spatial_mask_scale)
        
        # Sample block aspect-ratio
        min_ar, max_ar = self.multi_scale_config.aspect_ratio
        aspect_ratio = min_ar + random.random() * (max_ar - min_ar)
        
        # Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.num_patches_h)
        w = min(w, self.num_patches_w)
        
        return (t, h, w)
    
    def _sample_block_mask(self, block_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Sample a single block mask.
        
        Args:
            block_size: Tuple of (temporal_size, height, width)
            
        Returns:
            Block mask tensor of shape (num_patches_t, num_patches_h, num_patches_w)
        """
        t, h, w = block_size
        
        # Random starting position
        top = random.randint(0, self.num_patches_h - h + 1)
        left = random.randint(0, self.num_patches_w - w + 1)
        start = random.randint(0, self.num_patches_t - t + 1)
        
        # Create mask (1 = visible, 0 = masked)
        mask = torch.ones((self.num_patches_t, self.num_patches_h, self.num_patches_w), 
                         dtype=torch.int32, device=self.device)
        
        # Apply block mask
        mask[start:start + t, top:top + h, left:left + w] = 0
        
        # Context mask will only span the first X frames
        if self.max_context_duration < self.num_patches_t:
            mask[self.max_context_duration:, :, :] = 0
        
        return mask
    
    def _generate_tube_mask(self, batch_size: int, tube_size: int = 3) -> torch.Tensor:
        """
        Generate tube masks that extend through time.
        
        This creates tubes of masked patches that extend through the temporal dimension.
        """
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            masked_patches = 0
            
            while masked_patches < self.num_masked:
                # Random spatial position
                h_center = torch.randint(0, self.num_patches_h, (1,)).item()
                w_center = torch.randint(0, self.num_patches_w, (1,)).item()
                
                # Create tube extending through time
                h_start = max(0, h_center - tube_size // 2)
                h_end = min(self.num_patches_h, h_center + tube_size // 2 + 1)
                w_start = max(0, w_center - tube_size // 2)
                w_end = min(self.num_patches_w, w_center + tube_size // 2 + 1)
                
                for t in range(self.num_patches_t):
                    for h in range(h_start, h_end):
                        for w in range(w_start, w_end):
                            if masked_patches >= self.num_masked:
                                break
                            patch_idx = self._get_patch_index(t, h, w)
                            if not mask[patch_idx]:
                                mask[patch_idx] = True
                                masked_patches += 1
                        if masked_patches >= self.num_masked:
                            break
                    if masked_patches >= self.num_masked:
                        break
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_temporal_mask(self, batch_size: int) -> torch.Tensor:
        """Generate temporal masks that mask entire frames."""
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            # Randomly select frames to mask
            num_frames_to_mask = int(self.mask_ratio * self.num_patches_t)
            masked_frames = torch.randperm(self.num_patches_t, device=self.device)[:num_frames_to_mask]
            
            for frame_idx in masked_frames:
                start_idx = frame_idx * self.num_patches_h * self.num_patches_w
                end_idx = (frame_idx + 1) * self.num_patches_h * self.num_patches_w
                mask[start_idx:end_idx] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_spatial_mask(self, batch_size: int) -> torch.Tensor:
        """Generate spatial masks that mask spatial regions across all frames."""
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            # Randomly select spatial regions to mask
            num_spatial_patches = self.num_patches_h * self.num_patches_w
            num_spatial_to_mask = int(self.mask_ratio * num_spatial_patches)
            masked_spatial = torch.randperm(num_spatial_patches, device=self.device)[:num_spatial_to_mask]
            
            for spatial_idx in masked_spatial:
                h = spatial_idx // self.num_patches_w
                w = spatial_idx % self.num_patches_w
                
                # Mask this spatial position across all frames
                for t in range(self.num_patches_t):
                    patch_idx = self._get_patch_index(t, h, w)
                    mask[patch_idx] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_running_cell_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate running cell masks that create moving masked regions.
        
        This creates masks that move through time, simulating motion.
        """
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            # Create a moving cell
            cell_size = 3
            start_h = torch.randint(0, self.num_patches_h - cell_size, (1,)).item()
            start_w = torch.randint(0, self.num_patches_w - cell_size, (1,)).item()
            
            # Move the cell through time
            for t in range(self.num_patches_t):
                # Add some randomness to the movement
                h_offset = torch.randint(-1, 2, (1,)).item()
                w_offset = torch.randint(-1, 2, (1,)).item()
                
                h = max(0, min(self.num_patches_h - cell_size, start_h + h_offset))
                w = max(0, min(self.num_patches_w - cell_size, start_w + w_offset))
                
                # Mask the cell at this position
                for dh in range(cell_size):
                    for dw in range(cell_size):
                        patch_idx = self._get_patch_index(t, h + dh, w + dw)
                        mask[patch_idx] = True
                
                start_h, start_w = h, w
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _get_patch_index(self, t: int, h: int, w: int) -> int:
        """Convert 3D patch coordinates to 1D index."""
        return t * (self.num_patches_h * self.num_patches_w) + h * self.num_patches_w + w
    
    def get_patch_coordinates(self, patch_idx: int) -> Tuple[int, int, int]:
        """Convert 1D patch index to 3D coordinates."""
        t = patch_idx // (self.num_patches_h * self.num_patches_w)
        remainder = patch_idx % (self.num_patches_h * self.num_patches_w)
        h = remainder // self.num_patches_w
        w = remainder % self.num_patches_w
        return (t, h, w)


class AdaptiveMaskGenerator:
    """
    Adaptive mask generator that can switch between different strategies.
    
    This class provides curriculum learning capabilities by changing
    masking strategies over time.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        patch_size: Tuple[int, int] = (16, 16),
        temporal_patch_size: int = 2,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.device = device
        
        # Create mask generators for different strategies
        self.generators = {}
        for strategy in MaskingStrategy:
            self.generators[strategy] = VideoMaskGenerator(
                num_frames=input_size[0],
                image_size=(input_size[1], input_size[2]),
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                masking_strategy=strategy,
                device=device,
            )
    
    def generate_mask(
        self,
        batch_size: int,
        strategy: MaskingStrategy = MaskingStrategy.BLOCK,
        mask_ratio: float = 0.75,
    ) -> torch.Tensor:
        """Generate mask using specified strategy."""
        generator = self.generators[strategy]
        generator.mask_ratio = mask_ratio
        return generator.generate_mask(batch_size)
    
    def generate_curriculum_mask(
        self,
        batch_size: int,
        training_step: int,
        curriculum_schedule: List[Tuple[int, MaskingStrategy, float]],
    ) -> torch.Tensor:
        """
        Generate mask using curriculum learning schedule.
        
        Args:
            batch_size: Size of the batch
            training_step: Current training step
            curriculum_schedule: List of (step, strategy, ratio) tuples
            
        Returns:
            Generated mask tensor
        """
        # Find the appropriate strategy for current step
        current_strategy = MaskingStrategy.BLOCK
        current_ratio = 0.75
        
        for step, strategy, ratio in curriculum_schedule:
            if training_step >= step:
                current_strategy = strategy
                current_ratio = ratio
        
        return self.generate_mask(batch_size, current_strategy, current_ratio)


def create_video_mask(
    video_shape: Tuple[int, int, int, int, int],  # (B, T, H, W, C)
    patch_size: Tuple[int, int] = (16, 16),
    temporal_patch_size: int = 2,
    mask_ratio: float = 0.75,
    strategy: MaskingStrategy = MaskingStrategy.BLOCK,
    device: str = "cpu",
    multi_scale_config: Optional[MultiScaleMaskConfig] = None,
) -> torch.Tensor:
    """
    Create video masks for a batch of videos.
    
    Args:
        video_shape: Shape of input video (B, T, H, W, C)
        patch_size: Size of spatial patches (height, width)
        temporal_patch_size: Size of temporal patches
        mask_ratio: Ratio of patches to mask
        strategy: Masking strategy to use
        device: Device to create masks on
        multi_scale_config: Configuration for multi-scale masking
        
    Returns:
        Boolean mask tensor of shape (B, total_patches)
    """
    batch_size, num_frames, height, width, channels = video_shape
    
    generator = VideoMaskGenerator(
        num_frames=num_frames,
        image_size=(height, width),
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        mask_ratio=mask_ratio,
        masking_strategy=strategy,
        device=device,
        multi_scale_config=multi_scale_config,
    )
    
    return generator.generate_mask(batch_size)


def create_multi_scale_mask_config(
    spatial_scale: Tuple[float, float] = (0.15, 0.15),
    temporal_scale: Tuple[float, float] = (1.0, 1.0),
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    num_blocks: int = 8,
    max_temporal_keep: float = 1.0,
    max_keep: Optional[int] = None,
    full_complement: bool = False,
    inv_block: bool = False,
) -> MultiScaleMaskConfig:
    """
    Create multi-scale mask configuration similar to official VJEPA2.
    
    Args:
        spatial_scale: Range for spatial mask scale (min, max)
        temporal_scale: Range for temporal mask scale (min, max)
        aspect_ratio: Range for block aspect ratio (min, max)
        num_blocks: Number of blocks to generate
        max_temporal_keep: Maximum fraction of temporal frames to keep
        max_keep: Maximum number of patches to keep (optional)
        full_complement: Whether to use full complement masking
        inv_block: Whether to use inverse block masking
        
    Returns:
        MultiScaleMaskConfig instance
    """
    return MultiScaleMaskConfig(
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
        aspect_ratio=aspect_ratio,
        num_blocks=num_blocks,
        max_temporal_keep=max_temporal_keep,
        max_keep=max_keep,
        full_complement=full_complement,
        inv_block=inv_block,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different masking strategies
    input_size = (16, 224, 224)  # T, H, W
    batch_size = 2
    
    print("Testing video masking strategies...")
    
    # Test block masking
    block_generator = VideoMaskGenerator(
        input_size=input_size,
        strategy=MaskingStrategy.BLOCK,
        mask_ratio=0.75,
        device="cpu"
    )
    
    block_mask = block_generator.generate_mask(batch_size)
    print(f"Block mask shape: {block_mask.shape}")
    print(f"Block mask ratio: {block_mask.float().mean():.3f}")
    
    # Test tube masking
    tube_generator = VideoMaskGenerator(
        input_size=input_size,
        strategy=MaskingStrategy.TUBE,
        mask_ratio=0.75,
        device="cpu"
    )
    
    tube_mask = tube_generator.generate_mask(batch_size)
    print(f"Tube mask shape: {tube_mask.shape}")
    print(f"Tube mask ratio: {tube_mask.float().mean():.3f}")
    
    # Test adaptive generator
    adaptive_generator = AdaptiveMaskGenerator(
        input_size=input_size,
        device="cpu"
    )
    
    adaptive_mask = adaptive_generator.generate_mask(
        batch_size=batch_size,
        strategy=MaskingStrategy.RUNNING_CELL,
        mask_ratio=0.6
    )
    print(f"Adaptive mask shape: {adaptive_mask.shape}")
    print(f"Adaptive mask ratio: {adaptive_mask.float().mean():.3f}")
    
    # Test curriculum masking
    curriculum_schedule = [
        (0, MaskingStrategy.RANDOM, 0.5),
        (1000, MaskingStrategy.BLOCK, 0.6),
        (5000, MaskingStrategy.TUBE, 0.75),
    ]
    
    curriculum_mask = adaptive_generator.generate_curriculum_mask(
        batch_size=batch_size,
        training_step=2000,
        curriculum_schedule=curriculum_schedule
    )
    print(f"Curriculum mask shape: {curriculum_mask.shape}")
    print(f"Curriculum mask ratio: {curriculum_mask.float().mean():.3f}")
    
    print("Video masking tests completed successfully!") 