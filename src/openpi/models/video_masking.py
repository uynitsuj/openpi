"""
Video Masking Utilities for World Models

This module provides various masking strategies for video sequences, inspired by VJEPA-2
and other video understanding models. These masking strategies are used for self-supervised
learning from video data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Union
from enum import Enum
import math


class MaskingStrategy(Enum):
    """Different masking strategies for video sequences."""
    RANDOM = "random"
    BLOCK = "block"
    TUBE = "tube"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    RUNNING_CELL = "running_cell"


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
        """
        self.input_size = (num_frames, image_size[0], image_size[1])
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.mask_ratio = mask_ratio
        self.strategy = masking_strategy
        self.device = device
        
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
        """Generate masks that are consistent across time."""
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            # Select spatial locations to mask
            spatial_patches = self.num_patches_h * self.num_patches_w
            num_spatial_masked = int(self.mask_ratio * spatial_patches)
            
            # Random spatial locations
            spatial_mask = torch.zeros(spatial_patches, dtype=torch.bool, device=self.device)
            masked_spatial_indices = torch.randperm(spatial_patches, device=self.device)[:num_spatial_masked]
            spatial_mask[masked_spatial_indices] = True
            
            # Apply mask to all temporal locations
            for t in range(self.num_patches_t):
                for spatial_idx in range(spatial_patches):
                    if spatial_mask[spatial_idx]:
                        h = spatial_idx // self.num_patches_w
                        w = spatial_idx % self.num_patches_w
                        patch_idx = self._get_patch_index(t, h, w)
                        mask[patch_idx] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_spatial_mask(self, batch_size: int) -> torch.Tensor:
        """Generate masks that vary across time but are spatially structured."""
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            patches_per_frame = int(self.num_masked / self.num_patches_t)
            
            for t in range(self.num_patches_t):
                # Random spatial patches for this frame
                spatial_patches = self.num_patches_h * self.num_patches_w
                spatial_indices = torch.randperm(spatial_patches, device=self.device)[:patches_per_frame]
                
                for spatial_idx in spatial_indices:
                    h = spatial_idx // self.num_patches_w
                    w = spatial_idx % self.num_patches_w
                    patch_idx = self._get_patch_index(t, h, w)
                    mask[patch_idx] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _generate_running_cell_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate running cell masks inspired by VJEPA-2.
        
        This creates moving blocks of masked patches across space and time.
        """
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.total_patches, dtype=torch.bool, device=self.device)
            
            # Parameters for running cell
            cell_size = 3  # Size of the moving cell
            num_cells = self.num_masked // (cell_size * cell_size * cell_size)
            
            for _ in range(num_cells):
                # Random starting position
                t_start = torch.randint(0, max(1, self.num_patches_t - cell_size), (1,)).item()
                h_start = torch.randint(0, max(1, self.num_patches_h - cell_size), (1,)).item()
                w_start = torch.randint(0, max(1, self.num_patches_w - cell_size), (1,)).item()
                
                # Create moving cell
                for dt in range(cell_size):
                    for dh in range(cell_size):
                        for dw in range(cell_size):
                            t = t_start + dt
                            h = h_start + dh
                            w = w_start + dw
                            
                            if (t < self.num_patches_t and 
                                h < self.num_patches_h and 
                                w < self.num_patches_w):
                                patch_idx = self._get_patch_index(t, h, w)
                                mask[patch_idx] = True
            
            masks.append(mask)
        
        return torch.stack(masks, dim=0)
    
    def _get_patch_index(self, t: int, h: int, w: int) -> int:
        """Convert (t, h, w) coordinates to linear patch index."""
        return t * (self.num_patches_h * self.num_patches_w) + h * self.num_patches_w + w
    
    def get_patch_coordinates(self, patch_idx: int) -> Tuple[int, int, int]:
        """Convert linear patch index to (t, h, w) coordinates."""
        spatial_size = self.num_patches_h * self.num_patches_w
        t = patch_idx // spatial_size
        spatial_idx = patch_idx % spatial_size
        h = spatial_idx // self.num_patches_w
        w = spatial_idx % self.num_patches_w
        return t, h, w


class AdaptiveMaskGenerator:
    """
    Adaptive mask generator that can use different strategies for different stages of training.
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
        
        # Initialize generators for different strategies
        self.generators = {}
        for strategy in MaskingStrategy:
            self.generators[strategy] = VideoMaskGenerator(
                input_size=input_size,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                strategy=strategy,
                device=device,
            )
    
    def generate_mask(
        self,
        batch_size: int,
        strategy: MaskingStrategy = MaskingStrategy.BLOCK,
        mask_ratio: float = 0.75,
    ) -> torch.Tensor:
        """
        Generate masks using the specified strategy.
        
        Args:
            batch_size: Size of the batch
            strategy: Masking strategy to use
            mask_ratio: Ratio of patches to mask
            
        Returns:
            Boolean mask tensor
        """
        generator = self.generators[strategy]
        generator.mask_ratio = mask_ratio
        generator.num_masked = int(mask_ratio * generator.total_patches)
        
        return generator.generate_mask(batch_size)
    
    def generate_curriculum_mask(
        self,
        batch_size: int,
        training_step: int,
        curriculum_schedule: List[Tuple[int, MaskingStrategy, float]],
    ) -> torch.Tensor:
        """
        Generate masks using a curriculum schedule.
        
        Args:
            batch_size: Size of the batch
            training_step: Current training step
            curriculum_schedule: List of (step, strategy, mask_ratio) tuples
            
        Returns:
            Boolean mask tensor
        """
        # Find the current curriculum stage
        current_strategy = MaskingStrategy.BLOCK
        current_mask_ratio = 0.75
        
        for step, strategy, mask_ratio in curriculum_schedule:
            if training_step >= step:
                current_strategy = strategy
                current_mask_ratio = mask_ratio
            else:
                break
        
        return self.generate_mask(batch_size, current_strategy, current_mask_ratio)


def create_video_mask(
    video_shape: Tuple[int, int, int, int, int],  # (B, T, H, W, C)
    patch_size: Tuple[int, int] = (16, 16),
    temporal_patch_size: int = 2,
    mask_ratio: float = 0.75,
    strategy: MaskingStrategy = MaskingStrategy.BLOCK,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convenience function to create video masks.
    
    Args:
        video_shape: Shape of input video (B, T, H, W, C)
        patch_size: Spatial patch size
        temporal_patch_size: Temporal patch size
        mask_ratio: Ratio of patches to mask
        strategy: Masking strategy
        device: Device to create masks on
        
    Returns:
        Boolean mask tensor of shape (B, num_patches)
    """
    B, T, H, W, C = video_shape
    
    generator = VideoMaskGenerator(
        input_size=(T, H, W),
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        mask_ratio=mask_ratio,
        strategy=strategy,
        device=device,
    )
    
    return generator.generate_mask(B)


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