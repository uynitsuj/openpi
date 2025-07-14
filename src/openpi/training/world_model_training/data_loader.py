"""
World Model Data Loader

This module provides data loading functionality for world models that learn from video sequences
rather than action prediction. It handles video sequence loading, masking, and preprocessing.
"""

import dataclasses
import logging
from typing import Optional, Tuple, Dict, Any, Iterator, Sequence
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import datasets

from openpi.models.world_model import WorldModelInput, WorldModelOutput
from openpi.models.video_masking import VideoMaskGenerator, MaskingStrategy, create_video_mask
from openpi.shared import array_typing as at
from openpi.training import data_loader as base_data_loader
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

logger = logging.getLogger("openpi")


class ProgressiveMaskingSchedule:
    """Progressive masking schedule similar to official VJEPA2."""
    
    def __init__(self, total_steps: int = 50000):
        self.total_steps = total_steps
        
        # Default schedule based on official VJEPA2
        self.schedule = {
            # Phase 1: Initial training (0-30% of steps)
            'phase1': {
                'start_step': 0,
                'end_step': int(0.3 * total_steps),
                'mask_ratio': 0.5,
                'block_size': 4,
                'num_masked_patches': None,  # Use mask_ratio percentage
            },
            # Phase 2: Progressive difficulty (30-70% of steps)
            'phase2': {
                'start_step': int(0.3 * total_steps),
                'end_step': int(0.7 * total_steps),
                'mask_ratio': 0.75,
                'block_size': 8,
                'num_masked_patches': None,  # Use mask_ratio percentage
            },
            # Phase 3: High difficulty (70-100% of steps)
            'phase3': {
                'start_step': int(0.7 * total_steps),
                'end_step': total_steps,
                'mask_ratio': 0.9,
                'block_size': 16,
                'num_masked_patches': None,  # Use mask_ratio percentage
            }
        }
    
    def get_masking_params(self, current_step: int) -> dict:
        """Get masking parameters for current training step."""
        for phase_name, phase_config in self.schedule.items():
            if phase_config['start_step'] <= current_step < phase_config['end_step']:
                # Linear interpolation within phase
                phase_progress = (current_step - phase_config['start_step']) / (phase_config['end_step'] - phase_config['start_step'])
                
                # Interpolate mask_ratio (num_masked_patches is always None to use percentages)
                if phase_name == 'phase1':
                    # Phase 1: constant
                    mask_ratio = phase_config['mask_ratio']
                elif phase_name == 'phase2':
                    # Phase 2: interpolate from phase1 to phase2
                    prev_phase = self.schedule['phase1']
                    mask_ratio = prev_phase['mask_ratio'] + phase_progress * (phase_config['mask_ratio'] - prev_phase['mask_ratio'])
                else:  # phase3
                    # Phase 3: interpolate from phase2 to phase3
                    prev_phase = self.schedule['phase2']
                    mask_ratio = prev_phase['mask_ratio'] + phase_progress * (phase_config['mask_ratio'] - prev_phase['mask_ratio'])
                
                num_masked_patches = None  # Always use mask_ratio percentage
                
                return {
                    'mask_ratio': mask_ratio,
                    'block_size': phase_config['block_size'],
                    'num_masked_patches': num_masked_patches,
                    'phase': phase_name,
                    'phase_progress': phase_progress,
                }
        
        # Fallback to final phase
        final_phase = self.schedule['phase3']
        return {
            'mask_ratio': final_phase['mask_ratio'],
            'block_size': final_phase['block_size'],
            'num_masked_patches': None,  # Always use mask_ratio percentage
            'phase': 'phase3',
            'phase_progress': 1.0,
        }


@dataclasses.dataclass
class WorldModelDataConfig:
    """Configuration for world model data loading."""
    
    # LeRobot dataset configuration
    repo_id: str | None = None
    num_frames: int = 8  # Number of frames per video sequence
    frame_skip: int = 1  # Skip frames for temporal sampling
    image_size: Tuple[int, int] = (224, 224)
    
    # Video masking configuration
    masking_strategy: MaskingStrategy = MaskingStrategy.BLOCK
    mask_ratio: float = 0.5  # Ratio of patches to mask
    
    # Image preprocessing
    image_keys: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    normalize_images: bool = True
    
    # Dataset filtering
    min_episode_length: int = 10  # Minimum episode length to consider
    max_episodes: Optional[int] = None  # Limit number of episodes for debugging

    # Multi-view batching
    multi_view_batch_mode: bool = False  # If True, treat each camera view as a separate batch sample
    num_masked_patches: int | None = None  # If None, use mask_ratio percentage; if int, use fixed number
    
    # Progressive masking schedule
    use_progressive_masking: bool = True
    progressive_masking_schedule: dict = None  # Will be set to default if None


class WorldModelDataset(Dataset):
    """Dataset for loading video sequences for world model training."""
    
    def __init__(
        self,
        config: WorldModelDataConfig,
        split: str = "train",
        shuffle: bool = True,
        current_step: int = 0,  # Add current_step parameter
    ):
        self.config = config
        self.split = split
        self.shuffle = shuffle
        self.current_step = current_step
        
        # Add simple caching for video frames
        self._frame_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize progressive masking schedule if enabled
        if config.use_progressive_masking:
            if config.progressive_masking_schedule is None:
                # Use default schedule
                self.progressive_schedule = ProgressiveMaskingSchedule(total_steps=50000)
            else:
                # Use custom schedule
                self.progressive_schedule = ProgressiveMaskingSchedule(**config.progressive_masking_schedule)
        else:
            self.progressive_schedule = None
        
        # Initialize video mask generator with current parameters
        self._update_mask_generator()
        
        # Load dataset
        self.dataset = self._load_dataset()
        self.episodes = self._prepare_episodes()
        
        logger.info(f"Loaded {len(self.episodes)} valid episodes from {config.repo_id}")
        
        # Debug: check what keys are available in the first dataset item
        if len(self.episodes) > 0:
            first_item = self.dataset[0]
            logger.info(f"First dataset item keys: {list(first_item.keys())}")
            logger.info(f"First dataset item types: {[(k, type(v)) for k, v in first_item.items()]}")
    
    def _update_mask_generator(self):
        """Update the mask generator with current masking parameters."""
        if self.progressive_schedule is not None:
            # Get current masking parameters from progressive schedule
            masking_params = self.progressive_schedule.get_masking_params(self.current_step)
            mask_ratio = masking_params['mask_ratio']
            block_size = masking_params['block_size']
            num_masked_patches = masking_params['num_masked_patches']
        else:
            # Use fixed parameters from config
            mask_ratio = self.config.mask_ratio
            block_size = 4  # Default block size
            num_masked_patches = self.config.num_masked_patches
        
        # Initialize video mask generator with current parameters
        self.mask_generator = VideoMaskGenerator(
            num_frames=self.config.num_frames,
            image_size=self.config.image_size,
            masking_strategy=self.config.masking_strategy,
            mask_ratio=mask_ratio,
            num_masked_patches=num_masked_patches,
        )
        
        # Store current parameters for logging
        self.current_mask_ratio = mask_ratio
        self.current_block_size = block_size
        self.current_num_masked_patches = num_masked_patches
    
    def update_step(self, current_step: int):
        """Update the current training step and regenerate mask generator."""
        self.current_step = current_step
        self._update_mask_generator()
    
    def _load_dataset(self) -> lerobot_dataset.LeRobotDataset:
        """Load the LeRobot dataset using the proper infrastructure."""
        if self.config.repo_id is None:
            raise ValueError("repo_id must be specified for world model training")
        
        # Get dataset metadata
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(self.config.repo_id)
        
        # Calculate timestamps for the number of frames we need
        timestamps = [i / dataset_meta.fps for i in range(self.config.num_frames)]
        
        # Load dataset using LeRobot infrastructure (uses cached data)
        dataset = lerobot_dataset.LeRobotDataset(
            self.config.repo_id,
            delta_timestamps={
                "state": timestamps  # Use 'state' which contains the video data
            },
            video_backend="pyav",  # Use pyav for video loading
        )
        
        return dataset
    
    def _prepare_episodes(self) -> list[dict]:
        """Prepare episodes by grouping sequential frames."""
        episodes = []
        
        # Get dataset metadata for episode information
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(self.config.repo_id)
        
        # Get episode information from the dataset
        episode_info = dataset_meta.episodes
        
        # Process each episode
        for episode_idx, episode_data in episode_info.items():
            if self.config.max_episodes is not None and episode_idx >= self.config.max_episodes:
                break
                
            episode_length = episode_data["length"]
            
            if episode_length < self.config.min_episode_length:
                continue
            
            # Create sliding windows of video sequences
            for start_frame in range(0, episode_length - self.config.num_frames + 1, self.config.frame_skip):
                end_frame = start_frame + self.config.num_frames * self.config.frame_skip
                if end_frame > episode_length:
                    break
                
                # Calculate frame indices for this sequence
                frame_indices = [
                    start_frame + i * self.config.frame_skip
                    for i in range(self.config.num_frames)
                ]
                
                episodes.append({
                    "episode_index": episode_idx,
                    "frame_indices": frame_indices,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "episode_length": episode_length,
                })
        
        return episodes
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Tuple[WorldModelInput, WorldModelOutput]:
        """Get a video sequence and its masked version."""
        episode = self.episodes[idx]
        episode_idx = episode["episode_index"]
        frame_indices = episode["frame_indices"]
        
        # Check cache first
        cache_key = f"{episode_idx}_{frame_indices[0]}_{frame_indices[-1]}"
        if cache_key in self._frame_cache:
            self._cache_hits += 1
            frame_images = self._frame_cache[cache_key]
        else:
            self._cache_misses += 1
            # Load video frames from the LeRobot dataset
            frame_data = self.dataset[idx]
            
            # Debug: show what keys are available in the dataset
            # if idx < 5:  # Only debug first few items
            #     logger.info(f"Dataset item {idx} keys: {list(frame_data.keys())}")
            
            # Map our expected camera names to the actual dataset column names
            camera_mapping = {
                "base_0_rgb": "top_camera-images-rgb",
                "left_wrist_0_rgb": "left_camera-images-rgb", 
                "right_wrist_0_rgb": "right_camera-images-rgb",
            }
            
            # Extract images for all cameras
            frame_images = {}
            found_cameras = []
            for expected_key, actual_key in camera_mapping.items():
                if actual_key in frame_data:
                    found_cameras.append(expected_key)
                    images = frame_data[actual_key]
                    
                    # Handle different image formats
                    if isinstance(images, torch.Tensor):
                        # Direct torch tensor
                        image = images.numpy()
                    elif isinstance(images, dict) and "bytes" in images:
                        # Handle PIL images from bytes
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(images["bytes"]))
                        image = np.array(image)
                    elif hasattr(images, 'convert'):
                        # Handle PIL images directly
                        image = np.array(images)
                    elif isinstance(images, np.ndarray):
                        # Already numpy array
                        image = images
                    else:
                        logger.warning(f"Unknown image format for {actual_key}: {type(images)}")
                        continue
                    
                    # Resize and normalize
                    if image.shape[:2] != self.config.image_size:
                        image = self._resize_image(image, self.config.image_size)
                    
                    if self.config.normalize_images:
                        image = self._normalize_image(image)
                    
                    frame_images[expected_key] = image
            
            # Cache the processed frames
            self._frame_cache[cache_key] = frame_images
            
            # Limit cache size to prevent memory issues
            if len(self._frame_cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self._frame_cache))
                del self._frame_cache[oldest_key]
        
        # Multi-view batch mode: return a list of (input, output) for each camera view
        if self.config.multi_view_batch_mode:
            samples = []
            for cam_key in self.config.image_keys:
                if cam_key in frame_images:
                    video_tensor = self._frames_to_tensor([ {cam_key: frame_images[cam_key]} ], force_camera_key=cam_key )
                    mask = self.mask_generator.generate_mask(batch_size=1)
                    mask = mask.view(mask.size(0), -1)
                    model_input = WorldModelInput(
                        video_frames=video_tensor,
                        mask=mask,
                        camera_names=[cam_key],
                    )
                    model_output = WorldModelOutput(
                        predicted_features=video_tensor,
                        reconstruction_loss=jnp.array(0.0),
                        mask_ratio=jnp.array(self.current_mask_ratio),
                    )
                    samples.append((model_input, model_output))
            
            # Debug logging removed for cleaner output
            
            # Log cache statistics periodically
            if idx % 100 == 0 and idx > 0:
                total_requests = self._cache_hits + self._cache_misses
                if total_requests > 0:
                    cache_hit_rate = self._cache_hits / total_requests
                    logger.info(f"Cache stats: {self._cache_hits}/{total_requests} hits ({cache_hit_rate:.1%})")
        
            return samples  # List of (input, output) tuples
        else:
            # Single-view mode: use only the first available camera
            video_tensor = self._frames_to_tensor([frame_images])
            mask = self.mask_generator.generate_mask(batch_size=1)
            mask = mask.view(mask.size(0), -1)
            model_input = WorldModelInput(
                video_frames=video_tensor,
                mask=mask,
                camera_names=list(self.config.image_keys),
            )
            model_output = WorldModelOutput(
                predicted_features=video_tensor,
                reconstruction_loss=jnp.array(0.0),
                mask_ratio=jnp.array(self.current_mask_ratio),
            )
            return model_input, model_output
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size with optimized processing."""
        from PIL import Image
        
        # Handle different image shapes and types
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (C, H, W) -> (H, W, C) for PIL
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[2] == 3:  # (H, W, C) - already correct
                pass
            elif image.shape[0] == 1:  # (1, H, W) -> (H, W)
                image = image.squeeze(0)
            elif image.shape[2] == 1:  # (H, W, 1) -> (H, W)
                image = image.squeeze(2)
            else:
                # Unknown format, try to reshape
                logger.warning(f"Unexpected image shape: {image.shape}")
                if image.shape[0] == 1 and image.shape[1] == 1:
                    # (1, 1, H) -> (H, H) - square image
                    image = image.squeeze()
                else:
                    # Try to make it square
                    image = image.squeeze()
        
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Optimized resizing: only resize if necessary
        if len(image.shape) == 2:
            # 2D array, convert to RGB if needed
            if target_size[0] != image.shape[0] or target_size[1] != image.shape[1]:
                pil_image = Image.fromarray(image)
                resized = pil_image.resize(target_size, Image.LANCZOS)
                image = np.array(resized)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image - use faster resize if possible
            if target_size[0] != image.shape[0] or target_size[1] != image.shape[1]:
                # Use faster resize for common cases
                if target_size == (224, 224) and image.shape[:2] in [(480, 640), (640, 480), (720, 1280)]:
                    # Use faster resize for common video resolutions
                    pil_image = Image.fromarray(image)
                    resized = pil_image.resize(target_size, Image.BILINEAR)  # Faster than LANCZOS
                    image = np.array(resized)
                else:
                    pil_image = Image.fromarray(image)
                    resized = pil_image.resize(target_size, Image.LANCZOS)
                    image = np.array(resized)
        else:
            logger.warning(f"Cannot handle image shape {image.shape}, using zeros")
            image = np.zeros((*target_size, 3), dtype=np.uint8)
        
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1] range."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image * 2.0 - 1.0
    
    def _frames_to_tensor(self, frames: list[dict], force_camera_key: str = None) -> jnp.ndarray:
        """Convert list of frame dictionaries to a tensor. If force_camera_key is set, use only that camera."""
        frame_tensors = []
        for frame in frames:
            camera_images = []
            if force_camera_key is not None:
                if force_camera_key in frame:
                    camera_images.append(frame[force_camera_key])
            else:
                for key in self.config.image_keys:
                    if key in frame:
                        camera_images.append(frame[key])
                        break  # Only use the first camera
            if camera_images:
                image = camera_images[0]
                frame_tensors.append(image)
            else:
                height, width = self.config.image_size
                image = np.zeros((height, width, 3), dtype=np.float32)
                frame_tensors.append(image)
        video_tensor = np.stack(frame_tensors, axis=0)
        if video_tensor.shape[0] == 1 and self.config.num_frames > 1:
            video_tensor = np.repeat(video_tensor, self.config.num_frames, axis=0)
        return jnp.array(video_tensor)


class FakeWorldModelDataset(Dataset):
    """Fake dataset for testing world model training."""
    
    def __init__(self, config: WorldModelDataConfig, size: int = 1000):
        self.config = config
        self.size = size
        self.mask_generator = VideoMaskGenerator(
            num_frames=config.num_frames,
            image_size=config.image_size,
            masking_strategy=config.masking_strategy,
            mask_ratio=config.mask_ratio,
            num_masked_patches=config.num_masked_patches,
        )
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[WorldModelInput, WorldModelOutput]:
        """Generate fake video data."""
        # Generate random video frames
        height, width = self.config.image_size
        total_width = width * len(self.config.image_keys)
        
        video_frames = np.random.randn(
            self.config.num_frames, height, total_width, 3
        ).astype(np.float32)
        
        # Generate mask
        mask = self.mask_generator.generate_mask(batch_size=1)
        mask = mask.view(mask.size(0), -1)
        
        # Create input and output
        model_input = WorldModelInput(
            video_frames=jnp.array(video_frames),
            mask=mask,
            camera_names=list(self.config.image_keys),
        )
        
        model_output = WorldModelOutput(
            predicted_features=jnp.array(video_frames),
            reconstruction_loss=jnp.array(0.0),
            mask_ratio=jnp.array(self.config.mask_ratio),
        )
        
        return model_input, model_output


class WorldModelDataLoader:
    """Data loader for world model training."""
    
    def __init__(
        self,
        config: WorldModelDataConfig,
        batch_size: int,
        split: str = "train",
        shuffle: bool = True,
        num_workers: int = 0,  # Use 0 workers to avoid multiprocessing issues
        fake_data: bool = False,
        current_step: int = 0,  # Add current_step parameter
        prefetch_factor: int = 2,  # Add prefetch_factor for better GPU utilization
    ):
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_step = current_step
        
        # Create dataset
        self.dataset = WorldModelDataset(config, split=split, shuffle=shuffle, current_step=current_step)
        
        # Create indices for batching
        self.indices = list(range(len(self.dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)
        
        self.current_idx = 0
        
        # Enable pin_memory for faster GPU transfer
        self.pin_memory = True
    
    def update_step(self, current_step: int):
        """Update the current training step."""
        self.current_step = current_step
        self.dataset.update_step(current_step)
    
    def _collate_fn(self, batch: list) -> Tuple[WorldModelInput, WorldModelOutput]:
        """Collate function for batching."""
        # If multi_view_batch_mode, flatten the batch (list of lists)
        if self.config.multi_view_batch_mode:
            flat_batch = []
            for item in batch:
                if isinstance(item, list):
                    flat_batch.extend(item)
                else:
                    flat_batch.append(item)
            inputs, outputs = zip(*flat_batch)
        else:
            inputs, outputs = zip(*batch)
        
        # Convert PyTorch tensors to JAX arrays
        def torch_to_jax(tensor):
            if hasattr(tensor, 'detach'):
                numpy_array = tensor.detach().cpu().numpy()
            else:
                numpy_array = np.array(tensor)
            
            # Debug logging removed for cleaner output
            
            return jnp.array(numpy_array)
        
        # Stack inputs
        batched_input = WorldModelInput(
            video_frames=jnp.stack([inp.video_frames for inp in inputs]),
            mask=jnp.stack([torch_to_jax(inp.mask) for inp in inputs]).squeeze(1),  # Remove extra dimension from stacking
            camera_names=[inp.camera_names[0] for inp in inputs],
        )
        
        # Debug logging removed for cleaner output
        
        # Stack outputs
        batched_output = WorldModelOutput(
            predicted_features=jnp.stack([out.predicted_features for out in outputs]),
            reconstruction_loss=jnp.stack([out.reconstruction_loss for out in outputs]),
            mask_ratio=jnp.stack([out.mask_ratio for out in outputs]),
        )
        
        return batched_input, batched_output
    
    def __iter__(self) -> Iterator[Tuple[WorldModelInput, WorldModelOutput]]:
        """Iterate over batches."""
        if self.config.multi_view_batch_mode:
            # Flatten all multi-view samples up front
            flat_samples = []
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                if isinstance(item, list):
                    flat_samples.extend(item)
                else:
                    flat_samples.append(item)
            total = len(flat_samples)
            for i in range(0, total, self.batch_size):
                batch = flat_samples[i:i+self.batch_size]
                yield self._collate_fn(batch)
        else:
            while self.current_idx < len(self.indices):
                batch_inputs = []
                batch_outputs = []
                for _ in range(self.batch_size):
                    if self.current_idx >= len(self.indices):
                        break
                    idx = self.indices[self.current_idx]
                    item = self.dataset[idx]
                    batch_inputs.append(item[0])
                    batch_outputs.append(item[1])
                    self.current_idx += 1
                if batch_inputs:
                    yield self._collate_fn(list(zip(batch_inputs, batch_outputs)))
    
    def __len__(self) -> int:
        # In multi-view batch mode, count total number of multi-view samples
        if self.config.multi_view_batch_mode:
            total_samples = 0
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                if isinstance(item, list):
                    total_samples += len(item)
                else:
                    total_samples += 1
            return (total_samples + self.batch_size - 1) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def data_config(self) -> WorldModelDataConfig:
        return self.config


def create_world_model_data_loader(
    config: WorldModelDataConfig,
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 0,  # Force 0 workers to avoid multiprocessing issues
    fake_data: bool = False,
    current_step: int = 0,  # Add current_step parameter
) -> WorldModelDataLoader:
    """Create a world model data loader."""
    return WorldModelDataLoader(
        config=config,
        batch_size=batch_size,
        split=split,
        shuffle=shuffle,
        num_workers=num_workers,  # Always use 0 workers
        fake_data=fake_data,
        current_step=current_step,
    )
