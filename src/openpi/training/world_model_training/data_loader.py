"""
Optimized World Model Data Loader

This module provides highly optimized data loading functionality for world models
with improvements in memory management, parallel processing, and caching.
"""

import dataclasses
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, Iterator, Sequence, List
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, IterableDataset
import datasets

from openpi.models.world_model import WorldModelInput, WorldModelOutput
from openpi.models.video_masking import VideoMaskGenerator, MaskingStrategy, create_video_mask
from openpi.shared import array_typing as at
from openpi.training import data_loader as base_data_loader
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from .video_loader import VideoFrameLoader

logger = logging.getLogger("openpi")

from lerobot.common.constants import HF_LEROBOT_HOME


def maybe_time_reverse(frames, p=0.15, rng=None):
    """Apply time reversal augmentation."""
    import random
    if random.random() < p:
        return frames[::-1]
    return frames


def maybe_frame_drop(frames, drop_p=0.15):
    """Apply frame dropping augmentation."""
    import random
    keep = [i for i in range(len(frames)) if random.random() > drop_p]
    if len(keep) == 0: 
        keep = [0]
    return [frames[i] for i in keep]


def get_curriculum_mask_ratio(step: int, cfg) -> float:
    """Get curriculum mask ratio for current training step."""
    if step >= cfg.curriculum_steps:
        return cfg.end_ratio
    # linear ramp with proper clamping
    frac = min(1.0, step / max(1, cfg.curriculum_steps))
    mask_ratio = cfg.start_ratio + frac * (cfg.end_ratio - cfg.start_ratio)
    return mask_ratio


class ProgressiveMaskingSchedule:
    """Progressive masking schedule similar to official VJEPA2."""
    
    def __init__(self, total_steps: int = 50000):
        self.total_steps = total_steps
        
        # Progressive masking schedule: 25% → 50% → 75%
        self.schedule = {
            # Phase 1: Easy training (0-33% of steps) - 25% masking
            'phase1': {
                'start_step': 0,
                'end_step': int(0.33 * total_steps),
                'mask_ratio': 0.25,
                'block_size': 4,
                'num_masked_patches': None,  # Use mask_ratio percentage
            },
            # Phase 2: Medium difficulty (33-66% of steps) - 50% masking
            'phase2': {
                'start_step': int(0.33 * total_steps),
                'end_step': int(0.66 * total_steps),
                'mask_ratio': 0.50,
                'block_size': 8,
                'num_masked_patches': None,  # Use mask_ratio percentage
            },
            # Phase 3: Hard difficulty (66-100% of steps) - 75% masking
            'phase3': {
                'start_step': int(0.66 * total_steps),
                'end_step': total_steps,
                'mask_ratio': 0.75,
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
    
    repo_id: str | None = None
    num_frames: int = 8
    frame_skip: int = 1
    image_size: Tuple[int, int] = (224, 224)
    
    masking_strategy: MaskingStrategy = MaskingStrategy.BLOCK
    mask_ratio: float = 0.5
    
    image_keys: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    normalize_images: bool = True
    
    min_episode_length: int = 10
    max_episodes: Optional[int] = None

    multi_view_batch_mode: bool = False
    num_masked_patches: int | None = None
    
    use_progressive_masking: bool = True
    progressive_masking_schedule: dict = None
    
    chunk_size: int = 500
    
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    split_seed: int = 42
    
    # Video augmentation options
    use_video_augmentation: bool = True
    time_reverse_prob: float = 0.15
    frame_drop_prob: float = 0.15


class LRUCache:
    """Thread-safe LRU cache for frame sequences."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def __getstate__(self):
        """Prepare for pickling by excluding non-picklable lock."""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        del state['lock']
        return state
    
    def __setstate__(self, state):
        """Restore from pickle by recreating the lock."""
        self.__dict__.update(state)
        # Recreate the lock
        self.lock = threading.RLock()
    
    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            self.misses += 1
            return None
    
    def put(self, key: str, value):
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache)
            }


class BatchImageProcessor:
    """Optimized batch image processing using OpenCV."""
    
    @staticmethod
    def batch_resize_normalize(
        images: List[np.ndarray], 
        target_size: Tuple[int, int],
        normalize: bool = True
    ) -> List[np.ndarray]:
        """Batch process multiple images efficiently."""
        processed = []
        
        resize_logged = False
        for image in images:
            # Use OpenCV with smart interpolation selection
            if image.shape[:2] != target_size:
                current_h, current_w = image.shape[:2]
                target_h, target_w = target_size
                
                # Choose interpolation method based on scaling direction
                if current_h * current_w < target_h * target_w:
                    # Upscaling: use cubic interpolation for better quality (224->256)
                    interpolation = cv2.INTER_CUBIC
                    resize_type = "upscaling"
                else:
                    # Downscaling: use area interpolation for better anti-aliasing
                    interpolation = cv2.INTER_AREA
                    resize_type = "downscaling"
                
                # Log resize info once per batch
                if not resize_logged:
                    logger.info(f"Resizing images: {current_h}x{current_w} -> {target_h}x{target_w} ({resize_type} with {interpolation})")
                    resize_logged = True
                
                # OpenCV expects (width, height) while we use (height, width)
                image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
            
            if normalize:
                # Vectorized normalization
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                image = image * 2.0 - 1.0
            
            processed.append(image)
        
        return processed
    
    @staticmethod
    def batch_frames_to_tensor(frame_sequences: List[List[Dict]], config: WorldModelDataConfig) -> List[jnp.ndarray]:
        """Convert multiple frame sequences to tensors in batch."""
        tensors = []
        
        for frames in frame_sequences:
            frame_tensors = []
            for frame in frames:
                # Get first available camera
                for key in config.image_keys:
                    if key in frame:
                        frame_tensors.append(frame[key])
                        break
                else:
                    # Create dummy frame if no camera found
                    height, width = config.image_size
                    dummy = np.zeros((height, width, 3), dtype=np.float32)
                    frame_tensors.append(dummy)
            
            # Stack frames into video tensor
            video_tensor = np.stack(frame_tensors, axis=0)
            
            # Ensure correct number of frames
            if video_tensor.shape[0] != config.num_frames:
                if video_tensor.shape[0] > config.num_frames:
                    video_tensor = video_tensor[:config.num_frames]
                else:
                    # Pad with zeros
                    height, width = config.image_size
                    padding_frames = config.num_frames - video_tensor.shape[0]
                    padding = np.zeros((padding_frames, height, width, 3), dtype=np.float32)
                    video_tensor = np.concatenate([video_tensor, padding], axis=0)
            
            tensors.append(jnp.array(video_tensor))
        
        return tensors


class ParallelVideoLoader:
    """Parallel video frame loader with thread pool."""
    
    def __init__(self, video_loader: VideoFrameLoader, max_workers: int = 2):
        self.video_loader = video_loader
        self.max_workers = max_workers  # Store for pickle support
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout_count = 0
        self.max_timeouts = 10  # Fall back to sequential after this many timeouts
    
    def load_frame_batch(
        self, 
        requests: List[Tuple[int, List[int], str, Tuple[int, int]]]
    ) -> List[List[np.ndarray]]:
        """Load multiple frame sequences in parallel."""
        # Fall back to sequential if too many timeouts or executor is shut down
        if (self.timeout_count >= self.max_timeouts or 
            not hasattr(self, 'executor') or 
            self.executor._shutdown):
            if self.timeout_count >= self.max_timeouts:
                logger.warning(f"Too many timeouts ({self.timeout_count}), falling back to sequential loading")
            # Sequential loading fallback
            results = []
            for episode_idx, frame_indices, camera, target_size in requests:
                try:
                    frames = self.video_loader.load_frames(
                        episode_idx=episode_idx,
                        frame_indices=frame_indices,
                        camera=camera,
                        target_size=target_size
                    )
                    results.append(frames if frames else [])
                except Exception as e:
                    logger.warning(f"Failed to load frame batch: {type(e).__name__}: {e}")
                    results.append([])
            return results
        
        futures = []
        
        for episode_idx, frame_indices, camera, target_size in requests:
            try:
                future = self.executor.submit(
                    self.video_loader.load_frames,
                    episode_idx=episode_idx,
                    frame_indices=frame_indices,
                    camera=camera,
                    target_size=target_size
                )
                futures.append(future)
            except Exception as e:
                logger.warning(f"Failed to submit frame loading task: {e}")
                # Add empty result for failed submission
                futures.append(None)
        
        results = []
        for future in futures:
            if future is None:
                results.append([])
                continue
                
            try:
                frames = future.result(timeout=5)  # Reduced timeout
                results.append(frames if frames else [])
            except Exception as e:
                if isinstance(e, TimeoutError) or "timeout" in str(e).lower():
                    self.timeout_count += 1
                logger.warning(f"Failed to load frame batch: {type(e).__name__}: {e}")
                results.append([])
        
        return results
    
    def __getstate__(self):
        """Prepare for pickling by excluding non-picklable executor."""
        state = self.__dict__.copy()
        # Remove the unpicklable executor
        state.pop('executor', None)
        return state
    
    def __setstate__(self, state):
        """Restore from pickle by recreating the executor."""
        self.__dict__.update(state)
        # Recreate the executor
        max_workers = getattr(self, 'max_workers', 2)  # Default to 2 if not found
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def __del__(self):
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass  # Ignore shutdown errors


class OptimizedWorldModelDataset(Dataset):
    """Optimized dataset with improved caching, parallel loading, and memory management."""
    
    def __init__(
        self,
        config: WorldModelDataConfig,
        split: str = "train",
        shuffle: bool = True,
        current_step: int = 0,
        chunk_size: int = 1000,  # Increased default chunk size
        cache_size: int = 200,   # Increased cache size
        max_workers: int = 2,    # Parallel loading workers
        prefetch_size: int = 32, # Number of items to prefetch
    ):
        self.config = config
        self.split = split
        self.shuffle = shuffle
        self.current_step = current_step
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.prefetch_size = prefetch_size
        
        # Initialize optimized components
        self.frame_cache = LRUCache(max_size=cache_size)
        self.batch_processor = BatchImageProcessor()
        
        # Initialize mask generator
        self.mask_generator = VideoMaskGenerator(
            num_frames=config.num_frames,
            image_size=config.image_size,
            masking_strategy=config.masking_strategy,
            mask_ratio=config.mask_ratio,
            num_masked_patches=config.num_masked_patches,
        )
        
        # Progressive masking
        if config.use_progressive_masking:
            self.progressive_schedule = ProgressiveMaskingSchedule()
            if config.progressive_masking_schedule is None:
                config.progressive_masking_schedule = self.progressive_schedule.get_masking_params(current_step)
            self.current_mask_ratio = config.progressive_masking_schedule.get("mask_ratio", config.mask_ratio)
            self.current_num_masked_patches = config.progressive_masking_schedule.get("num_masked_patches", config.num_masked_patches)
        else:
            self.current_mask_ratio = config.mask_ratio
            self.current_num_masked_patches = config.num_masked_patches
        
        # Load dataset
        if config.repo_id is None:
            raise ValueError("repo_id must be specified for world model training")
        
        self.dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(config.repo_id)
        self.episode_info = self.dataset_meta.episodes
        self.episode_info = self._apply_split(self.episode_info, split)
        self.dataset = self._load_dataset()
        
        # Initialize video loader with parallel processing
        dataset_name = self.config.repo_id
        dataset_cache_path = os.path.expanduser(f"{HF_LEROBOT_HOME}/{dataset_name}")
        self.video_loader = VideoFrameLoader(dataset_cache_path)
        self.parallel_loader = ParallelVideoLoader(self.video_loader, max_workers=max_workers)
        
        # Calculate dimensions
        self.total_episodes = len(self.episode_info)
        self.total_sequences = self._calculate_total_sequences()
        
        logger.info(f"Optimized dataset loaded: {self.total_episodes} episodes, {self.total_sequences} sequences for {split}")
        
        # Initialize prefetching before chunk loading
        self.prefetch_cache = {}
        self.prefetch_lock = threading.Lock()
        
        # Initialize chunked loading
        self.current_chunk_start = 0
        self.episodes = []
        self.episode_indices = []
        self._load_chunk(0)
        
        # Start prefetching
        self._start_prefetching()
    
    def _apply_split(self, episode_info: dict, split: str) -> dict:
        """Apply train/validation split to episode information."""
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        episode_list = list(episode_info.items())
        episode_list.sort(key=lambda x: x[0])
        
        import random
        random.seed(self.config.split_seed)
        random.shuffle(episode_list)
        
        total_episodes = len(episode_list)
        train_end = int(self.config.train_split_ratio * total_episodes)
        val_end = int((self.config.train_split_ratio + self.config.val_split_ratio) * total_episodes)
        
        if split == "train":
            selected_episodes = episode_list[:train_end]
        elif split == "validation":
            selected_episodes = episode_list[train_end:val_end]
        elif split == "test":
            selected_episodes = episode_list[val_end:]
        
        split_episode_info = dict(selected_episodes)
        logger.info(f"Split {split}: {len(split_episode_info)} episodes")
        
        if len(split_episode_info) == 0 and split != "train":
            import warnings
            warnings.warn(f"Split '{split}' is empty! Using fallback.")
            fallback_episodes = episode_list[:max(1, int(0.05 * total_episodes))]
            split_episode_info = dict(fallback_episodes)
        
        return split_episode_info
    
    def _calculate_total_sequences(self) -> int:
        """Calculate total number of sequences."""
        total = 0
        for episode_idx, episode_data in self.episode_info.items():
            episode_length = episode_data["length"]
            if episode_length < self.config.min_episode_length:
                continue
            
            num_windows = max(0, episode_length - self.config.num_frames + 1)
            if self.config.frame_skip > 1:
                num_windows = num_windows // self.config.frame_skip
            total += num_windows
        
        return total
    
    def _load_chunk(self, chunk_start: int):
        """Load chunk with optimized processing."""
        self.episodes = []
        self.episode_indices = []
        
        episode_indices = list(self.episode_info.keys())
        chunk_end = min(chunk_start + self.chunk_size, len(episode_indices))
        
        for local_idx in range(chunk_start, chunk_end):
            if local_idx >= len(episode_indices):
                break
            
            episode_idx = episode_indices[local_idx]
            episode_data = self.episode_info[episode_idx]
            episode_length = episode_data["length"]
            
            if episode_length < self.config.min_episode_length:
                continue
            
            for start_frame in range(0, episode_length - self.config.num_frames + 1, self.config.frame_skip):
                end_frame = start_frame + self.config.num_frames * self.config.frame_skip
                if end_frame > episode_length:
                    break
                
                frame_indices = [
                    start_frame + i * self.config.frame_skip
                    for i in range(self.config.num_frames)
                ]
                
                self.episodes.append({
                    "episode_index": episode_idx,
                    "frame_indices": frame_indices,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "episode_length": episode_length,
                })
                self.episode_indices.append(episode_idx)
        
        if self.shuffle:
            import random
            combined = list(zip(self.episodes, self.episode_indices))
            random.shuffle(combined)
            self.episodes, self.episode_indices = zip(*combined) if combined else ([], [])
        
        self.current_chunk_start = chunk_start
        self.current_chunk_end = chunk_end
        
        # Clear caches when loading new chunk
        self.frame_cache.clear()
        with self.prefetch_lock:
            self.prefetch_cache.clear()
    
    def _load_dataset(self) -> lerobot_dataset.LeRobotDataset:
        """Load dataset efficiently."""
        if self.config.repo_id is None:
            raise ValueError("repo_id must be specified")
        
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(self.config.repo_id)
        timestamps = [i / dataset_meta.fps for i in range(self.config.num_frames)]
        
        dataset = lerobot_dataset.LeRobotDataset(
            self.config.repo_id,
            delta_timestamps={"state": timestamps},
        )
        
        return dataset
    
    def _start_prefetching(self):
        """Start background prefetching thread."""
        def prefetch_worker():
            try:
                for i in range(min(self.prefetch_size, len(self.episodes))):
                    try:
                        with self.prefetch_lock:
                            if i not in self.prefetch_cache:
                                episode = self.episodes[i]
                                cache_key = f"{episode['episode_index']}_{episode['frame_indices'][0]}_{episode['frame_indices'][-1]}"
                                
                                if self.frame_cache.get(cache_key) is None:
                                    # Load frames
                                    frames = self._load_frames_optimized(episode)
                                    self.prefetch_cache[i] = frames
                    except Exception as e:
                        logger.debug(f"Prefetch failed for index {i}: {e}")
                        # Continue with other items
                        continue
            except Exception as e:
                logger.warning(f"Prefetch worker thread failed: {e}")
        
        try:
            prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
            prefetch_thread.start()
        except Exception as e:
            logger.warning(f"Failed to start prefetch thread: {e}")
            # Continue without prefetching
    
    def _load_frames_optimized(self, episode: dict) -> List[dict]:
        """Load frames with optimized processing."""
        episode_idx = episode["episode_index"]
        frame_indices = episode["frame_indices"]
        
        try:
            # Use parallel video loader for faster loading
            requests = [(episode_idx, [frame_idx], "top_camera-images-rgb", self.config.image_size) 
                       for frame_idx in frame_indices]
            
            frame_batches = self.parallel_loader.load_frame_batch(requests)
            
            frame_sequences = []
            for batch in frame_batches:
                if batch and len(batch) > 0:
                    frame_image = batch[0]
                    frame_images = {"base_0_rgb": frame_image}
                else:
                    # Dummy frame
                    height, width = self.config.image_size
                    dummy_frame = np.zeros((height, width, 3), dtype=np.float32)
                    frame_images = {"base_0_rgb": dummy_frame}
                
                frame_sequences.append(frame_images)
            
            # Apply video augmentations if enabled and in training split
            if (self.config.use_video_augmentation and 
                self.split == "train" and 
                len(frame_sequences) > 1):
                
                # Apply time reversal
                frame_sequences = maybe_time_reverse(
                    frame_sequences, 
                    p=self.config.time_reverse_prob
                )
                
                # Apply frame dropping
                frame_sequences = maybe_frame_drop(
                    frame_sequences,
                    drop_p=self.config.frame_drop_prob
                )
                
                # Pad/truncate back to fixed number of frames
                if len(frame_sequences) != self.config.num_frames:
                    if len(frame_sequences) > self.config.num_frames:
                        frame_sequences = frame_sequences[:self.config.num_frames]
                    else:
                        # Pad with repeated last frame
                        while len(frame_sequences) < self.config.num_frames:
                            frame_sequences.append(frame_sequences[-1])
            
            # Batch process images
            if self.config.normalize_images:
                all_images = [frame["base_0_rgb"] for frame in frame_sequences]
                processed_images = self.batch_processor.batch_resize_normalize(
                    all_images, self.config.image_size, normalize=True
                )
                for i, processed_img in enumerate(processed_images):
                    frame_sequences[i]["base_0_rgb"] = processed_img
            
            return frame_sequences
            
        except Exception as e:
            logger.warning(f"Failed to load frames for episode {episode_idx}: {e}")
            # Return dummy frames
            height, width = self.config.image_size
            dummy_frame = np.zeros((height, width, 3), dtype=np.float32)
            return [{"base_0_rgb": dummy_frame} for _ in frame_indices]
    
    def update_step(self, current_step: int):
        """Update training step and progressive masking."""
        self.current_step = current_step
        
        if self.config.use_progressive_masking:
            masking_params = self.progressive_schedule.get_masking_params(current_step)
            self.current_mask_ratio = masking_params.get("mask_ratio", self.config.mask_ratio)
            self.current_num_masked_patches = masking_params.get("num_masked_patches", self.config.num_masked_patches)
            
            self.mask_generator = VideoMaskGenerator(
                num_frames=self.config.num_frames,
                image_size=self.config.image_size,
                masking_strategy=self.config.masking_strategy,
                mask_ratio=self.current_mask_ratio,
                num_masked_patches=self.current_num_masked_patches,
            )
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getstate__(self):
        """Prepare for pickling by excluding non-picklable objects."""
        state = self.__dict__.copy()
        # Remove non-picklable objects
        state.pop('prefetch_lock', None)
        state.pop('parallel_loader', None)  # ThreadPoolExecutor not picklable
        return state
    
    def __setstate__(self, state):
        """Restore from pickle by recreating non-picklable objects."""
        self.__dict__.update(state)
        # Recreate non-picklable objects
        self.prefetch_lock = threading.Lock()
        # Recreate parallel loader
        if hasattr(self, 'video_loader'):
            max_workers = getattr(self, 'max_workers', 2)
            self.parallel_loader = ParallelVideoLoader(self.video_loader, max_workers=max_workers)
    
    def __getitem__(self, idx: int) -> Tuple[WorldModelInput, WorldModelOutput]:
        """Optimized data loading with caching and prefetching."""
        # Check prefetch cache first
        with self.prefetch_lock:
            if idx in self.prefetch_cache:
                frame_sequences = self.prefetch_cache.pop(idx)
            else:
                # Find episode and load frames
                episode_indices = list(self.episode_info.keys())
                sequence_count = 0
                target_episode_idx = 0
                
                for episode_idx in episode_indices:
                    episode_data = self.episode_info[episode_idx]
                    episode_length = episode_data["length"]
                    if episode_length < self.config.min_episode_length:
                        continue
                    num_sequences = max(0, episode_length - self.config.num_frames + 1)
                    if self.config.frame_skip > 1:
                        num_sequences = num_sequences // self.config.frame_skip
                    if sequence_count + num_sequences > idx:
                        target_episode_idx = episode_indices.index(episode_idx)
                        break
                    sequence_count += num_sequences
                else:
                    return self._create_dummy_sample()
                
                # Load chunk if needed
                chunk_start = (target_episode_idx // self.chunk_size) * self.chunk_size
                if chunk_start != self.current_chunk_start or len(self.episodes) == 0:
                    self._load_chunk(chunk_start)
                
                # Calculate local index
                chunk_global_start = 0
                for episode_idx in episode_indices[:chunk_start]:
                    episode_data = self.episode_info[episode_idx]
                    episode_length = episode_data["length"]
                    if episode_length < self.config.min_episode_length:
                        continue
                    num_sequences = max(0, episode_length - self.config.num_frames + 1)
                    if self.config.frame_skip > 1:
                        num_sequences = num_sequences // self.config.frame_skip
                    chunk_global_start += num_sequences
                
                local_idx = idx - chunk_global_start
                if local_idx < 0 or local_idx >= len(self.episodes):
                    return self._create_dummy_sample()
                
                episode = self.episodes[local_idx]
                cache_key = f"{episode['episode_index']}_{episode['frame_indices'][0]}_{episode['frame_indices'][-1]}"
                
                # Check frame cache
                frame_sequences = self.frame_cache.get(cache_key)
                if frame_sequences is None:
                    frame_sequences = self._load_frames_optimized(episode)
                    self.frame_cache.put(cache_key, frame_sequences)
        
        # Create model input/output
        video_tensor = self._frames_to_tensor_optimized(frame_sequences)
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
    
    def _create_dummy_sample(self) -> Tuple[WorldModelInput, WorldModelOutput]:
        """Create dummy sample."""
        height, width = self.config.image_size
        video_frames = np.zeros((self.config.num_frames, height, width, 3), dtype=np.float32)
        mask = self.mask_generator.generate_mask(batch_size=1)
        mask = mask.view(mask.size(0), -1)
        
        model_input = WorldModelInput(
            video_frames=jnp.array(video_frames),
            mask=mask,
            camera_names=list(self.config.image_keys),
        )
        model_output = WorldModelOutput(
            predicted_features=jnp.array(video_frames),
            reconstruction_loss=jnp.array(0.0),
            mask_ratio=jnp.array(self.current_mask_ratio),
        )
        return model_input, model_output
    
    def _frames_to_tensor_optimized(self, frames: List[dict]) -> jnp.ndarray:
        """Optimized frame to tensor conversion."""
        tensors = self.batch_processor.batch_frames_to_tensor([frames], self.config)
        return tensors[0] if tensors else jnp.zeros((self.config.num_frames, *self.config.image_size, 3))
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return {
            'frame_cache': self.frame_cache.get_stats(),
            'prefetch_cache_size': len(self.prefetch_cache) if hasattr(self, 'prefetch_cache') else 0,
        }


def create_world_model_data_loader(
    config: WorldModelDataConfig,
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 0,      # Default workers (disabled for spawn compatibility)
    chunk_size: int = 1000,    # Increased default  
    cache_size: int = 200,     # Increased default
    max_workers: int = 2,      # Conservative parallel loading
    prefetch_factor: int = 2,  # Conservative prefetch
    pin_memory: bool = False,  # Disable by default for stability
) -> DataLoader:
    """Create optimized data loader with improved performance."""
    
    dataset = OptimizedWorldModelDataset(
        config=config,
        split=split,
        shuffle=shuffle,
        chunk_size=chunk_size,
        cache_size=cache_size,
        max_workers=max_workers,
        prefetch_size=batch_size * 2,  # Prefetch 2 batches
    )
    
    def collate_fn(batch):
        """Optimized collate function."""
        inputs, outputs = zip(*batch)
        
        def torch_to_jax(tensor):
            if hasattr(tensor, 'detach'):
                return jnp.array(tensor.detach().cpu().numpy())
            return jnp.array(tensor)
        
        batched_input = WorldModelInput(
            video_frames=jnp.stack([inp.video_frames for inp in inputs]),
            mask=jnp.stack([torch_to_jax(inp.mask) for inp in inputs]).squeeze(1),
            camera_names=[inp.camera_names[0] for inp in inputs],
        )
        
        batched_output = WorldModelOutput(
            predicted_features=jnp.stack([out.predicted_features for out in outputs]),
            reconstruction_loss=jnp.stack([out.reconstruction_loss for out in outputs]),
            mask_ratio=jnp.stack([out.mask_ratio for out in outputs]),
        )
        
        return batched_input, batched_output
    
    # Handle prefetch_factor for single-threaded mode
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': False,  # Dataset handles shuffling
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory,
        'drop_last': True,  # Consistent batch sizes
    }
    
    # Only add prefetch_factor if using workers
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = True  # Keep workers alive
    else:
        # Remove prefetch_factor for single-threaded mode
        dataloader_kwargs.pop('prefetch_factor', None)
    
    return DataLoader(**dataloader_kwargs)


