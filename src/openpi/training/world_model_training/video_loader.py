#!/usr/bin/env python3
"""
Video loader for accessing MP4 files directly from the dataset cache.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger("openpi")

class VideoFrameLoader:
    """Load video frames directly from MP4 files."""
    
    def __init__(self, dataset_cache_path: str):
        """
        Initialize the video frame loader.
        
        Args:
            dataset_cache_path: Path to the dataset cache directory
        """
        self.dataset_cache_path = dataset_cache_path
        self.video_cache = {}  # Cache opened video files
        
    def get_video_path(self, episode_idx: int, camera: str = "top_camera-images-rgb") -> str:
        """
        Get the path to the video file for a given episode and camera.
        
        Args:
            episode_idx: Episode index
            camera: Camera name (top_camera-images-rgb, left_camera-images-rgb, right_camera-images-rgb)
            
        Returns:
            Path to the video file
        """
        # Calculate chunk number (assuming 1000 episodes per chunk)
        chunk_num = episode_idx // 1000
        chunk_dir = f"chunk-{chunk_num:03d}"
        
        # Format episode number
        episode_str = f"episode_{episode_idx:06d}.mp4"
        
        # Construct full path
        video_path = os.path.join(
            self.dataset_cache_path,
            "videos",
            chunk_dir,
            camera,
            episode_str
        )
        
        return video_path
    
    def load_frames(self, episode_idx: int, frame_indices: List[int], 
                   camera: str = "top_camera-images-rgb", 
                   target_size: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
        """
        Load specific frames from a video file.
        
        Args:
            episode_idx: Episode index
            frame_indices: List of frame indices to load
            camera: Camera name
            target_size: Target size for resizing frames
            
        Returns:
            List of frames as numpy arrays
        """
        video_path = self.get_video_path(episode_idx, camera)
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            # Return dummy frames
            dummy_frames = []
            for _ in frame_indices:
                dummy_frame = np.zeros((*target_size, 3), dtype=np.float32)
                dummy_frames.append(dummy_frame)
            return dummy_frames
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {video_path}")
            cap.release()
            # Return dummy frames
            dummy_frames = []
            for _ in frame_indices:
                dummy_frame = np.zeros((*target_size, 3), dtype=np.float32)
                dummy_frames.append(dummy_frame)
            return dummy_frames
        
        frames = []
        try:
            for frame_idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}")
                    # Create dummy frame
                    dummy_frame = np.zeros((*target_size, 3), dtype=np.float32)
                    frames.append(dummy_frame)
                    continue
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                if frame.shape[:2] != target_size:
                    frame = cv2.resize(frame, target_size)
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
                
        finally:
            cap.release()
        
        return frames
    
    def get_video_info(self, episode_idx: int, camera: str = "top_camera-images-rgb") -> Optional[dict]:
        """
        Get information about a video file.
        
        Args:
            episode_idx: Episode index
            camera: Camera name
            
        Returns:
            Dictionary with video information or None if video not found
        """
        video_path = self.get_video_path(episode_idx, camera)
        
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': frame_count / fps if fps > 0 else 0
            }
        finally:
            cap.release() 