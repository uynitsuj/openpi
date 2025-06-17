"""
Data processing utilities for YAMS data conversion.

This module contains functions for loading, processing, and validating YAMS episode data.
"""

import gc
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
try:
    from .video_utils import extract_video_frames_fast, resize_frames_vectorized
except ImportError:
    from video_utils import extract_video_frames_fast, resize_frames_vectorized

# YAMS data configuration
CAMERA_KEYS = [
    "left_camera-images-rgb",
    "right_camera-images-rgb", 
    "top_camera-images-rgb"
]

CAMERA_KEY_MAPPING = {
    "left_camera-images-rgb": "exterior_image_1_left",
    "right_camera-images-rgb": "exterior_image_2_right", 
    "top_camera-images-rgb": "exterior_image_3_top"
}


def find_episode_directories(parent_dir: Path | List[Path]) -> List[Path]:
    """Find all YAMS episode directories."""
    
    # Handle both single path and list of paths
    if isinstance(parent_dir, list):
        parent_dirs = [Path(dir) for dir in parent_dir]
    else:
        parent_dirs = [Path(parent_dir)]

    episode_dirs = []
    for parent in parent_dirs:
        for item in parent.iterdir():
            if item.is_dir() and item.name.startswith("episode_") and item.name.endswith(".npy.mp4"):
                episode_dirs.append(item)
    return sorted(episode_dirs)


def load_episode_annotations(episode_path: Path) -> Dict:
    """Load annotation data for a YAMS episode to check quality labels."""
    annotations = {}
    
    # Look for annotation files for each camera
    camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
    
    for camera_prefix in camera_prefixes:
        annotation_file = episode_path / f"{camera_prefix}_annotation.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                    annotations[camera_prefix] = annotation_data
            except Exception as e:
                print(f"Warning: Could not load annotation file {annotation_file}: {e}")
    
    return annotations


def is_episode_good_quality(episode_path: Path) -> bool:
    """Check if an episode is labeled as 'good' quality based on annotations."""
    annotations = load_episode_annotations(episode_path)
    
    if not annotations:
        return False
    
    # Check if any camera has "good" quality label
    for camera_prefix, annotation_data in annotations.items():
        video_labels = annotation_data.get('video_labels', [])
        for label in video_labels:
            if (label.get('class_description') == 'overall_quality' and 
                label.get('label') == 'good'):
                return True
    
    return False


def load_yams_episode_data_fast(episode_path: Path) -> Optional[Dict]:
    """Load data for a specific YAMS episode with maximum optimizations."""
    episode_data = {
        'metadata': {},
        'joint_data': {},
        'images': {}
    }
    
    try:
        # Load metadata first (small file)
        metadata_file = episode_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                episode_data['metadata'] = json.load(f)
        
        # Load joint data using memory mapping for large files
        essential_joint_files = [
            "left-joint_pos.npy",
            "right-joint_pos.npy", 
            "left-gripper_pos.npy",
            "right-gripper_pos.npy"
        ]
        
        # Load all joint files at once
        joint_data = {}
        for joint_file in essential_joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                # Use memory mapping for faster loading of large arrays
                joint_data[joint_file.replace('.npy', '')] = np.load(file_path, mmap_mode='r')
        
        episode_data['joint_data'] = joint_data
        
        # Load videos with priority and early termination
        video_priorities = ["_crf18.mp4", ".mp4", "_low_res.mp4"]
        camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
        
        # Process videos in parallel within the episode
        for camera_prefix in camera_prefixes:
            for priority_suffix in video_priorities:
                video_file = episode_path / f"{camera_prefix}{priority_suffix}"
                if video_file.exists():
                    frames = extract_video_frames_fast(video_file)
                    if len(frames) > 0:
                        episode_data['images'][camera_prefix] = frames
                    break  # Found video for this camera, move to next
        
        # Force garbage collection after loading large video data
        gc.collect()
        
        return episode_data
        
    except Exception as e:
        print(f"Error loading YAMS episode data from {episode_path}: {e}")
        return None


def process_joint_data(joint_data: Dict) -> Optional[np.ndarray]:
    """Process and combine joint data from YAMS episode."""
    # Check if we have the required joint data
    required_keys = ['left-joint_pos', 'right-joint_pos', 'left-gripper_pos', 'right-gripper_pos']
    if not all(key in joint_data for key in required_keys):
        return None
    
    # Get joint positions and gripper positions (use views, not copies)
    left_joint_pos = joint_data['left-joint_pos']  # Shape: (N, 6)
    right_joint_pos = joint_data['right-joint_pos']  # Shape: (N, 6)
    left_gripper_pos = joint_data['left-gripper_pos']  # Shape: (N, 1)
    right_gripper_pos = joint_data['right-gripper_pos']  # Shape: (N, 1)
    
    # Apply joint position flipping (create copies only when needed)
    left_joint_pos = np.flip(left_joint_pos, axis=1).copy()  # Copy needed for flip
    right_joint_pos = np.flip(right_joint_pos, axis=1).copy()  # Copy needed for flip
    
    # Pre-allocate full state array to avoid multiple concatenations
    seq_length = len(left_joint_pos)
    full_joint_state = np.empty((seq_length, 14), dtype=np.float32)
    
    # Fill the array efficiently
    full_joint_state[:, :6] = left_joint_pos
    full_joint_state[:, 6:7] = left_gripper_pos
    full_joint_state[:, 7:13] = right_joint_pos
    full_joint_state[:, 13:14] = right_gripper_pos
    
    return full_joint_state


def process_images(images: Dict, seq_length: int, resize_size: int, skip_videos: bool = False) -> Dict:
    """Process and resize images from YAMS episode."""
    processed_images = {}
    
    if not skip_videos:
        for camera_key in CAMERA_KEYS:
            if camera_key in images:
                camera_frames = images[camera_key][:seq_length]
                
                # Skip stereo processing for now (commented out by user)
                # if "top" in camera_key and camera_frames.shape[2] > camera_frames.shape[1]:
                #     camera_frames = camera_frames[:, :, :camera_frames.shape[2]//2, :]
                
                # Batch resize all frames at once using fastest method
                resized_frames = resize_frames_vectorized(camera_frames, resize_size)
                processed_images[camera_key] = resized_frames
    
    return processed_images

def calculate_actions(full_joint_state: np.ndarray, seq_length: int):
    joint_states = full_joint_state[:seq_length]
    joint_actions = joint_states.copy()  # absolute actions = joint state itself
    return joint_states, joint_actions

# def calculate_actions(full_joint_state: np.ndarray, seq_length: int) -> tuple: # RELATIVE, DOES NOT SEEM TO WORK WELL
#     """Calculate joint actions from joint states."""
#     # Pre-calculate all actions at once (vectorized)
#     joint_states = full_joint_state[:seq_length]  # Current states
#     next_states = full_joint_state[1:seq_length+1]  # Next states
    
#     # Calculate delta actions vectorized
#     joint_actions = next_states - joint_states
    
#     # For grippers, use absolute position instead of delta (vectorized)
#     joint_actions[:, 6] = next_states[:, 6]    # left gripper absolute
#     joint_actions[:, 13] = next_states[:, 13]  # right gripper absolute
    
#     return joint_states, joint_actions


def create_frame_data(joint_states: np.ndarray, joint_actions: np.ndarray, 
                     processed_images: Dict, seq_length: int, skip_videos: bool = False) -> List[Dict]:
    """Create frame data for LeRobot dataset."""
    frames_data = []
    
    for step in range(seq_length):
        # Use pre-calculated values (no computation in loop)
        frame_data = {
            "joint_positions": joint_states[step],
            "actions": joint_actions[step],
        }
        
        # Add camera images (direct indexing, no copies) - only if not skipping videos
        if not skip_videos:
            for camera_key in CAMERA_KEYS:
                if camera_key in processed_images and camera_key in CAMERA_KEY_MAPPING:
                    lerobot_key = CAMERA_KEY_MAPPING[camera_key]
                    frame_data[lerobot_key] = processed_images[camera_key][step]
        
        frames_data.append(frame_data)
    
    return frames_data