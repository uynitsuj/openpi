import numpy as np
import json
from pathlib import Path
import viser.transforms as vtf
from openpi.utils.video_processor import resize_and_pad_video, get_video_resolution

def load_episode_data(episode_path: Path, cfg: dict, base_dir: Path, ep_idx: int):
    """Load data for a specific episode."""
    try:
        # Load action data (4x4 transformation matrices)
        action_data = {}
        action_files = [
            "action-left-hand_in_quest_world_frame.npy",
            "action-left-head.npy", 
            "action-left-pos.npy",
            "action-left-quest_world_frame.npy",
            "action-left-hand.npy",
            "action-right-hand_in_quest_world_frame.npy",
            "action-right-head.npy",
            "action-right-pos.npy", 
            "action-right-quest_world_frame.npy",
            "action-right-hand.npy"
        ]
        
        for action_file in action_files:
            file_path = episode_path / action_file
            if file_path.exists():
                action_data[action_file.replace('.npy', '')] = np.load(file_path)
        
        # Load joint data
        joint_data = {}
        joint_files = [
            "left-gripper_pos.npy",
            "left-joint_eff.npy",
            "left-joint_pos.npy", 
            "left-joint_vel.npy",
            "right-gripper_pos.npy",
            "right-joint_eff.npy",
            "right-joint_pos.npy",
            "right-joint_vel.npy"
        ]
        
        for joint_file in joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                joint_data[joint_file.replace('.npy', '')] = np.load(file_path)
        
        # Load timestamp data
        timestamp_data = {}
        timestamp_files = [
            "timestamp.npy",
            "timestamp_end.npy",
            "left_camera-timestamp.npy",
            "right_camera-timestamp.npy", 
            "top_camera-timestamp.npy"
        ]
        
        for timestamp_file in timestamp_files:
            file_path = episode_path / timestamp_file
            if file_path.exists():
                timestamp_data[timestamp_file.replace('.npy', '')] = np.load(file_path)
        
        # Load annotation data if available
        annotation_file = episode_path / "top_camera-images-rgb_annotation.json"
        annotations = None
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
        # Process video files
        video_files = {}
        camera_names = []
        
        if not cfg.skip_videos:
            chunk_id = ep_idx // cfg.chunk_size
            for video_file in episode_path.glob("*.mp4"):
                camera_name = video_file.stem
                video_files[camera_name] = video_file
                ret = resize_and_pad_video(
                    input_path=str(video_file), 
                    output_path=str(base_dir / "videos" / f"chunk-{chunk_id:03d}" / f"{camera_name}" / f"episode_{ep_idx:06d}.mp4"), 
                    target_size=cfg.resize_size, # type: ignore
                    fps=cfg.fps//cfg.temporal_subsample_factor, # type: ignore
                    frame_stride=cfg.temporal_subsample_factor, # type: ignore
                    keep_left_half = True if "top" in camera_name and get_video_resolution(str(video_file))[0]/get_video_resolution(str(video_file))[1] > 3 else False, # if top camera is 3x wider than tall, it's highly likely to be a concat of two videos
                    crop_to_square = cfg.crop_images_to_square # type: ignore
                )
                if not ret:
                    raise ValueError(f"Failed to resize and pad video {video_file}")
        
        return {
            'action_data': action_data,
            'joint_data': joint_data, 
            'timestamp_data': timestamp_data,
            'camera_names': camera_names,
            'annotations': annotations,
            'video_files': video_files
        }
        
    except Exception as e:
        print(f"Error loading episode data: {e}")
        raise


def _load_episode_data(episode_path: Path):
    """Load data for a specific episode (legacy function for backward compatibility)."""
    # This is the original function - keeping for any existing usage
    # But redirecting to the main function
    return load_episode_data(episode_path)


# Add these validation functions after the existing imports and before the XMIConfig class

def validate_episode_data(episode_data: dict) -> tuple[bool, str]:
    """
    Validate that episode data contains all required fields and has valid structure.
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(episode_data, dict):
        return False, "Episode data is not a dictionary"
    
    # Check required top-level keys
    required_keys = ['action_data', 'joint_data']
    for key in required_keys:
        if key not in episode_data:
            return False, f"Missing required key: {key}"
        if not isinstance(episode_data[key], dict):
            return False, f"Key '{key}' is not a dictionary"
    
    # Check required action_data keys
    required_action_keys = [
        'action-left-head',
        # 'action-left-hand_in_quest_world_frame', or 'action-left-hand'
        # 'action-left-quest_world_frame',
        # 'action-right-hand_in_quest_world_frame', or 'action-right-hand'
        # 'action-right-quest_world_frame',
        'action-left-pos',
        'action-right-pos'
    ]

    action_data = episode_data['action_data']
    for key in required_action_keys:
        if key not in action_data:
            return False, f"Missing required action_data key: {key}"
        if action_data[key] is None:
            return False, f"action_data['{key}'] is None"
    
    # Check that either (action-left-hand and action-right-hand) or (action-left-hand_in_quest_world_frame and action-right-hand_in_quest_world_frame) are in action_data
    if 'action-left-hand' not in action_data and 'action-right-hand' not in action_data:
        if 'action-left-hand_in_quest_world_frame' not in action_data and 'action-right-hand_in_quest_world_frame' not in action_data:
            return False, "Either (action-left-hand and action-right-hand) or (action-left-hand_in_quest_world_frame and action-right-hand_in_quest_world_frame) must be in action_data"

    # Check required joint_data keys
    required_joint_keys = ['left-joint_pos', 'right-joint_pos']
    joint_data = episode_data['joint_data']
    for key in required_joint_keys:
        if key not in joint_data:
            return False, f"Missing required joint_data key: {key}"
        if joint_data[key] is None:
            return False, f"joint_data['{key}'] is None"
    
    # Check that arrays have reasonable lengths
    try:
        # Get lengths of key arrays
        action_lengths = []
        for key in required_action_keys:
            if hasattr(action_data[key], '__len__'):
                action_lengths.append(len(action_data[key]))
            else:
                return False, f"action_data['{key}'] is not array-like"
        
        joint_lengths = []
        for key in required_joint_keys:
            if hasattr(joint_data[key], '__len__'):
                joint_lengths.append(len(joint_data[key]))
            else:
                return False, f"joint_data['{key}'] is not array-like"
        
        # Check if all arrays have reasonable and consistent lengths
        all_lengths = action_lengths + joint_lengths
        if len(set(all_lengths)) > 1:  # All arrays must have same length
            return False, f"Inconsistent array lengths: {all_lengths}"
        
        min_length = min(all_lengths)
        if min_length < 2:
            return False, f"Episode too short: {min_length} frames"
            
    except Exception as e:
        return False, f"Error checking array lengths: {str(e)}"
    
    return True, "Valid"


def validate_array_data(data: np.ndarray, name: str, expected_shape: tuple = None) -> tuple[bool, str]:
    """
    Validate numpy array data for common issues.
    
    Args:
        data: numpy array to validate
        name: name of the data for error messages
        expected_shape: optional expected shape (None means any shape is acceptable)
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if data is None:
        return False, f"{name} is None"
    
    if not isinstance(data, np.ndarray):
        return False, f"{name} is not a numpy array (type: {type(data)})"
    
    # Check for NaN values
    if np.any(np.isnan(data)):
        nan_count = np.sum(np.isnan(data))
        return False, f"{name} contains {nan_count} NaN values"
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        inf_count = np.sum(np.isinf(data))
        return False, f"{name} contains {inf_count} infinite values"
    
    # Check dtype
    if not np.issubdtype(data.dtype, np.number):
        return False, f"{name} has non-numeric dtype: {data.dtype}"
    
    # Check shape if specified
    if expected_shape is not None:
        if data.shape != expected_shape:
            return False, f"{name} has incorrect shape: {data.shape}, expected: {expected_shape}"
    
    # Check for reasonable value ranges (detect obvious data corruption)
    if np.any(np.abs(data) > 1e6):  # Very large values might indicate corruption
        max_val = np.max(np.abs(data))
        return False, f"{name} contains suspiciously large values (max: {max_val})"
    
    return True, "Valid"


def validate_records(records: list) -> tuple[bool, str]:
    """
    Validate final records before saving to ensure they're ready for LeRobot format.
    
    Args:
        records: list of record dictionaries
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not records:
        return False, "No records provided"
    
    if not isinstance(records, list):
        return False, f"Records is not a list (type: {type(records)})"
    
    required_keys = ["state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            return False, f"Record {i} is not a dictionary"
        
        # Check required keys
        for key in required_keys:
            if key not in record:
                return False, f"Record {i} missing required key: {key}"
            
            if record[key] is None:
                return False, f"Record {i}['{key}'] is None"
        
        # Validate state and actions arrays
        try:
            state = record["state"]
            actions = record["actions"]
            
            if not isinstance(state, list) or not isinstance(actions, list):
                return False, f"Record {i}: state and actions must be lists"
            
            # if len(state) != 20:
            #     return False, f"Record {i}: state has {len(state)} elements, expected 20"
            
            # if len(actions) != 20:
            #     return False, f"Record {i}: actions has {len(actions)} elements, expected 20"
            
            # Check for None values in state/actions
            for j, val in enumerate(state):
                if val is None:
                    return False, f"Record {i}: state[{j}] is None"
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    return False, f"Record {i}: state[{j}] has invalid value: {val}"
            
            for j, val in enumerate(actions):
                if val is None:
                    return False, f"Record {i}: actions[{j}] is None"
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    return False, f"Record {i}: actions[{j}] has invalid value: {val}"
        
        except Exception as e:
            return False, f"Record {i}: Error validating arrays: {str(e)}"
    
    return True, "Valid"


def validate_images(images: list, camera_key: str) -> tuple[bool, str]:
    """
    Validate image data for a camera.
    
    Args:
        images: list of image arrays
        camera_key: name of the camera for error messages
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not images:
        return False, f"No images for camera {camera_key}"
    
    for i, img in enumerate(images):
        if img is None:
            return False, f"Image {i} for camera {camera_key} is None"
        
        if not isinstance(img, np.ndarray):
            return False, f"Image {i} for camera {camera_key} is not a numpy array"
        
        if len(img.shape) != 3:
            return False, f"Image {i} for camera {camera_key} has wrong dimensions: {img.shape}"
        
        if img.shape[2] != 3:
            return False, f"Image {i} for camera {camera_key} does not have 3 channels: {img.shape}"
        
        # Check for reasonable image values
        if img.dtype == np.uint8:
            if np.any(img > 255) or np.any(img < 0):
                return False, f"Image {i} for camera {camera_key} has invalid uint8 values"
        elif np.issubdtype(img.dtype, np.floating):
            if np.any(img > 1.0) or np.any(img < 0.0):
                return False, f"Image {i} for camera {camera_key} has invalid float values (should be [0,1])"
        else:
            return False, f"Image {i} for camera {camera_key} has unsupported dtype: {img.dtype}"
    
    return True, "Valid"

