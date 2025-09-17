#!/usr/bin/env python3
"""
Fast XMI to LeRobot format converter.

This script bypasses the LeRobot dataset creation completely and directly creates
the dataset in the same format as LeRobot, avoiding memory accumulation and 
ffmpeg-python import issues while keeping XMI-specific transform calculations.

Usage:
uv run openpi/scripts/xmi_data/convert_xmi_data_fast.py

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run openpi/scripts/xmi_data/convert_xmi_data_fast.py --push_to_hub
"""

import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple
import numpy as np
import tyro
import gc
import shutil
import viser.transforms as vtf
from openpi.utils.xmi_dataloader_utils import load_episode_data, validate_episode_data, validate_array_data, validate_records, validate_images
from openpi.utils.matrix_utils import *
from openpi.utils.key_frame_select_utils import zed_tf_intrinsics, select_keyframes_helper
from copy import deepcopy
from collections import deque

# Video processing imports for timestamp validation
try:
    import torch
    import torchvision
    HAS_VIDEO_PROCESSING = True
except ImportError:
    print("Warning: Video processing libraries not available. Video timestamp validation disabled.")
    HAS_VIDEO_PROCESSING = False

# Set environment variable for dataset storage

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import write_episode_stats
    HAS_LEROBOT = True
except ImportError:
    print("Warning: LeRobot not available. Hub push functionality disabled.")
    HAS_LEROBOT = False

# TODO: remove this once we have a better way to handle this
DEG30_MOUNTS = ["20250725", "20250724", "20250727", "20250728", "20250801", "20250804", "20250811", "20250812", "20250815", "20250818", "20250820", "20250822", "20250825", "20250905", "20250908", "20250909", "20250910", "20250911"]

@dataclass
class XMIConfig:
    # Input data paths
    raw_dataset_folders: List[str] = field(default_factory=lambda: [
        ## Hummus data
        # "/nfs_us/hummus_xmi_data/pick_place_beverage",
        # "/nfs_us/hummus_xmi_data/pick_place_chips_bag",
        # "/nfs_us/hummus_xmi_data/pick_place_candy_bag",
        # "/nfs_us/hummus_xmi_data/pick_place_cleaning_sponge",
        # "/nfs_us/hummus_xmi_data/pick_place_dish_detergent",
        # "/nfs_us/hummus_xmi_data/pick_place_paper_towel",
        # "/nfs_us/hummus_xmi_data/pick_place_shaving_razor",
        # "/nfs_us/hummus_xmi_data/pick_place_soup_can",
        # "/nfs_us/hummus_xmi_data/pick_place_toothpaste",
        # "/nfs_us/hummus_xmi_data/pick_place_yoghurt",

        ## US XMI-RBY1 Coffee Cup Data
        # "/home/justinyu/Downloads/20250630",
        # "/home/justinyu/Downloads/data_20250708",

        ## US XMI-RBY1 In-Domain Soup Can Data
        # "/nfs_us/data/us_xmi_01/20250714",
        # "/nfs_us/data/us_xmi_01/20250715",
        # "/nfs_us/data/us_xmi_01/20250717/hand_off",
        # "/nfs_us/data/us_xmi_01/20250717/missed_grasps", # Hmmm maybe don't include this causes gripper to prematurely predict close gripper action in deployment which has the opposite intended effect
        # "/nfs_us/data/us_xmi_01/20250724", 
        # "/nfs_us/data/us_xmi_01/20250725", 
        # "/nfs_us/data/us_xmi_01/20250728",

        # "/nfs_us/data/us_xmi_01/20250727",# Better head data than prior data
        # "/nfs_us/data/us_xmi_01/20250801",
        # "/nfs_us/data/us_xmi_01/20250804",
        # "/nfs_us/data/us_xmi_01/20250811_tabletop_soup_can",

        # Shelf Soup Can Data
        # "/nfs_us/data/us_xmi_01/20250806_shelf_soup",
        # "/nfs_us/data/us_xmi_01/20250807_shelf_soup",

        # "/nfs_us/data/us_xmi_01/20250812_shelf_soup", # Better head data than prior data
        # "/nfs_us/data/us_xmi_01/20250815_shelf_soup",
        # "/nfs_us/data/us_xmi_01/20250818_shelf_soup",
        # "/nfs_us/data/us_xmi_01/20250820_shelf_soup",
        # "/nfs_us/data/us_xmi_01/20250822_shelf_soup",
        # "/nfs_us/data/us_xmi_01/20250825_shelf_soup",

        # Dishrack unload data
        # "/nfs_us/data/us_xmi_01/20250801_dishrack_unload",
        # "/nfs_us/data/us_xmi_01/20250823_dishrack_unload",

        # Identify the fruit to your left and transfer it from the shelf to the bowl
        # "/nfs_us/data/us_xmi_01/20250905_sort_fruit_memory",
        # "/nfs_us/data/us_xmi_01/20250908_sort_fruit_memory_tabletop",

        # Identify the fruit to your left and transfer it to the plate
        # "/nfs_us/data/us_xmi_01/20250909_sort_fruit_memory_plate",

        # Identify the item to your left and transfer it to the shopping basket
        "/nfs_us/data/us_xmi_01/20250909_sort_item_memory_shopping_basket",
        "/nfs_us/data/us_xmi_01/20250910",
        "/nfs_us/data/us_xmi_01/20250911",

        # ## Oreo data
        # # "/nfs_us/data/oreo_xmi/clean_whiteboard", 
        # "/nfs_us/data/oreo_xmi/fold_napkin_place_utensils_inside_and_roll_it_up",
        # "/nfs_us/data/oreo_xmi/folding_skirt_pile_and_stacking",
        # "/nfs_us/data/oreo_xmi/folding_trousers_pile_and_stacking",
        # "/nfs_us/data/oreo_xmi/folding_tshirt_pile_and_stacking",
        # "/nfs_us/data/oreo_xmi/insert_the_plug",
        # "/nfs_us/data/oreo_xmi/packing_luggage",
        # # "/nfs_us/data/oreo_xmi/painting_nails",
        # "/nfs_us/data/oreo_xmi/place_fake_bread",
        # "/nfs_us/data/oreo_xmi/place_trousers_on_hanger",
        # "/nfs_us/data/oreo_xmi/place_tshirt_on_hanger",
        # # "/nfs_us/data/oreo_xmi/put_pillow_into_pillowcase",
        # "/nfs_us/data/oreo_xmi/serve_a_lunch_box",
        # "/nfs_us/data/oreo_xmi/sorting_stationery_into_containers",
        # "/nfs_us/data/oreo_xmi/untangling_cables",
        # "/nfs_us/data/oreo_xmi/zip_up_a_jacket",

        ## SZ XMI-RBY OOD Data
        # "/nfs_us/data/sz_xmi_02/pick_place_cleaning_sponge",
        # "/home/justinyu/nfs_us/data/sz_xmi_02/pick_place_soup_can",

    ])
    
    # Language instructions corresponding to each dataset folder (ANNOTATION OVERRIDES)
    language_instructions: List[str] = field(default_factory=lambda: [
        ## Hummus data
        # "pick up the beverage and place it in the bin",
        # "pick up the chips bag and place it in the bin",
        # "pick up the candy bag and place it in the bin",
        # "pick up the cleaning sponge and place it in the bin",
        # "pick up the dish detergent and place it in the bin",
        # "pick up the paper towel and place it in the bin",
        # "pick up the shaving razor and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the toothpaste and place it in the bin",
        # "pick up the yogurt and place it in the bin",

        ## US XMI-RBY1 Coffee Cup Data
        # "place the coffee cup on the dish",
        # "place the coffee cup on the dish",

        ## US XMI-RBY1 In-Domain Soup Can Data
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",

        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the soup can and place it in the bin",


        # Shelf Soup Can Data
        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",

        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",
        # "put the soup can in the shopping basket",

        # Dishrack Unload Data
        # "unload the dishes from the dishrack",
        # "unload the dishes from the dishrack",

        # Identify the fruit to your left and transfer it from the shelf to the bowl
        # "Identify the fruit to your left and transfer it from the shelf to the bowl",

        # Identify the fruit to your left and transfer it to the plate
        # "Identify the fruit to your left and transfer it to the plate",

        # Identify the item to your left and transfer it to the shopping basket
        "Identify the item to your left and transfer it to the shopping basket",
        "Identify the item to your left and transfer it to the shopping basket",
        "Identify the item to your left and transfer it to the shopping basket",


        # ## Oreo data
        # # "clean the whiteboard with the eraser",
        # "place the utensils inside the napkin and roll it up",
        # "fold the skirt and stack it in a neat pile",
        # "fold the trousers and stack it in a neat pile",
        # "fold the tshirt and stack it in a neat pile",
        # "insert the plug into the socket",
        # "pack the luggage",
        # # "paint the nails",
        # "place the bread on the plate",
        # "place the trousers on the hanger",
        # "place the tshirt on the hanger",
        # # "put the pillow into the pillowcase",
        # "serve the lunch box",
        # "sort the stationery into the containers",
        # "untangle the cables and put them in the bin",
        # "zip up the jacket",

        ## SZ XMI-RBY OOD Data
        # "pick up the cleaning sponge and place it in the bin",
        # "pick up the soup can and place it in the bin",
    ])
    
    # Repository name for output dataset
    repo_name: str = "uynitsuj/sort_item_memory_xmi_data_20250911_w_negative_trajs"
    
    # Camera settings
    camera_keys: List[str] = field(default_factory=lambda: [
        "left_camera-images-rgb",
        "right_camera-images-rgb", 
        "top_camera-images-rgb"
    ])
    
    camera_key_mapping: dict = field(default_factory=lambda: {
        "left_camera-images-rgb": "left_camera-images-rgb",
        "right_camera-images-rgb": "right_camera-images-rgb", 
        "top_camera-images-rgb": "top_camera-images-rgb"
    })
    
    # Calibration files TODO: very bad, going forward calibration data should be attached to and extracted from episode metadata
    left_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Left_Controller_20250603_15/calib_results/controller2franka.npy"
    right_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Right_Controller_20250603_15/calib_results/controller2franka.npy"
    quest_to_zed_calib: str = "/nfs_us/justinyu/us_xmi_calib/Head_Franka_20250604_12/calib_results/head2cam.npy"
    
    # Processing settings
    resize_size: int = 224
    fps: int = 30 # Framerate of original video
    temporal_subsample_factor: int = 1 # Subsample every N frames (1 = no subsampling)
    crop_images_to_square: bool = True # Whether to crop images to square (if False, will keep original aspect ratio and pad with black instead)
    chunk_size: int = 1000
    max_workers: int = 10
    max_episodes: Optional[int] = None
    skip_videos: bool = False
    first_frame_head_reorient: bool = False
    no_filter_quality: bool = True  # If True, will not filter out low quality episodes
    include_head_pose: bool = True # Whether to include head state in the lerobot proprio state and action (for head / active vision retargeting) NOTE: makes state dim and action dim 29 instead of 20 (adds 6d rot 3d pos)

    keyframe_max_len: int = 2 # Number of past keyframes to log for keyframe history model training

    gripper_action_delay_tsteps: int = 9 # Number of timesteps to delay the gripper action by
    move_old_handoffs: bool = False # Whether to move old handoffs from the dataset to a separate folder named old_handoffs (detected by checking if gripper action minimum is low for both grippers in one trajectory)

    perturb_z_height: bool = True # Whether to perturb the z height of the head and hands around average tabletop height
    perturb_z_height_range: Tuple[float, float] = (0.6, 1.0) # Range of z height of hands perturbation (uniformly sampled)

    perturb_xy_position: bool = True # Whether to perturb the xy position of the head and hands around average tabletop position
    perturb_x_position_range: Tuple[float, float] = (-0.01, 0.01) # Range of x position perturbation (uniformly sampled)
    perturb_y_position_range: Tuple[float, float] = (-0.02, 0.02) # Range of y position perturbation (uniformly sampled)

    delta_proprio_keys: str = None # Makes the proprioceptive state for this axis delta. Can set to None to keep both state and action absolute

    # Validation settings
    # max_se3_diff_in_meters: float = 0.20  # Maximum allowed SE3 difference in meters
    max_se3_diff_in_meters: float = 0.5  # Maximum allowed SE3 difference in meters
    # max_se3_diff_in_degrees: float = 11  # Maximum allowed SE3 difference in degrees
    max_se3_diff_in_degrees: float = 40  # Maximum allowed SE3 difference in degrees
    video_timestamp_tolerance_s: float = 0.0001  # Maximum allowed deviation from expected frame interval for video validation (seconds)
    
    # Hub settings
    push_to_hub: bool = False
    push_to_hub_only: bool = False
    
    # Memory management
    max_frames_per_chunk: int = 1000
    
    # Video encoding settings
    benchmark_encoders: bool = True
    encoder_name: Optional[str] = None
    encoding_quality: str = 'fast'

    # Debug settings
    debug: bool = False
    problematic_data_dir: Optional[str] = None  # Directory to move problematic episodes
    
    def __post_init__(self):
        # Set default problematic data directory if not specified
        if self.problematic_data_dir is None:
            self.problematic_data_dir = f"./problematic_episodes_{self.repo_name.replace('/', '_')}"


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img

def find_episode_directories(raw_dataset_folders: List[str]) -> List[Path]:
    """Find all episode directories from the input folders."""
    episode_dirs = []
    
    for folder in raw_dataset_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        # Find all subdirectories that contain episodes
        trajs = [d for d in folder_path.iterdir() if d.is_dir()]
        episode_dirs.extend(trajs)
    
    return sorted(episode_dirs)



def load_episode_annotations(episode_path: Path) -> dict:
    """Load annotation data for a YAMS episode to check quality labels."""
    annotations = {}

    # Look for annotation files for each camera
    camera_prefixes = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

    for camera_prefix in camera_prefixes:
        annotation_file = episode_path / f"{camera_prefix}_annotation.json"
        if annotation_file.exists():
            try:
                with open(annotation_file) as f:
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
        video_labels = annotation_data.get("video_labels", [])
        for label in video_labels:
            if label.get("class_description") == "overall_quality" and label.get("label") == "good":
                return True

    return False


def move_problematic_episode(episode_path: Path, error_msg: str, cfg: XMIConfig, episode_idx: int = None) -> bool:
    """
    Move a problematic episode to a separate directory for analysis.
    
    Args:
        episode_path: Path to the problematic episode
        error_msg: Error message describing the problem
        cfg: Configuration object
        episode_idx: Optional episode index for logging
        
    Returns:
        bool: True if move was successful, False otherwise
    """
    try:
        # Create problematic data directory
        problematic_dir = Path(cfg.problematic_data_dir)
        problematic_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory based on error type
        error_type = error_msg.split(':')[0].replace(' ', '_').replace('❌', '').strip()
        error_subdir = problematic_dir / error_type
        error_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate destination path
        dest_path = error_subdir / episode_path.name
        
        # Move the episode directory
        if episode_path.exists():
            shutil.move(str(episode_path), str(dest_path))
            
            # Create error log file
            error_log_path = dest_path / "error_log.txt"
            with open(error_log_path, 'w') as f:
                f.write(f"Episode Index: {episode_idx}\n")
                f.write(f"Original Path: {episode_path}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Moved At: {pd.Timestamp.now()}\n")
            
            episode_str = f"episode {episode_idx}" if episode_idx is not None else "episode"
            print(f"  📁 Moved problematic {episode_str} to: {dest_path}")
            return True
        else:
            print(f"  ⚠️  Episode path does not exist: {episode_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Failed to move problematic episode: {e}")
        return False


def validate_video_timestamp_synchronization(
    episode_path: Path,
    cfg: XMIConfig,
    episode_idx: int = None,
    tolerance_s: float = 0.1
) -> tuple[bool, Optional[str]]:
    """
    Validate that video timestamps have consistent frame intervals matching the expected FPS.
    
    This function checks if video files can be loaded and if their frame intervals
    are consistent with the expected frame rate. This prevents timestamp synchronization 
    errors that would cause failures during downstream processing.
    
    Args:
        episode_path: Path to the episode directory
        cfg: Configuration object containing camera and timing settings
        episode_idx: Optional episode index for logging
        tolerance_s: Maximum allowed deviation from expected frame interval (seconds)
        
    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not HAS_VIDEO_PROCESSING:
        # Skip validation if video processing libraries aren't available
        return True, None
    
    if cfg.skip_videos:
        # Skip validation if videos aren't being processed
        return True, None
    
    try:
        # Expected frame interval based on original video fps (before subsampling)
        expected_frame_interval = 1.0 / cfg.fps
        
        # Check each camera video
        for cam_key in cfg.camera_keys:
            # Look for video files in the episode directory
            video_patterns = [
                episode_path / f"{cam_key}.mp4",
                episode_path / f"{cam_key}.avi", 
                episode_path / f"{cam_key}.mov"
            ]
            
            video_path = None
            for pattern in video_patterns:
                if pattern.exists():
                    video_path = pattern
                    break
            
            if video_path is None:
                continue  # Skip if no video file found for this camera
            
            # Set torchvision backend
            torchvision.set_video_backend("pyav")
            
            try:
                # Load video metadata
                reader = torchvision.io.VideoReader(str(video_path), "video")
                
                # Get video info
                video_info = reader.get_metadata()["video"]
                video_fps = video_info["fps"][0] if isinstance(video_info["fps"], list) else video_info["fps"]
                video_duration = video_info["duration"][0] if isinstance(video_info["duration"], list) else video_info["duration"]
                
                # Check if video FPS matches expected FPS
                fps_tolerance = 1.0  # Allow 1 FPS difference
                if abs(video_fps - cfg.fps) > fps_tolerance:
                    reader = None
                    return False, f"Video {cam_key} FPS mismatch: {video_fps} != {cfg.fps} (expected)"
                
                if video_duration < 0.1:  # Less than 100ms
                    reader = None
                    return False, f"Video {cam_key} too short: {video_duration}s duration"
                
                # Load the first several frames to check timestamp consistency
                reader.seek(0.0, keyframes_only=True)
                
                loaded_ts = []
                frame_count = 0
                
                for frame in reader:
                    current_ts = frame["pts"]
                    loaded_ts.append(current_ts)
                    frame_count += 1
                    
                    # Get enough frames to check consistency (at least 5)
                    if frame_count >= 10:
                        break
                
                reader.container.close()
                reader = None
                
                if len(loaded_ts) < 3:
                    return False, f"Video {cam_key} could not load sufficient frames for validation (got {len(loaded_ts)})"
                
                # Check frame interval consistency
                frame_intervals = []
                for i in range(1, len(loaded_ts)):
                    interval = loaded_ts[i] - loaded_ts[i-1]
                    frame_intervals.append(interval)
                
                # Calculate statistics of frame intervals
                mean_interval = np.mean(frame_intervals)
                max_deviation = np.max(np.abs(np.array(frame_intervals) - mean_interval))
                
                # Check if frame intervals are consistent with expected FPS
                expected_deviation = abs(mean_interval - expected_frame_interval)
                if expected_deviation > tolerance_s:
                    return False, (
                        f"Video {cam_key} frame interval mismatch: "
                        f"mean={mean_interval:.4f}s, expected={expected_frame_interval:.4f}s, "
                        f"deviation={expected_deviation:.4f}s > {tolerance_s}s tolerance"
                    )
                
                # Check if frame intervals are internally consistent
                if max_deviation > tolerance_s:
                    return False, (
                        f"Video {cam_key} inconsistent frame intervals: "
                        f"max_deviation={max_deviation:.4f}s > {tolerance_s}s tolerance. "
                        f"Intervals: {[f'{x:.4f}' for x in frame_intervals[:5]]}"
                    )
                
            except Exception as e:
                if 'reader' in locals() and reader is not None:
                    try:
                        reader.container.close()
                    except:
                        pass
                    reader = None
                return False, f"Video {cam_key} validation failed: {str(e)}"
        
        return True, None
        
    except Exception as e:
        return False, f"Video timestamp validation error: {str(e)}"


def process_xmi_transforms(episode_data: dict, cfg: XMIConfig, episode_path: Path = None, episode_idx: int = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process XMI episode data and apply all coordinate transformations.
    Returns (states, actions) as numpy arrays with shape (seq_length, 20).
    """
    
    # VALIDATION: Check episode data structure
    is_valid, error_msg = validate_episode_data(episode_data)
    if not is_valid:
        print(f"❌ Episode data validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Episode data validation failed: {error_msg}", cfg, episode_idx)
        return None, None
    
    # Extract relevant data
    action_data = episode_data['action_data'] # (oculus gripper action)
    joint_data = episode_data['joint_data'] # (robotiq gripper proprio)
    
    # Load controller calibration transformations TODO: search for calibration data in metadata, fallback to this if not found for backward compatibility
    left_controller_calib_tf = vtf.SE3.from_matrix(np.load(cfg.left_controller_calib)).inverse()
    right_controller_calib_tf = vtf.SE3.from_matrix(np.load(cfg.right_controller_calib)).inverse()
    quest_to_zed_calib_tf = vtf.SE3.from_matrix(np.load(cfg.quest_to_zed_calib)).inverse()
    
    # Transform from Quest coordinate system to world coordinate system
    q2w = vtf.SE3.from_rotation_and_translation(
        vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
    )

    # HEAD PROCESSING
    
    head_z_tf = vtf.SE3.from_matrix(action_data["action-left-head"][0])
    head_data_all = vtf.SE3.from_matrix(action_data["action-left-head"])

    head_data_all = q2w @ head_data_all
    head_z_tf = q2w @ head_z_tf

    # Some hardcoded hacks for old data TODO: remove these hacks eventually
    #########################################################

    metadata_file = episode_path / "metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    # else:
    #     raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        if "config_path" in metadata.keys() and 'sz' in metadata["config_path"][0]:
            sz_to_us_tf = vtf.SE3.from_matrix(np.array([ # TODO: very bad temporary hack for hummus data, going forward calibration data should extracted in the metadata
                [1.0,      0.0,        0.0,        0.0],
                [0.0,  0.956304,  0.292371,  0.02955328],
                [0.0,  -0.292371,   0.956304,  0.00117534],
                [0.0,      0.0,        0.0,        1.0]
            ]))
            quest_to_zed_calib_tf = quest_to_zed_calib_tf @ sz_to_us_tf.inverse()
    if metadata is not None:
        if "start_datetime" in metadata.keys() and any(mount in metadata["start_datetime"] for mount in DEG30_MOUNTS):
            old_mount_to_new_mount_tf = vtf.SE3.from_rotation_and_translation(
                vtf.SO3.from_rpy_radians(-25 * np.pi/180, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
            )
            quest_to_zed_calib_tf = quest_to_zed_calib_tf @ old_mount_to_new_mount_tf.inverse() # TODO: REMOVE HACK EVENTUALLY
    else:
        old_mount_to_new_mount_tf = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.from_rpy_radians(-25 * np.pi/180, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
        )
        quest_to_zed_calib_tf = quest_to_zed_calib_tf @ old_mount_to_new_mount_tf.inverse() # TODO: REMOVE HACK EVENTUALLY

    #########################################################


    head_translation = np.mean(head_data_all.wxyz_xyz[:, -3:], axis=0)
    head_translation[1] = -head_translation[1]
    head_translation[2] = 0.0
    if cfg.first_frame_head_reorient:
        # Determine direction that head z axis is pointing in the first frame to reorient the RBY1 base frame
        head_z_axis_rot = vtf.SO3.from_rpy_radians(
            -head_z_tf.rotation().as_rpy_radians().roll,
            head_z_tf.rotation().as_rpy_radians().pitch,
            -head_z_tf.rotation().as_rpy_radians().yaw
        )

        head_z_axis = head_z_axis_rot.as_matrix()[:, 2]

        # Project onto x-y plane
        head_z_axis_xy = head_z_axis[:2]
        head_z_axis_xy = head_z_axis_xy / np.linalg.norm(head_z_axis_xy)

        # Convert to angle
        head_z_axis_angle = np.arctan2(head_z_axis_xy[1], head_z_axis_xy[0])

        rby1_base_frame_wxyz = vtf.SO3.from_rpy_radians(0.0, 0.0, head_z_axis_angle).wxyz
    else:
        # If first frame head reorientation is not enabled, use the average of the two controller orientations to reorient the RBY1 base frame
        def mean_angle(thetas):
            """Circular mean of a list/array of angles in radians."""
            return np.angle(np.exp(1j * np.asarray(thetas)).mean())

        if "action-left-hand" in action_data.keys():
            left_hand_matrix = action_data["action-left-hand"][0]
            left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
            left_hand_tf = q2w @ left_hand_tf
        elif "action-left-hand_in_quest_world_frame" in action_data.keys():
            left_hand_matrix = action_data["action-left-hand_in_quest_world_frame"][0]
            world_frame = action_data["action-left-quest_world_frame"][0]
            left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
            left_hand_tf = q2w @ vtf.SE3.from_matrix(world_frame) @ left_hand_tf
        else:
            raise ValueError("No left hand data found")

        left_hand_tf.wxyz_xyz[:4] = left_hand_tf.rotation().normalize().wxyz

        left_hand_tf_pos = left_hand_tf.wxyz_xyz[-3:]
        left_hand_tf_pos[1] = -left_hand_tf_pos[1]

        left_hand_tf_reflected = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(
            -left_hand_tf.rotation().as_rpy_radians().roll,
            left_hand_tf.rotation().as_rpy_radians().pitch,
            -left_hand_tf.rotation().as_rpy_radians().yaw,
        ), left_hand_tf_pos)

        # Add end effector TCP frame with offset (same as combined viewer)
        pitch_180 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0), np.array([0.0, 0.0, 0.0]))
        yaw_45 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4), np.array([0.0, 0.0, 0.0]))
        offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.identity(), np.array([-0.08275, 0.0, 0.005]))
        ee_tf = yaw_45 @ offset @ pitch_180

        tf_left_ee_ik_target = left_hand_tf_reflected @ left_controller_calib_tf @ ee_tf

        hand_z_axis = tf_left_ee_ik_target.as_matrix()[:, 2] 
        hand_z_axis_angle_left = np.arctan2(hand_z_axis[1], hand_z_axis[0])

        if "action-right-hand" in action_data.keys():
            right_hand_matrix = action_data["action-right-hand"][0]
            right_hand_tf = vtf.SE3.from_matrix(right_hand_matrix)
            right_hand_tf = q2w @ right_hand_tf
        elif "action-right-hand_in_quest_world_frame" in action_data.keys():
            right_hand_matrix = action_data["action-right-hand_in_quest_world_frame"][0]
            right_world_frame = action_data["action-right-quest_world_frame"][0]
            right_hand_in_world = np.linalg.inv(world_frame) @ right_world_frame @ right_hand_matrix
            right_hand_tf = vtf.SE3.from_matrix(right_hand_in_world)
            right_hand_tf = q2w @ vtf.SE3.from_matrix(right_world_frame) @ right_hand_tf
        else:
            raise ValueError("No right hand data found")

        right_hand_tf.wxyz_xyz[:4] = right_hand_tf.rotation().normalize().wxyz

        right_hand_tf_pos = right_hand_tf.wxyz_xyz[-3:]
        right_hand_tf_pos[1] = -right_hand_tf_pos[1]

        right_hand_tf_reflected = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(
            -right_hand_tf.rotation().as_rpy_radians().roll,
            right_hand_tf.rotation().as_rpy_radians().pitch,
            -right_hand_tf.rotation().as_rpy_radians().yaw,
        ), right_hand_tf_pos)

        tf_right_ee_ik_target = right_hand_tf_reflected @ right_controller_calib_tf @ ee_tf

        hand_z_axis = tf_right_ee_ik_target.as_matrix()[:, 2]
        hand_z_axis_angle_right = np.arctan2(hand_z_axis[1], hand_z_axis[0])
        # Average the two angles
        avg_angle = mean_angle([hand_z_axis_angle_left, hand_z_axis_angle_right])

        rby1_base_frame_wxyz = (vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi + avg_angle)).wxyz

    rby1_base_frame_position = head_translation

    # LEFT HAND PROCESSING
    world_frame = None
    if "action-left-hand" in action_data.keys():
        left_hand_matrix = action_data["action-left-hand"]
        left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
        left_hand_tf = q2w @ left_hand_tf
    elif "action-left-hand_in_quest_world_frame" in action_data.keys():
        left_hand_matrix = action_data["action-left-hand_in_quest_world_frame"]
        world_frame = action_data["action-left-quest_world_frame"]
        left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
        left_hand_tf = q2w @ vtf.SE3.from_matrix(world_frame) @ left_hand_tf
    else:
        raise ValueError("No left hand data found")

    left_hand_tf.wxyz_xyz[:, :4] = left_hand_tf.rotation().normalize().wxyz

    left_hand_tf_pos = left_hand_tf.wxyz_xyz[:, -3:]
    left_hand_tf_pos[:, 1] = -left_hand_tf_pos[:, 1]

    left_hand_tf_reflected = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(
        -left_hand_tf.rotation().as_rpy_radians().roll,
        left_hand_tf.rotation().as_rpy_radians().pitch,
        -left_hand_tf.rotation().as_rpy_radians().yaw,
    ), left_hand_tf_pos)

    # Add end effector TCP frame with offset (same as combined viewer)
    pitch_180 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0), np.array([0.0, 0.0, 0.0]))
    yaw_45 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4), np.array([0.0, 0.0, 0.0]))
    offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.identity(), np.array([-0.08275, 0.0, 0.005])) # (WE ARE NOT ADDING TCP TO FLANGE HERE, THIS GOES TO XMI FLANGE)
    ee_tf = yaw_45 @ offset @ pitch_180

    tf_left_ee_ik_target = left_hand_tf_reflected @ left_controller_calib_tf @ ee_tf # TODO: left_controller_calib_tf np.load from file should eventually be replaced with metadata extraction

    left_ee_ik_target_handle_position = tf_left_ee_ik_target.wxyz_xyz[:, -3:]
    left_ee_ik_target_handle_wxyz = tf_left_ee_ik_target.wxyz_xyz[:, :4]

    # RIGHT HAND PROCESSING
    
    if "action-right-hand" in action_data.keys():
        right_hand_matrix = action_data["action-right-hand"]
        right_hand_tf = vtf.SE3.from_matrix(right_hand_matrix)
        right_hand_tf = q2w @ right_hand_tf
    elif "action-right-hand_in_quest_world_frame" in action_data.keys():
        right_hand_matrix = action_data["action-right-hand_in_quest_world_frame"]
        right_world_frame = action_data["action-right-quest_world_frame"]
        right_hand_in_world = np.linalg.inv(world_frame) @ right_world_frame @ right_hand_matrix
        right_hand_tf = vtf.SE3.from_matrix(right_hand_in_world)
        right_hand_tf = q2w @ vtf.SE3.from_matrix(right_world_frame) @ right_hand_tf
    else:
        raise ValueError("No right hand data found")

    right_hand_tf.wxyz_xyz[:, :4] = right_hand_tf.rotation().normalize().wxyz

    right_hand_tf_pos = right_hand_tf.wxyz_xyz[:, -3:]
    right_hand_tf_pos[:, 1] = -right_hand_tf_pos[:, 1]

    right_hand_tf_reflected = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(
        -right_hand_tf.rotation().as_rpy_radians().roll,
        right_hand_tf.rotation().as_rpy_radians().pitch,
        -right_hand_tf.rotation().as_rpy_radians().yaw,
    ), right_hand_tf_pos)

    tf_right_ee_ik_target = right_hand_tf_reflected @ right_controller_calib_tf @ ee_tf # TODO: right_controller_calib_tf np.load from file should eventually be replaced with metadata extraction

    right_ee_ik_target_handle_position = tf_right_ee_ik_target.wxyz_xyz[:, -3:]
    right_ee_ik_target_handle_wxyz = tf_right_ee_ik_target.wxyz_xyz[:, :4]

    # Transform end-effector poses to be relative to RBY1 base frame
    # Create RBY1 base frame transformation
    rby1_base_transform = vtf.SE3.from_rotation_and_translation(
        vtf.SO3(wxyz=rby1_base_frame_wxyz), 
        rby1_base_frame_position
    )
    # Invert to get transformation from world to RBY1 base frame
    world_to_rby1_base = rby1_base_transform.inverse()
    
    # Transform left end-effector poses to RBY1 base frame coordinates
    left_ee_transforms_world = vtf.SE3.from_rotation_and_translation(
        vtf.SO3(wxyz=left_ee_ik_target_handle_wxyz),
        left_ee_ik_target_handle_position
    )
    left_ee_transforms_rby1_base = world_to_rby1_base @ left_ee_transforms_world
    left_ee_ik_target_handle_position = left_ee_transforms_rby1_base.wxyz_xyz[:, -3:]
    left_ee_ik_target_handle_wxyz = left_ee_transforms_rby1_base.wxyz_xyz[:, :4]
    
    # Transform right end-effector poses to RBY1 base frame coordinates
    right_ee_transforms_world = vtf.SE3.from_rotation_and_translation(
        vtf.SO3(wxyz=right_ee_ik_target_handle_wxyz),
        right_ee_ik_target_handle_position
    )
    right_ee_transforms_rby1_base = world_to_rby1_base @ right_ee_transforms_world
    right_ee_ik_target_handle_position = right_ee_transforms_rby1_base.wxyz_xyz[:, -3:]
    right_ee_ik_target_handle_wxyz = right_ee_transforms_rby1_base.wxyz_xyz[:, :4]

   

    head_tf = vtf.SE3.from_rotation_and_translation(
        vtf.SO3.from_rpy_radians(-head_data_all.rotation().as_rpy_radians().roll, 
        head_data_all.rotation().as_rpy_radians().pitch, 
        -head_data_all.rotation().as_rpy_radians().yaw),
        np.concatenate([head_data_all.wxyz_xyz[:, -3:-2], -head_data_all.wxyz_xyz[:, -2:-1], head_data_all.wxyz_xyz[:, -1:]], axis=1))
    
    head_offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(-np.pi/2, 0.0, 0.0), np.array([0.0, 0.0, 0.0]))
    # import pdb; pdb.set_trace()
    tf_head_target_world =  vtf.SE3(np.concatenate([head_tf.rotation().wxyz, head_tf.translation()], axis=1)) @ quest_to_zed_calib_tf @ head_offset # TODO: quest_to_zed_calib_tf np.load from file should eventually be replaced with metadata extraction
    head_transforms_rby1_base = world_to_rby1_base @ tf_head_target_world



    # Get gripper positions
    left_gripper_pos = joint_data['left-joint_pos'][..., None]
    right_gripper_pos = joint_data['right-joint_pos'][..., None]

    left_gripper_action = action_data['action-left-pos']
    right_gripper_action = action_data['action-right-pos']


    if cfg.move_old_handoffs and left_gripper_action.min() < 0.5 and right_gripper_action.min() < 0.5:
        print(f"Handoff detected, moving episode to old_handoffs folder{episode_path}")
        os.makedirs("old_handoffs", exist_ok=True)
        shutil.move(episode_path, os.path.join("old_handoffs", os.path.basename(episode_path)))

        return None, None


    # Check array lengths are consistent  
    assert len(left_gripper_pos) == len(right_gripper_pos)

    # Convert quaternions to 6d rotation representation
    # Convert from wxyz quaternions to rotation matrices, then to 6D representation
    left_rot_matrices = vtf.SO3(wxyz=left_ee_ik_target_handle_wxyz).as_matrix()  # Shape: (N, 3, 3)
    right_rot_matrices = vtf.SO3(wxyz=right_ee_ik_target_handle_wxyz).as_matrix()  # Shape: (N, 3, 3)
    
    # Convert rotation matrices to 6D representation using matrix_utils
    left_6d_rot = rot_mat_to_rot_6d(left_rot_matrices)  # Shape: (N, 6)
    right_6d_rot = rot_mat_to_rot_6d(right_rot_matrices)  # Shape: (N, 6)

    head_rot_matrices = vtf.SO3(wxyz=head_transforms_rby1_base.wxyz_xyz[:, :4]).as_matrix()  # Shape: (N, 3, 3)

    head_6d_rot = rot_mat_to_rot_6d(head_rot_matrices)  # Shape: (N, 6)
    head_position = head_transforms_rby1_base.wxyz_xyz[:, -3:]
    
    # Ensure all arrays have the same length
    seq_length = max(
        len(left_6d_rot), len(right_6d_rot),
        len(left_ee_ik_target_handle_position), len(right_ee_ik_target_handle_position),
        len(left_gripper_pos), len(right_gripper_pos), 
        len(head_6d_rot), len(head_position)
    )
    
    # Pad arrays to same length if needed
    def pad_to_length(arr, target_length):
        if len(arr) < target_length:
            last_val = arr[-1:] if len(arr) > 0 else np.zeros((1,) + arr.shape[1:])
            padding = np.repeat(last_val, target_length - len(arr), axis=0)
            return np.concatenate([arr, padding], axis=0)
        return arr[:target_length]
    
    left_6d_rot = pad_to_length(left_6d_rot, seq_length)
    right_6d_rot = pad_to_length(right_6d_rot, seq_length)
    head_6d_rot = pad_to_length(head_6d_rot, seq_length)
    head_position = pad_to_length(head_position, seq_length)
    left_ee_ik_target_handle_position = pad_to_length(left_ee_ik_target_handle_position, seq_length)
    right_ee_ik_target_handle_position = pad_to_length(right_ee_ik_target_handle_position, seq_length)
    left_gripper_pos = pad_to_length(left_gripper_pos, seq_length)
    right_gripper_pos = pad_to_length(right_gripper_pos, seq_length)

    left_gripper_action = pad_to_length(left_gripper_action, seq_length)
    right_gripper_action = pad_to_length(right_gripper_action, seq_length)


    # Validate jerky motion, logic inherited from https://github.com/xdofai/lab42/blob/d224643d2a96fbddada5c3029aeaba677ee5e76a/xdof/data_delivery/utils/validators.py#L409
    from viser.transforms import SE3
    for trajectory in [right_ee_transforms_rby1_base.as_matrix(), left_ee_transforms_rby1_base.as_matrix(), head_transforms_rby1_base.as_matrix()]:
        se3_poses = [SE3.from_matrix(pose) for pose in trajectory]
        se3_diffs = []
        for i in range(len(se3_poses) - 1):
            # Calculate relative transform: T2 * T1.inverse()
            relative_transform = se3_poses[i + 1].multiply(se3_poses[i].inverse())

            # Get translation and rotation differences
            translation_diff = np.linalg.norm(relative_transform.translation())
            rotation_diff = np.rad2deg(np.linalg.norm(relative_transform.rotation().log()))

            se3_diffs.append((translation_diff, rotation_diff))

        se3_diffs = np.array(se3_diffs)

        # Check for jerky motion in both translation and rotation
        translation_indices = np.where(se3_diffs[:, 0] > cfg.max_se3_diff_in_meters)[0]
        rotation_indices = np.where(se3_diffs[:, 1] > cfg.max_se3_diff_in_degrees)[0]
        # Check translation
        if len(translation_indices) > 0:
            print(f"jerky_cartesian_motion_detected (translation)")
            move_problematic_episode(episode_path, f"Jerky motion detected (translation)", cfg, episode_idx)
            return None, None
        if len(rotation_indices) > 0:
            print(f"jerky_cartesian_motion_detected (rotation)")
            move_problematic_episode(episode_path, f"Jerky motion detected (rotation)", cfg, episode_idx)
            return None, None
    
    # Combine into full end-effector state
    # FORMAT: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
    if len(left_gripper_pos.shape) == 1:
        left_gripper_pos = left_gripper_pos[..., None]
        right_gripper_pos = right_gripper_pos[..., None]
    elif len(left_gripper_pos.shape) == 3:
        left_gripper_pos = left_gripper_pos[:, :, 0]
        right_gripper_pos = right_gripper_pos[:, :, 0]
    if len(left_gripper_action.shape) == 1:
        left_gripper_action = left_gripper_action[..., None]
        right_gripper_action = right_gripper_action[..., None]
    elif len(left_gripper_action.shape) == 3:
        left_gripper_action = left_gripper_action[:, :, 0]
        right_gripper_action = right_gripper_action[:, :, 0]

    assert len(left_gripper_pos.shape) == 2
    assert len(right_gripper_pos.shape) == 2
    assert len(left_gripper_action.shape) == 2
    assert len(right_gripper_action.shape) == 2

    if cfg.gripper_action_delay_tsteps > 0:
        left_gripper_action = np.concatenate([
            np.ones((cfg.gripper_action_delay_tsteps, left_gripper_action.shape[1])) * left_gripper_action[0],
            left_gripper_action[:-cfg.gripper_action_delay_tsteps]
        ], axis=0)
        right_gripper_action = np.concatenate([
            np.ones((cfg.gripper_action_delay_tsteps, right_gripper_action.shape[1])) * right_gripper_action[0],
            right_gripper_action[:-cfg.gripper_action_delay_tsteps]
        ], axis=0)


    proprio_data = np.concatenate([
        left_6d_rot, left_ee_ik_target_handle_position, left_gripper_pos,
        right_6d_rot, right_ee_ik_target_handle_position, right_gripper_pos
    ], axis=1)

    state_dim = 20
    action_dim = 20

    if cfg.include_head_pose:
        proprio_data = np.concatenate([
            proprio_data,
            head_6d_rot, head_position
        ], axis=1)
        state_dim = 29

    action_data = np.concatenate([
        left_6d_rot, left_ee_ik_target_handle_position, left_gripper_action,
        right_6d_rot, right_ee_ik_target_handle_position, right_gripper_action
    ], axis=1)

    if cfg.include_head_pose:
        action_data = np.concatenate([
            action_data,
            head_6d_rot, head_position
        ], axis=1)
        action_dim = 29
    # We need seq_length - 1 steps since we calculate actions as deltas
    seq_length = seq_length - 1
    
    if seq_length <= 0:
        return None, None

    # Calculate actions as deltas
    states = []
    actions = []

    
    for step in range(seq_length):
        # Current state
        state_t = proprio_data[step].copy()

        if cfg.delta_proprio_keys is not None:
            # Make the proprioceptive state for this axis delta instead of absolute
            if "z" in cfg.delta_proprio_keys:
                if step > 0:
                    state_t[8] = proprio_data[step][8] - proprio_data[step-1][8]
                    state_t[18] = proprio_data[step][18] - proprio_data[step-1][18]
                else:
                    state_t[8] = 0.0
                    state_t[18] = 0.0

        action_t = action_data[step]
        
        states.append(state_t)
        actions.append(action_t)
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    if cfg.perturb_z_height:
        random_z_height = np.random.uniform(cfg.perturb_z_height_range[0], cfg.perturb_z_height_range[1])
        left_hand_z_t0 = states[0, 8]
        right_hand_z_t0 = states[0, 18]
        avg_z_t0 = (left_hand_z_t0 + right_hand_z_t0) / 2
        diff_z_t0 = random_z_height - avg_z_t0
        states[:, 8] += diff_z_t0
        states[:, 18] += diff_z_t0
        states[:, -1] += diff_z_t0

        actions[:, 8] += diff_z_t0
        actions[:, 18] += diff_z_t0
        actions[:, -1] += diff_z_t0

    if cfg.perturb_xy_position:
        random_x_position = np.random.uniform(cfg.perturb_x_position_range[0], cfg.perturb_x_position_range[1])
        random_y_position = np.random.uniform(cfg.perturb_y_position_range[0], cfg.perturb_y_position_range[1])
        random_xy_position = np.array([random_x_position, random_y_position])
        states[:, 6:8] += random_xy_position
        states[:, 16:18] += random_xy_position
        states[:, 26:28] += random_xy_position

        actions[:, 6:8] += random_xy_position
        actions[:, 16:18] += random_xy_position
        actions[:, 26:28] += random_xy_position
    

    
    # VALIDATION: Check final processed arrays
    is_valid, error_msg = validate_array_data(states, "states", expected_shape=(seq_length, state_dim))
    if not is_valid:
        print(f"❌ States validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"States validation failed: {error_msg}", cfg, episode_idx)
        return None, None
    
    is_valid, error_msg = validate_array_data(actions, "actions", expected_shape=(seq_length, action_dim))
    if not is_valid:
        print(f"❌ Actions validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Actions validation failed: {error_msg}", cfg, episode_idx)
        return None, None

    
    return states, actions

def keyframes_precompute(actions: np.ndarray, cfg: XMIConfig):
    zed_factory_intrinsics = np.array([ # Currently hardcoded TODO: move to metadata (both raw data and LeRobot data)
        [532.395, 0, 638.22],
        [0, 532.325, 363.7015],
        [0, 0, 1]
    ])

    capture_resolution = (1280, 720) # Currently hardcoded TODO: move to metadata (both raw data and LeRobot data)
    new_resolution = (224, 224) # Currently hardcoded TODO: move to metadata (both raw data and LeRobot data)

    zed_post_crop_intrinsics = zed_tf_intrinsics(zed_factory_intrinsics, capture_resolution=capture_resolution, new_resolution=new_resolution)
    top_camera_fov = 2 * np.arctan(new_resolution[0] / (2 * zed_post_crop_intrinsics[0, 0]))    
    
    keyframe_idxs_for_this_traj = []
    past_idxs = deque(maxlen=cfg.keyframe_max_len)
    if actions is None:
        return []
    for traj_idx in range(actions.shape[0]):
        past_head_traj = actions[0:traj_idx+1][:, 20:29] # Causal 9D head trajectory (index -1 is current head pose index 0 is t0)
        
        past_idxs = select_keyframes_helper(past_head_traj, top_camera_fov, past_idxs)

        if len(past_idxs) < past_idxs.maxlen:
            while len(past_idxs) < past_idxs.maxlen:
                past_idxs.append(past_idxs[-1])

        keyframe_idxs_for_this_traj.append(deepcopy(list(past_idxs)))

    # print(f"Keyframe idxs for this traj: {keyframe_idxs_for_this_traj}")
    # import pdb; pdb.set_trace()
    return keyframe_idxs_for_this_traj

def process_episode_in_chunks(episode_data: dict, cfg: XMIConfig, max_chunk_frames: int = 1000, episode_path: Path = None, episode_idx: int = None) -> list:
    """Process episode data in memory-efficient chunks to handle long episodes."""
    
    # Process XMI transforms first
    states, actions = process_xmi_transforms(episode_data, cfg, episode_path, episode_idx)

    keyframe_idxs_for_this_traj = keyframes_precompute(actions, cfg)

    
    if states is None or actions is None:
        return []
    
    original_total_length = len(states)
    if original_total_length <= 0:
        return []
    
    # Calculate global subsampling indices for consistency
    if cfg.temporal_subsample_factor > 1:
        global_subsample_indices = list(range(0, original_total_length, cfg.temporal_subsample_factor))
        states = states[global_subsample_indices]
        actions = actions[global_subsample_indices]
    else:
        global_subsample_indices = list(range(original_total_length))
    
    total_length = len(states)
    all_records = []
    
    # Process in chunks to avoid OOM
    for chunk_start in range(0, total_length, max_chunk_frames):
        chunk_end = min(chunk_start + max_chunk_frames, total_length)
        chunk_length = chunk_end - chunk_start
        
        # Process joint data for this chunk
        chunk_states = states[chunk_start:chunk_end]
        chunk_actions = actions[chunk_start:chunk_end]
        
        # Create records for this chunk
        for step in range(chunk_length):
            global_step = chunk_start + step
            state = chunk_states[step]
            action = chunk_actions[step]
            keyframe_idx = keyframe_idxs_for_this_traj[step]

            record = {
                "state": state.tolist(),
                "actions": action.tolist(),
                "keyframe_idx": keyframe_idx,
                "timestamp": [global_step / (cfg.fps / cfg.temporal_subsample_factor)],
                "frame_index": [global_step],
                "episode_index": [0],  # Will be updated later
                "index": [global_step],
                "task_index": [0],  # Will be updated later
            }
            all_records.append(record)
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # VALIDATION: Check final records before returning
    is_valid, error_msg = validate_records(all_records)
    if not is_valid:
        print(f"❌ Records validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Records validation failed: {error_msg}", cfg, episode_idx)
        return []
    
    return all_records


def compute_basic_episode_stats(episode_idx: int, episode_info: dict, cfg: XMIConfig, base_dir: Path) -> dict:
    """Compute basic statistics for an episode to create v2.1 compatible episodes_stats.jsonl"""
    
    # Load the episode parquet file
    chunk_id = episode_idx // cfg.chunk_size
    parquet_path = base_dir / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_idx:06d}.parquet"

    state_dim = 20  # XMI uses 20-dimensional state/action space
    action_dim = 20

    if cfg.include_head_pose:
        state_dim += 9
        action_dim += 9
    
    if not parquet_path.exists():
        # Return minimal stats if parquet doesn't exist
        return {
            "state": {
                "min": np.zeros(state_dim, dtype=np.float32),
                "max": np.zeros(state_dim, dtype=np.float32), 
                "mean": np.zeros(state_dim, dtype=np.float32),
                "std": np.ones(state_dim, dtype=np.float32),
                "count": np.array([1], dtype=np.int64)
            },
            "actions": {
                "min": np.zeros(action_dim, dtype=np.float32),
                "max": np.zeros(action_dim, dtype=np.float32),
                "mean": np.zeros(action_dim, dtype=np.float32),
                "std": np.ones(action_dim, dtype=np.float32),
                "count": np.array([1], dtype=np.int64)
            },
        }
    
    # Load episode data
    df = pd.read_parquet(parquet_path)
    episode_length = len(df)
    
    episode_stats = {}
    
    # Compute stats for state and actions (vector features)
    for feature_name in ["state", "actions"]:
        if feature_name in df.columns:
            # Convert list columns to numpy arrays
            data = np.array(df[feature_name].tolist(), dtype=np.float32)
            
            episode_stats[feature_name] = {
                "min": data.min(axis=0),
                "max": data.max(axis=0), 
                "mean": data.mean(axis=0),
                "std": data.std(axis=0),
                "count": np.array([episode_length], dtype=np.int64),
            }
    
    # Add stats for scalar features with proper keepdims handling
    for feature_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if feature_name in df.columns:
            # Handle potential list columns by converting to numpy array first
            # Use pandas Series to avoid numpy deprecation warnings
            raw_series = df[feature_name]
            
            # Safely check if this is a list column
            try:
                first_item = raw_series.iloc[0]
                if isinstance(first_item, list):
                    # If it's a list column, flatten it properly
                    data = np.array([item for sublist in raw_series for item in sublist], dtype=np.float32)
                else:
                    # If it's already flat, convert using pandas to_numpy() which avoids deprecation warnings
                    data = raw_series.to_numpy(dtype=np.float32)
            except (IndexError, AttributeError):
                # Fallback: try to convert using pandas to_numpy()
                data = raw_series.to_numpy(dtype=np.float32)
            
            if len(data.shape) > 1:
                data = data.flatten()
            
            # For 1D data, LeRobot expects keepdims=True if original was 1D
            episode_stats[feature_name] = {
                "min": np.array([data.min().item()], dtype=np.float32),
                "max": np.array([data.max().item()], dtype=np.float32),
                "mean": np.array([data.mean().item()], dtype=np.float32),
                "std": np.array([data.std().item()], dtype=np.float32),
                "count": np.array([episode_length], dtype=np.int64),
            }
    
    
    # Add video stats if not skipping videos (normalized to [0,1] range)
    if not cfg.skip_videos:
        for cam_key in cfg.camera_keys:
            # For images/videos, LeRobot expects shape (C, H, W) stats normalized to [0,1]
            # We provide reasonable defaults for RGB images
            episode_stats[cam_key] = {
                "min": np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1, 1),
                "max": np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape(3, 1, 1),
                "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1),
                "std": np.array([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1),
                "count": np.array([episode_length], dtype=np.int64),
            }
    
    return episode_stats


def write_episode_metadata_immediately(episode_data: dict, tasks: list[str], base_dir: Path):
    """Write episode and task metadata immediately after processing each episode."""
    
    # Write episode metadata
    episodes_file = base_dir / "meta" / "episodes.jsonl"
    with open(episodes_file, "a") as f:
        f.write(json.dumps(episode_data) + "\n")
    
    # Load existing tasks to avoid duplicates
    existing_tasks = {}
    tasks_file = base_dir / "meta" / "tasks.jsonl"
    
    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            for line in f:
                task_data = json.loads(line.strip())
                existing_tasks[task_data['task']] = task_data['task_index']
    
    # Add new tasks if they don't exist
    for task in tasks:
        if task not in existing_tasks:
            task_index = len(existing_tasks)
            existing_tasks[task] = task_index
            
            # Append new task immediately
            with open(tasks_file, "a") as f:
                f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")
    
    return existing_tasks


def process_xmi_episode(
    idx: int, episode_path: Path, language_instruction: str, cfg: XMIConfig, episode_base: Path,
    base_dir: Path
):
    """Process a single XMI episode and save it directly to LeRobot format."""
    
    # print(f"Processing episode {idx}: {episode_path.name}")
    
    # Load episode data
    # Quality filtering
    if not cfg.no_filter_quality and not is_episode_good_quality(episode_path):
        print(f"  Skipping episode {idx}: poor quality")
        return None
    episode_data = load_episode_data(episode_path, cfg, base_dir, idx)
    if not episode_data:
        print(f"  ❌ Failed to load episode {idx}")
        move_problematic_episode(episode_path, "Failed to load episode data", cfg, idx)
        return None
    
    # Validate video timestamp synchronization
    is_valid, error_msg = validate_video_timestamp_synchronization(episode_path, cfg, idx, cfg.video_timestamp_tolerance_s)
    if not is_valid:
        print(f"  ❌ Video timestamp validation failed for episode {idx}: {error_msg}")
        move_problematic_episode(episode_path, f"Video timestamp synchronization failed: {error_msg}", cfg, idx)
        return None
    
    # Process episode in memory-efficient chunks
    # try:
    records = process_episode_in_chunks(episode_data, cfg, max_chunk_frames=cfg.max_frames_per_chunk, episode_path=episode_path, episode_idx=idx)
    if not records:
        print(f"  ❌ No valid data in episode {idx}")
        # Episode already moved by process_episode_in_chunks if it was a validation failure
        return None
    
    seq_length = len(records)
    # print(f"  Episode {idx}: {seq_length} frames total")
                
    # except Exception as e:
    #     error_msg = f"Error processing episode {idx}: {e}"
    #     print(f"  ❌ {error_msg}")
    #     move_problematic_episode(episode_path, error_msg, cfg, idx)
    #     return None
    
    # Update episode and task indices in records
    for record in records:
        record["episode_index"] = [idx]
        record["index"] = [record["frame_index"][0]]  # Global frame index will be updated later
    
    # VALIDATION: Final check before saving
    is_valid, error_msg = validate_records(records)
    if not is_valid:
        print(f"  ❌ Final validation failed for episode {idx}: {error_msg}")
        move_problematic_episode(episode_path, f"Final validation failed: {error_msg}", cfg, idx)
        return None
    
    # Save parquet (joint positions + actions per frame)
    episode_path_out = episode_base / f"episode_{idx:06d}.parquet"
    episode_path_out.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        pd.DataFrame(records).to_parquet(episode_path_out)
    except Exception as e:
        print(f"  ❌ Failed to save parquet for episode {idx}: {e}")
        move_problematic_episode(episode_path, f"Failed to save parquet for episode {idx}: {e}", cfg, idx)
        return None
    
    # Save videos if not skipping
    # if not cfg.skip_videos and image_data:
    #     chunk_id = idx // cfg.chunk_size
    #     for cam_key in cfg.camera_keys:
    #         if cam_key in image_data and cam_key in cfg.camera_key_mapping:
    #             video_dir = base_dir / "videos" / f"chunk-{chunk_id:03d}" / cam_key
    #             video_dir.mkdir(parents=True, exist_ok=True)
    #             save_path = video_dir / f"episode_{idx:06d}.mp4"
                
    #             frames = image_data[cam_key]
    #             if frames:
    #                 # VALIDATION: Final check on video frames
    #                 is_valid, error_msg = validate_images(frames, cam_key)
    #                 if not is_valid:
    #                     print(f"  ⚠️  Video validation failed for {cam_key} in episode {idx}: {error_msg}")
    #                     move_problematic_episode(episode_path, f"Video validation failed for {cam_key} in episode {idx}: {error_msg}", cfg, idx)
    #                     continue
                    
    #                 # print(f"  Encoding video {cam_key}: {len(frames)} frames")
    #                 try:
    #                     if encoder_name:
    #                         encode_video_optimized(frames, save_path, cfg.fps, cfg.temporal_subsample_factor, encoder_name, encoding_quality)
    #                     else:
    #                         encode_video_simple(frames, save_path, cfg.fps, cfg.temporal_subsample_factor)
    #                 except Exception as e:
    #                     print(f"  ⚠️  Video encoding failed for {cam_key} in episode {idx}: {e}")
    #                     move_problematic_episode(episode_path, f"Video encoding failed for {cam_key} in episode {idx}: {e}", cfg, idx)
    
    # Compute and write episode stats immediately
    episode_stats = compute_basic_episode_stats(idx, {"length": seq_length}, cfg, base_dir)
    if HAS_LEROBOT:
        write_episode_stats(idx, episode_stats, base_dir)
    
    # Write episode metadata immediately
    episode_metadata = {
        "episode_index": idx,
        "tasks": [language_instruction],
        "length": seq_length,
        "original_episode_name": episode_path.name,  # Add original episode directory name
        "original_episode_path": str(episode_path),  # Add full original path for reference
    }
    
    task_mapping = write_episode_metadata_immediately(episode_metadata, [language_instruction], base_dir)
    
    # Update task index in the episode metadata
    task_index = task_mapping.get(language_instruction, 0)
    episode_metadata["task_index"] = task_index
    
    # Clean up memory
    del episode_data, records
    gc.collect()
    
    # print(f"  ✅ Completed episode {idx}: {seq_length} frames, task '{language_instruction}'")
    
    # Return metadata for final statistics
    return episode_metadata


def renumber_episodes_consecutively(all_episodes: list, base_dir: Path, cfg: XMIConfig) -> list:
    """
    Renumber episodes consecutively to fill gaps from skipped episodes.
    
    Args:
        all_episodes: List of successful episode metadata
        base_dir: Base directory of the dataset
        cfg: Configuration object
        
    Returns:
        list: Updated episode metadata with consecutive indices
    """
    print(f"\n=== Renumbering Episodes Consecutively ===")
    
    # Sort episodes by original index to maintain order
    all_episodes.sort(key=lambda x: x['episode_index'])
    
    # Create mapping from old to new indices
    old_to_new_mapping = {}
    updated_episodes = []
    
    for new_idx, episode_metadata in enumerate(all_episodes):
        old_idx = episode_metadata['episode_index']
        old_to_new_mapping[old_idx] = new_idx
        
        # Update the episode metadata
        episode_metadata['episode_index'] = new_idx
        updated_episodes.append(episode_metadata)
        
        print(f"  Episode {old_idx} → {new_idx}")
    
    # Now we need to rename all the files
    print(f"Renaming files for {len(updated_episodes)} episodes...")
    
    # Rename parquet files
    for episode_metadata in updated_episodes:
        old_idx = None
        new_idx = episode_metadata['episode_index']
        
        # Find the original index from the mapping
        for old, new in old_to_new_mapping.items():
            if new == new_idx:
                old_idx = old
                break
        
        if old_idx == new_idx:
            continue  # No change needed
        
        # Rename parquet file
        old_chunk_id = old_idx // cfg.chunk_size
        new_chunk_id = new_idx // cfg.chunk_size
        
        old_parquet_path = base_dir / "data" / f"chunk-{old_chunk_id:03d}" / f"episode_{old_idx:06d}.parquet"
        new_parquet_path = base_dir / "data" / f"chunk-{new_chunk_id:03d}" / f"episode_{new_idx:06d}.parquet"
        
        if old_parquet_path.exists():
            # Create new chunk directory if needed
            new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_parquet_path), str(new_parquet_path))
        
        # Rename video files
        if not cfg.skip_videos:
            for cam_key in cfg.camera_keys:
                old_video_path = base_dir / "videos" / f"chunk-{old_chunk_id:03d}" / cam_key / f"episode_{old_idx:06d}.mp4"
                new_video_path = base_dir / "videos" / f"chunk-{new_chunk_id:03d}" / cam_key / f"episode_{new_idx:06d}.mp4"
                
                if old_video_path.exists():
                    # Create new video directory if needed
                    new_video_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_video_path), str(new_video_path))
    
    # Update parquet files with new episode indices
    print("Updating parquet files with new episode indices...")
    for episode_metadata in updated_episodes:
        new_idx = episode_metadata['episode_index']
        new_chunk_id = new_idx // cfg.chunk_size
        parquet_path = base_dir / "data" / f"chunk-{new_chunk_id:03d}" / f"episode_{new_idx:06d}.parquet"
        
        if parquet_path.exists():
            # Load, update, and save the parquet file
            df = pd.read_parquet(parquet_path)
            # Preserve list format for scalar fields to maintain shape consistency
            df['episode_index'] = [[new_idx] for _ in range(len(df))]
            # Calculate global frame indices while preserving list format
            global_indices = []
            for _, row in df.iterrows():
                frame_idx = row['frame_index'][0] if isinstance(row['frame_index'], list) else row['frame_index']
                global_indices.append([frame_idx + new_idx * 10000])
            df['index'] = global_indices
            df.to_parquet(parquet_path)
    
    # Clean up empty chunk directories
    for chunk_dir in (base_dir / "data").iterdir():
        if chunk_dir.is_dir() and not any(chunk_dir.iterdir()):
            shutil.rmtree(chunk_dir)
    
    if not cfg.skip_videos:
        for chunk_dir in (base_dir / "videos").iterdir():
            if chunk_dir.is_dir():
                # Check if any camera subdirectories have files
                has_files = False
                for cam_dir in chunk_dir.iterdir():
                    if cam_dir.is_dir() and any(cam_dir.iterdir()):
                        has_files = True
                        break
                if not has_files:
                    shutil.rmtree(chunk_dir)
    
    print(f"✅ Successfully renumbered {len(updated_episodes)} episodes")
    return updated_episodes


def rewrite_episode_metadata(all_episodes: list, base_dir: Path):
    """
    Rewrite episode and task metadata files with updated indices.
    
    Args:
        all_episodes: List of episode metadata with updated indices
        base_dir: Base directory of the dataset
    """
    print("Rewriting metadata files...")
    
    # Clear existing metadata files
    episodes_file = base_dir / "meta" / "episodes.jsonl"
    tasks_file = base_dir / "meta" / "tasks.jsonl"
    
    if episodes_file.exists():
        episodes_file.unlink()
    if tasks_file.exists():
        tasks_file.unlink()
    
    # Collect all unique tasks
    all_tasks = set()
    for episode in all_episodes:
        all_tasks.update(episode['tasks'])
    
    # Write tasks file
    task_mapping = {}
    with open(tasks_file, 'w') as f:
        for task_index, task in enumerate(sorted(all_tasks)):
            task_mapping[task] = task_index
            f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")
    
    # Write episodes file with updated task indices
    with open(episodes_file, 'w') as f:
        for episode in all_episodes:
            # Update task_index to match the new mapping
            episode['task_index'] = task_mapping.get(episode['tasks'][0], 0)
            f.write(json.dumps(episode) + "\n")


def rewrite_episodes_stats(all_episodes: list, base_dir: Path, cfg: XMIConfig):
    """
    Rewrite episode stats files with updated indices.
    
    Args:
        all_episodes: List of episode metadata with updated indices
        base_dir: Base directory of the dataset
        cfg: Configuration object
    """
    if not HAS_LEROBOT:
        return
    
    print("Rewriting episode stats...")
    
    # Clear existing stats files
    stats_dir = base_dir / "meta" / "stats"
    if stats_dir.exists():
        shutil.rmtree(stats_dir)
    
    # Recreate stats for each episode with new indices
    for episode in all_episodes:
        new_idx = episode['episode_index']
        episode_stats = compute_basic_episode_stats(new_idx, episode, cfg, base_dir)
        write_episode_stats(new_idx, episode_stats, base_dir)


def write_episode_name_mapping(all_episodes: list, base_dir: Path):
    """
    Write a JSON file that maps original episode names to LeRobot episode indices.
    
    Args:
        all_episodes: List of episode metadata with original names and new indices
        base_dir: Base directory of the dataset
    """
    print("Writing episode name mapping...")
    
    # Create mapping dictionary
    episode_mapping = {}
    
    for episode in all_episodes:
        lerobot_episode_name = f"episode_{episode['episode_index']:06d}"
        original_name = episode.get('original_episode_name', 'unknown')
        original_path = episode.get('original_episode_path', 'unknown')
        
        episode_mapping[original_name] = {
            "lerobot_episode_index": episode['episode_index'],
            "lerobot_episode_name": lerobot_episode_name,
            "original_episode_path": original_path,
            "task": episode['tasks'][0] if episode['tasks'] else 'unknown',
            "length": episode['length']
        }
    
    # Write to JSON file
    mapping_file = base_dir / "meta" / "episode_name_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(episode_mapping, f, indent=2, sort_keys=True)
    
    print(f"✅ Episode name mapping written to: {mapping_file}")
    print(f"   Mapped {len(episode_mapping)} episodes")


def main(cfg: XMIConfig):
    """Main function to convert XMI data to LeRobot format."""
    
    print("=== Fast XMI to LeRobot Converter ===")
    print(cfg)
    
    # Handle push-to-hub-only mode
    if cfg.push_to_hub_only:
        print("🚀 Push-to-Hub-Only Mode")
        base_dir = HF_LEROBOT_HOME / cfg.repo_name
        
        if not base_dir.exists():
            print(f"❌ Dataset directory does not exist: {base_dir}")
            print("Cannot push non-existent dataset to hub.")
            return
        
        # Verify dataset structure exists
        required_files = [
            base_dir / "meta" / "info.json",
            base_dir / "meta" / "episodes.jsonl", 
            base_dir / "meta" / "tasks.jsonl"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("Cannot push incomplete dataset to hub.")
            return
        
        if not HAS_LEROBOT:
            print("❌ Cannot push to hub: LeRobot not available")
            print("Install lerobot package to enable hub push functionality")
            return
        
        # Load info.json to get dataset statistics
        with open(base_dir / "meta" / "info.json", 'r') as f:
            info = json.load(f)
        
        print(f"📊 Dataset Info:")
        print(f"  Repository: {cfg.repo_name}")
        print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
        print(f"  Total frames: {info.get('total_frames', 'unknown')}")
        print(f"  Dataset path: {base_dir}")
        
        # Perform hub push
        try:
            dataset = LeRobotDataset(repo_id=cfg.repo_name, root=base_dir)
            print(f"✅ LeRobotDataset loaded with {len(dataset)} frames")
            
            print(f"🚀 Pushing dataset to Hugging Face Hub: {cfg.repo_name}")
            dataset.push_to_hub(
                tags=["xmi", "rby", "xdof", "bimanual"],
                private=True,
                push_videos=not cfg.skip_videos,
                license="apache-2.0",
            )
            print(f"✅ Dataset successfully pushed to hub: {cfg.repo_name}")
            print(f"🔗 View at: https://huggingface.co/datasets/{cfg.repo_name}")
            
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")
        
        return  # Exit early since we're only pushing to hub
    
    # Normal processing mode
    print("🔄 Dataset Processing Mode")
    print(f"Input paths: {cfg.raw_dataset_folders}")
    print(f"Output path: {HF_LEROBOT_HOME/cfg.repo_name}")
    print(f"Repository name: {cfg.repo_name}")
    print(f"Skip videos: {cfg.skip_videos}")
    print(f"Max episodes: {cfg.max_episodes or 'unlimited'}")
    print(f"Max workers: {cfg.max_workers}")
    
    # Find episodes
    episode_dirs = find_episode_directories(cfg.raw_dataset_folders)
    if cfg.max_episodes:
        episode_dirs = episode_dirs[:cfg.max_episodes]
    
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if not episode_dirs:
        print("No episodes found!")
        return
    
    # Prepare folders
    base_dir = HF_LEROBOT_HOME / cfg.repo_name
    
    # Clean up any existing dataset in the output directory
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Create directories
    (base_dir / "data").mkdir(parents=True, exist_ok=True)
    (base_dir / "meta").mkdir(exist_ok=True)
    if not cfg.skip_videos:
        (base_dir / "videos").mkdir(exist_ok=True)
    
    # Create chunk directories (generous allocation for original indices)
    num_chunks = (len(episode_dirs) + cfg.chunk_size - 1) // cfg.chunk_size
    episode_base = base_dir / "data"
    for i in range(num_chunks):
        (episode_base / f"chunk-{i:03d}").mkdir(parents=True, exist_ok=True)
    
    # Process episodes
    all_episodes = []
    
    # Ensure we have the same number of language instructions as dataset folders
    language_instructions = cfg.language_instructions
    if len(language_instructions) < len(cfg.raw_dataset_folders):
        # Extend with the last instruction if we don't have enough
        language_instructions.extend([language_instructions[-1]] * (len(cfg.raw_dataset_folders) - len(language_instructions)))
    
    # Map each episode to its corresponding language instruction based on source folder
    episode_language_map = {}
    for i, folder in enumerate(cfg.raw_dataset_folders):
        folder_path = Path(folder)
        language_instruction = language_instructions[i] if i < len(language_instructions) else language_instructions[0]
        for episode_dir in episode_dirs:
            if str(episode_dir).startswith(str(folder_path)):
                episode_language_map[episode_dir] = language_instruction
    
    if cfg.max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = []
            for i, episode_path in enumerate(episode_dirs):
                chunk_id = i // cfg.chunk_size
                language_instruction = episode_language_map.get(episode_path, cfg.language_instructions[0])
                futures.append(
                    executor.submit(
                        process_xmi_episode,
                        i,
                        episode_path,
                        language_instruction,
                        cfg,
                        episode_base / f"chunk-{chunk_id:03d}",
                        base_dir
                    )
                )
            
            for f in tqdm(futures, desc="Processing episodes"):
                result = f.result()
                if result is not None:
                    all_episodes.append(result)
    else:
        # Sequential processing (for debugging)
        for i, episode_path in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
            chunk_id = i // cfg.chunk_size
            language_instruction = episode_language_map.get(episode_path, cfg.language_instructions[0])
            result = process_xmi_episode(
                i,
                episode_path, 
                language_instruction,
                cfg,
                episode_base / f"chunk-{chunk_id:03d}",
                base_dir
            )
            if result is not None:
                all_episodes.append(result)
    
    print(f"Successfully processed {len(all_episodes)} episodes")
    
    if not all_episodes:
        print("No episodes were processed!")
        return
    
    # Renumber episodes consecutively to fill gaps
    all_episodes = renumber_episodes_consecutively(all_episodes, base_dir, cfg)
    
    # Rewrite metadata files with updated indices
    rewrite_episode_metadata(all_episodes, base_dir)
    
    # Rewrite episode stats with updated indices
    rewrite_episodes_stats(all_episodes, base_dir, cfg)
    
    # Write episode name mapping
    write_episode_name_mapping(all_episodes, base_dir)
    
    # Calculate final dataset statistics
    total_frames = sum(e["length"] for e in all_episodes)
    actual_chunks = (len(all_episodes) + cfg.chunk_size - 1) // cfg.chunk_size
    
    # Get unique tasks
    all_tasks = {}
    tasks_file = base_dir / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            for line in f:
                task_data = json.loads(line.strip())
                all_tasks[task_data['task_index']] = task_data['task']

    state_dim = 20
    action_dim = 20

    if cfg.include_head_pose:
        state_dim = 29
        action_dim = 29
    
    # Write info.json
    features = {
        "state": {
            "dtype": "float32", 
            "shape": [state_dim],  # XMI uses 20-dimensional state space. If include_head_pose, then 29.
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": [action_dim],  # XMI uses 20-dimensional action space. If include_head_pose, then 29.
            "names": ["actions"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    
    # Add camera features if not skipping videos
    if not cfg.skip_videos:
        for cam_key in cfg.camera_keys:
            if cam_key in cfg.camera_key_mapping:
                lerobot_key = cfg.camera_key_mapping[cam_key]
                features[lerobot_key] = {
                    "dtype": "video",
                    "shape": [cfg.resize_size, cfg.resize_size, 3],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.fps": cfg.fps / cfg.temporal_subsample_factor,
                        "video.height": cfg.resize_size,
                        "video.width": cfg.resize_size,
                        "video.channels": 3,
                        "video.codec": "libx264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                }
    
    info = {
        "codebase_version": "v2.1",
        "robot_type": "xmi",
        "total_episodes": len(all_episodes),
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(cfg.camera_keys) * len(all_episodes) if not cfg.skip_videos else 0,
        "total_chunks": actual_chunks,
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps / cfg.temporal_subsample_factor,
        "splits": {"train": f"0:{len(all_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features
    }
    
    with open(base_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Dataset saved to: {HF_LEROBOT_HOME/cfg.repo_name}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total frames: {total_frames}")
    print(f"Total chunks: {actual_chunks}")
    
    # Print summary of problematic episodes
    print_problematic_episodes_summary(cfg)
    
    # Push to hub if enabled
    if cfg.push_to_hub and HAS_LEROBOT:
        print(f"\nPreparing to push dataset to Hugging Face Hub...")
        
        try:
            from huggingface_hub import HfApi, whoami
            
            # Check authentication
            user_info = whoami()
            print(f"✅ Authenticated as: {user_info['name']}")
            
            # Create repository
            api = HfApi()
            print(f"🏗️  Ensuring repository exists: {cfg.repo_name}")
            repo_url = api.create_repo(
                repo_id=cfg.repo_name,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )
            print(f"✅ Repository ready: {repo_url}")
            
            # Create version tag
            try:
                api.create_tag(
                    repo_id=cfg.repo_name,
                    tag="v2.1",
                    repo_type="dataset"
                )
                print(f"✅ Version tag created: v2.1")
            except Exception as tag_error:
                print(f"⚠️  Version tag creation failed (may already exist): {tag_error}")
            
            # Instantiate LeRobotDataset and push
            dataset = LeRobotDataset(repo_id=cfg.repo_name, root=base_dir)
            print(f"✅ LeRobotDataset loaded with {len(dataset)} frames")
            
            print(f"🚀 Pushing dataset to Hugging Face Hub: {cfg.repo_name}")
            dataset.push_to_hub(
                tags=["xmi", "rby", "xdof", "bimanual"],
                private=True,
                push_videos=not cfg.skip_videos,
                license="apache-2.0",
            )
            print(f"✅ Dataset successfully pushed to hub: {cfg.repo_name}")
            print(f"🔗 View at: https://huggingface.co/datasets/{cfg.repo_name}")
            
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")
            print("Dataset was created successfully locally, but hub push failed.")
    elif cfg.push_to_hub and not HAS_LEROBOT:
        print("❌ Cannot push to hub: LeRobot not available")
        print("Install lerobot package to enable hub push functionality")


def print_problematic_episodes_summary(cfg: XMIConfig):
    """Print a summary of problematic episodes that were moved."""
    problematic_dir = Path(cfg.problematic_data_dir)
    
    if not problematic_dir.exists():
        return
    
    print(f"\n=== Problematic Episodes Summary ===")
    print(f"Problematic data directory: {problematic_dir}")
    
    total_problematic = 0
    for error_subdir in problematic_dir.iterdir():
        if error_subdir.is_dir():
            episode_count = len([d for d in error_subdir.iterdir() if d.is_dir()])
            if episode_count > 0:
                print(f"  {error_subdir.name}: {episode_count} episodes")
                total_problematic += episode_count
    
    print(f"Total problematic episodes: {total_problematic}")
    
    if total_problematic > 0:
        print(f"💡 You can investigate these episodes manually in: {problematic_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(XMIConfig)
    main(cfg)
