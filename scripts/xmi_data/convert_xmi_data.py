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
from typing import List, Optional
import numpy as np
import tyro
import gc
import shutil
import viser.transforms as vtf
from openpi.utils.xmi_dataloader_utils import load_episode_data, validate_episode_data, validate_array_data, validate_records, validate_images
from openpi.utils.matrix_utils import *

# Set environment variable for dataset storage

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import write_episode_stats
    HAS_LEROBOT = True
except ImportError:
    print("Warning: LeRobot not available. Hub push functionality disabled.")
    HAS_LEROBOT = False


@dataclass
class XMIConfig:
    # Input data paths
    raw_dataset_folders: List[str] = field(default_factory=lambda: [
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
        # "/home/justinyu/Downloads/20250630",
        # "/home/justinyu/Downloads/data_20250708",
        "/nfs_us/data/us_xmi_01/20250714",
        "/nfs_us/data/us_xmi_01/20250715",
        "/nfs_us/data/oreo_xmi/clean_whiteboard",
        "/nfs_us/data/oreo_xmi/fold_napkin_place_utensils_inside_and_roll_it_up",
        "/nfs_us/data/oreo_xmi/folding_skirt_pile_and_stacking",
        "/nfs_us/data/oreo_xmi/folding_trousers_pile_and_stacking",
        "/nfs_us/data/oreo_xmi/folding_tshirt_pile_and_stacking",
        "/nfs_us/data/oreo_xmi/insert_the_plug",
        "/nfs_us/data/oreo_xmi/packing_luggage",
        "/nfs_us/data/oreo_xmi/painting_nails",
        "/nfs_us/data/oreo_xmi/place_fake_bread",
        "/nfs_us/data/oreo_xmi/place_trousers_on_hanger",
        "/nfs_us/data/oreo_xmi/place_tshirt_on_hanger",
        "/nfs_us/data/oreo_xmi/put_pillow_into_pillowcase",
        "/nfs_us/data/oreo_xmi/serve_a_lunch_box",
        "/nfs_us/data/oreo_xmi/sorting_stationery_into_containers",
        "/nfs_us/data/oreo_xmi/untangling_cables",
        "/nfs_us/data/oreo_xmi/zip_up_a_jacket",
    ])
    
    # Language instructions corresponding to each dataset folder (ANNOTATION OVERRIDES)
    language_instructions: List[str] = field(default_factory=lambda: [
        # "pick up the beverage and place it in the bin",
        # "pick up the chips bag and place it in the bin",
        # "pick up the candy bag and place it in the bin",
        # "pick up the cleaning sponge and place it in the bin",
        # "pick up the dish detergent and place it in the bin",
        # "pick up the paper towel and place it in the bin",
        # "pick up the shaving razor and place it in the bin",
        # "pick up the soup can and place it in the bin",
        # "pick up the toothpaste and place it in the bin",
        # "pick up the yoghurt and place it in the bin",
        # "place the coffee cup on the dish",
        # "place the coffee cup on the dish"
        "pick up the soup can and place it in the bin",
        "pick up the soup can and place it in the bin",
        "clean the whiteboard with the eraser",
        "place the utensils inside the napkin and roll it up",
        "fold the skirt and stack it in a neat pile",
        "fold the trousers and stack it in a neat pile",
        "fold the tshirt and stack it in a neat pile",
        "insert the plug into the socket",
        "pack the luggage",
        "paint the nails",
        "place the bread on the plate",
        "place the trousers on the hanger",
        "place the tshirt on the hanger",
        "put the pillow into the pillowcase",
        "serve the lunch box",
        "sort the stationery into the containers",
        "untangle the cables and put them in the bin",
        "zip up the jacket",
    ])
    
    # Repository name for output dataset
    repo_name: str = "uynitsuj/oreo_xmi_subsample_2"
    
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
    
    # Calibration files
    left_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Left_Controller_20250603_15/calib_results/controller2franka.npy"
    right_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Right_Controller_20250603_15/calib_results/controller2franka.npy"
    
    # Processing settings
    resize_size: int = 224
    fps: int = 30 # Framerate of original video
    temporal_subsample_factor: int = 2 # Subsample every N frames (1 = no subsampling)
    chunk_size: int = 1000
    max_workers: int = 8
    max_episodes: Optional[int] = None
    skip_videos: bool = False
    first_frame_head_reorient: bool = False

    delta_proprio_keys: str = "z" # Makes the proprioceptive state for this axis delta. Can set to None to keep absolute
    
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


def detect_available_encoders():
    """Detect available hardware and software video encoders."""
    import subprocess
    
    encoders = []
    
    # Test for available encoders by checking ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              universal_newlines=True, timeout=10)
        encoder_output = result.stdout
        
        # Check for hardware encoders (in order of preference)
        hardware_encoders = [
            ('h264_nvenc', 'NVIDIA NVENC H.264'),
            ('hevc_nvenc', 'NVIDIA NVENC H.265'),
            ('h264_qsv', 'Intel Quick Sync H.264'),
            ('hevc_qsv', 'Intel Quick Sync H.265'),
            ('h264_amf', 'AMD VCE H.264'),
            ('hevc_amf', 'AMD VCE H.265'),
            ('h264_videotoolbox', 'Apple VideoToolbox H.264'),
            ('hevc_videotoolbox', 'Apple VideoToolbox H.265'),
        ]
        
        for encoder_name, description in hardware_encoders:
            if encoder_name in encoder_output:
                encoders.append((encoder_name, description, 'hardware'))
        
        # Software encoders as fallback
        software_encoders = [
            ('libx264', 'Software H.264', 'software'),
            ('libx265', 'Software H.265', 'software'),
        ]
        
        for encoder_name, description, enc_type in software_encoders:
            if encoder_name in encoder_output:
                encoders.append((encoder_name, description, enc_type))
                
    except Exception as e:
        print(f"Warning: Could not detect encoders: {e}")
        # Fallback to basic software encoder
        encoders = [('libx264', 'Software H.264 (fallback)', 'software')]
    
    return encoders


def get_encoder_settings(encoder_name: str, quality: str = 'fast') -> dict:
    """Get optimized settings for different encoders."""
    
    settings: dict = {
        'pix_fmt': 'yuv420p',
        'movflags': '+faststart'  # Enable fast start for web playback
    }
    
    if 'nvenc' in encoder_name:
        # NVIDIA NVENC settings
        if quality == 'fastest':
            encoder_settings = {
                'preset': 'p1',      # Fastest preset
                'tune': 'ull',       # Ultra-low latency
                'rc': 'vbr',         # Variable bitrate
                'cq': '28',          # Quality (lower = better, 18-28 typical)
                'b:v': '3M',         # Target bitrate
                'maxrate': '6M',     # Max bitrate
                'bufsize': '6M',     # Buffer size
                'gpu': '0'           # GPU index
            }
        else:  # 'fast'
            encoder_settings = {
                'preset': 'p4',      # Faster preset
                'tune': 'hq',        # High quality
                'rc': 'vbr',
                'cq': '23',
                'b:v': '5M',
                'maxrate': '10M',
                'bufsize': '10M',
                'gpu': '0'
            }
        settings.update(encoder_settings)
    
    elif 'qsv' in encoder_name:
        # Intel Quick Sync settings
        if quality == 'fastest':
            encoder_settings = {
                'preset': 'veryfast',
                'global_quality': '28',
                'look_ahead': '0',
                'b:v': '3M'
            }
        else:  # 'fast'
            encoder_settings = {
                'preset': 'fast',
                'global_quality': '23',
                'look_ahead': '1',
                'b:v': '5M'
            }
        settings.update(encoder_settings)
    
    elif 'amf' in encoder_name:
        # AMD VCE settings
        if quality == 'fastest':
            encoder_settings = {
                'quality': 'speed',
                'rc': 'vbr_peak',
                'qp_i': '28',
                'qp_p': '30',
                'b:v': '3M'
            }
        else:  # 'fast'
            encoder_settings = {
                'quality': 'balanced',
                'rc': 'vbr_peak',
                'qp_i': '22',
                'qp_p': '24',
                'b:v': '5M'
            }
        settings.update(encoder_settings)
    
    elif 'videotoolbox' in encoder_name:
        # Apple VideoToolbox settings
        if quality == 'fastest':
            encoder_settings = {
                'q:v': '65',         # Quality (0-100, higher = better)
                'realtime': '1',     # Real-time encoding
                'b:v': '3M'
            }
        else:  # 'fast'
            encoder_settings = {
                'q:v': '55',
                'b:v': '5M'
            }
        settings.update(encoder_settings)
    
    else:
        # Software encoder fallback (libx264/libx265)
        if quality == 'fastest':
            encoder_settings = {
                'preset': 'ultrafast',
                'crf': '28',
                'tune': 'fastdecode'
            }
        else:  # 'fast'
            encoder_settings = {
                'preset': 'veryfast',
                'crf': '23'
            }
        settings.update(encoder_settings)
    
    return settings


def benchmark_encoder(encoder_name: str, test_frames: List[np.ndarray], fps: int, temporal_subsample_factor: int = 1):
    """Benchmark an encoder with test frames."""
    import tempfile
    import time
    import subprocess
    
    if not test_frames:
        return float('inf')
    
    height, width, _ = test_frames[0].shape
    
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            try:
                # Write test frames
                frame_data = np.stack(test_frames).astype(np.uint8)
                temp_input.write(frame_data.tobytes())
                temp_input.flush()
                
                # Get encoder settings
                settings = get_encoder_settings(encoder_name, 'fastest')
                
                # Build ffmpeg command
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{width}x{height}',
                    '-framerate', str(fps/temporal_subsample_factor),
                    '-i', temp_input.name,
                    '-vcodec', encoder_name,
                ]
                
                # Add encoder-specific settings
                for key, value in settings.items():
                    if key not in ['common']:
                        cmd.extend([f'-{key}', str(value)])
                
                cmd.append(temp_output.name)
                
                # Benchmark encoding time
                start_time = time.time()
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                end_time = time.time()
                
                if result.returncode == 0:
                    encoding_time = end_time - start_time
                    return encoding_time
                else:
                    return float('inf')
                    
            except Exception:
                return float('inf')
            finally:
                # Cleanup
                try:
                    os.unlink(temp_input.name)
                    os.unlink(temp_output.name)
                except:
                    pass


def select_best_encoder(test_frames: List[np.ndarray] = None, fps: int = 30, temporal_subsample_factor: int = 1):
    """Select the best available encoder, optionally with benchmarking."""
    encoders = detect_available_encoders()
    
    if not encoders:
        return 'libx264', 'fast'
    
    print(f"Available encoders: {[(name, desc) for name, desc, _ in encoders]}")
    
    # If we have test frames, benchmark the encoders
    if test_frames and len(test_frames) >= 10:
        print("Benchmarking encoders...")
        benchmark_results = []
        
        # Test up to 3 fastest hardware encoders + software fallback
        test_encoders = [enc for enc in encoders if enc[2] == 'hardware'][:3]
        test_encoders.extend([enc for enc in encoders if enc[2] == 'software'][:1])
        
        for encoder_name, description, enc_type in test_encoders:
            print(f"Testing {encoder_name} ({description})...")
            # Use subset of frames for benchmarking
            test_subset = test_frames[:min(10, len(test_frames))]
            encode_time = benchmark_encoder(encoder_name, test_subset, fps, temporal_subsample_factor)
            if encode_time != float('inf'):
                benchmark_results.append((encoder_name, encode_time, description))
                print(f"  {encoder_name}: {encode_time:.2f}s for {len(test_subset)} frames")
            else:
                print(f"  {encoder_name}: Failed")
        
        if benchmark_results:
            # Sort by encoding time (fastest first)
            benchmark_results.sort(key=lambda x: x[1])
            best_encoder = benchmark_results[0][0]
            print(f"Best encoder: {best_encoder} ({benchmark_results[0][2]})")
            
            # Use fastest quality for best performance
            quality = 'fastest' if any('nvenc' in best_encoder or 'qsv' in best_encoder or 'amf' in best_encoder 
                                    for best_encoder in [best_encoder]) else 'fast'
            return best_encoder, quality
    
    # Default selection without benchmarking
    # Prefer hardware encoders in order of typical performance
    preferred_order = ['h264_nvenc', 'h264_qsv', 'h264_amf', 'h264_videotoolbox', 'libx264']
    
    for preferred in preferred_order:
        for encoder_name, description, enc_type in encoders:
            if encoder_name == preferred:
                quality = 'fastest' if enc_type == 'hardware' else 'fast'
                print(f"Selected encoder: {encoder_name} ({description}) with {quality} quality")
                return encoder_name, quality
    
    # Fallback to first available
    encoder_name, description, enc_type = encoders[0]
    quality = 'fastest' if enc_type == 'hardware' else 'fast'
    print(f"Using fallback encoder: {encoder_name} ({description}) with {quality} quality")
    return encoder_name, quality


def encode_video_optimized(frames: List[np.ndarray], save_path: Path, fps: int, temporal_subsample_factor: int = 1,
                          encoder_name: str = None, quality: str = 'fast'):
    """Encode frames into a video using optimized ffmpeg settings."""
    import subprocess
    import tempfile
    
    if not frames:
        print(f"Error: No frames provided for encoding to {save_path}")
        return
    
    height, width, _ = frames[0].shape
    
    # Get encoder settings
    settings = get_encoder_settings(encoder_name, quality)
    
    # Create temporary raw video file
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
        temp_path = temp_file.name
        # Write frames as raw video data
        frame_data = np.stack(frames).astype(np.uint8)
        temp_file.write(frame_data.tobytes())
    
    try:
        # Calculate effective framerate (frames are already subsampled)
        effective_fps = fps / temporal_subsample_factor
        
        # Build optimized ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-framerate', str(effective_fps),
            '-i', temp_path,
            '-vcodec', encoder_name,
        ]
        
        # Add encoder-specific settings
        for key, value in settings.items():
            if key not in ['common']:
                cmd.extend([f'-{key}', str(value)])
        
        # Add common settings
        cmd.extend(['-r', str(effective_fps)])
        cmd.append(str(save_path))
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            print(f"Warning: Optimized encoding failed for {save_path}, trying fallback")
            print(f"Error: {result.stderr.decode()}")
            # Fallback to simple encoding
            encode_video_simple(frames, save_path, fps, temporal_subsample_factor)
        
    except Exception as e:
        print(f"Exception in optimized encoding for {save_path}: {e}")
        # Fallback to simple encoding
        encode_video_simple(frames, save_path, fps, temporal_subsample_factor)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def encode_video_simple(frames: List[np.ndarray], save_path: Path, fps: int, temporal_subsample_factor: int = 1):
    """Simple fallback encoding function."""
    import subprocess
    import tempfile
    
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
        temp_path = temp_file.name
        frame_data = np.stack(frames).astype(np.uint8)
        temp_file.write(frame_data.tobytes())
    
    try:
        # Calculate effective framerate (frames are already subsampled)
        effective_fps = fps / temporal_subsample_factor
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-framerate', str(effective_fps),
            '-i', temp_path,
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-pix_fmt', 'yuv420p',
            '-r', str(effective_fps),
            str(save_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            print(f"Simple encoding also failed for {save_path}: {result.stderr.decode()}")
    except Exception as e:
        print(f"Simple encoding exception for {save_path}: {e}")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


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
    
    # Load controller calibration transformations
    left_controller_calib_tf = vtf.SE3.from_matrix(np.load(cfg.left_controller_calib)).inverse()
    right_controller_calib_tf = vtf.SE3.from_matrix(np.load(cfg.right_controller_calib)).inverse()
    
    # Transform from Quest coordinate system to world coordinate system
    q2w = vtf.SE3.from_rotation_and_translation(
        vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
    )

    # HEAD PROCESSING
    # Determine direction that head z axis is pointing in the first frame to reorient the RBY1 base frame
    
    head_z_tf = vtf.SE3.from_matrix(action_data["action-left-head"][0])
    head_data_all = vtf.SE3.from_matrix(action_data["action-left-head"])

    head_data_all = q2w @ head_data_all
    head_z_tf = q2w @ head_z_tf

    # average head height
    head_height = np.mean(head_data_all.wxyz_xyz[:, -1])
    # print(f"Average head height: {head_height}m")

    head_translation = np.array([head_z_tf.translation()[0], -head_z_tf.translation()[1], 0.0])
    if cfg.first_frame_head_reorient:
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
        # print(f"Head z axis angle: {head_z_axis_angle}")
    else:
        # Default to the direction of left gripper z axis (worried that randomly oriented world frame proprio input could mess up normalization statistics)
        # Make the current hand z axis point toward world negative x axis

        left_hand_matrix = action_data["action-left-hand_in_quest_world_frame"][0]
        world_frame = action_data["action-left-quest_world_frame"][0]
        left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
        left_hand_tf = q2w @ vtf.SE3.from_matrix(world_frame) @ left_hand_tf

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

        right_hand_matrix = action_data["action-right-hand_in_quest_world_frame"][0]
        right_world_frame = action_data["action-right-quest_world_frame"][0]
        right_hand_in_world = np.linalg.inv(world_frame) @ right_world_frame @ right_hand_matrix
        right_hand_tf = vtf.SE3.from_matrix(right_hand_in_world)
        right_hand_tf = q2w @ vtf.SE3.from_matrix(right_world_frame) @ right_hand_tf

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
        rby1_base_frame_wxyz = (vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi + (hand_z_axis_angle_left + hand_z_axis_angle_right) / 2.0)).wxyz



    rby1_base_frame_position = head_translation

    # LEFT HAND PROCESSING
    left_hand_matrix = action_data["action-left-hand_in_quest_world_frame"]
    world_frame = action_data["action-left-quest_world_frame"]
    left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
    left_hand_tf = q2w @ vtf.SE3.from_matrix(world_frame) @ left_hand_tf

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
    offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.identity(), np.array([-0.08275, 0.0, 0.005]))
    ee_tf = yaw_45 @ offset @ pitch_180

    tf_left_ee_ik_target = left_hand_tf_reflected @ left_controller_calib_tf @ ee_tf

    left_ee_ik_target_handle_position = tf_left_ee_ik_target.wxyz_xyz[:, -3:]
    left_ee_ik_target_handle_wxyz = tf_left_ee_ik_target.wxyz_xyz[:, :4]

    # RIGHT HAND PROCESSING
    left_quest_world_frame = action_data["action-left-quest_world_frame"]
    right_hand_matrix = action_data["action-right-hand_in_quest_world_frame"]
    right_world_frame = action_data["action-right-quest_world_frame"]
    right_hand_in_world = np.linalg.inv(left_quest_world_frame) @ right_world_frame @ right_hand_matrix
    right_hand_tf = vtf.SE3.from_matrix(right_hand_in_world)
    right_hand_tf = q2w @ vtf.SE3.from_matrix(right_world_frame) @ right_hand_tf

    right_hand_tf_pos = right_hand_tf.wxyz_xyz[:, -3:]
    right_hand_tf_pos[:, 1] = -right_hand_tf_pos[:, 1]

    right_hand_tf_reflected = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(
        -right_hand_tf.rotation().as_rpy_radians().roll,
        right_hand_tf.rotation().as_rpy_radians().pitch,
        -right_hand_tf.rotation().as_rpy_radians().yaw,
    ), right_hand_tf_pos)

    tf_right_ee_ik_target = right_hand_tf_reflected @ right_controller_calib_tf @ ee_tf

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

    # Get gripper positions
    left_gripper_pos = joint_data['left-joint_pos'][..., None]
    right_gripper_pos = joint_data['right-joint_pos'][..., None]

    left_gripper_action = action_data['action-left-pos']
    right_gripper_action = action_data['action-right-pos']

    # Check array lengths are consistent  
    assert len(left_gripper_pos) == len(right_gripper_pos)

    # Convert quaternions to 6d rotation representation
    # Convert from wxyz quaternions to rotation matrices, then to 6D representation
    left_rot_matrices = vtf.SO3(wxyz=left_ee_ik_target_handle_wxyz).as_matrix()  # Shape: (N, 3, 3)
    right_rot_matrices = vtf.SO3(wxyz=right_ee_ik_target_handle_wxyz).as_matrix()  # Shape: (N, 3, 3)
    
    # Convert rotation matrices to 6D representation using matrix_utils
    left_6d_rot = rot_mat_to_rot_6d(left_rot_matrices)  # Shape: (N, 6)
    right_6d_rot = rot_mat_to_rot_6d(right_rot_matrices)  # Shape: (N, 6)
    
    # Ensure all arrays have the same length
    seq_length = max(
        len(left_6d_rot), len(right_6d_rot),
        len(left_ee_ik_target_handle_position), len(right_ee_ik_target_handle_position),
        len(left_gripper_pos), len(right_gripper_pos)
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
    left_ee_ik_target_handle_position = pad_to_length(left_ee_ik_target_handle_position, seq_length)
    right_ee_ik_target_handle_position = pad_to_length(right_ee_ik_target_handle_position, seq_length)
    left_gripper_pos = pad_to_length(left_gripper_pos, seq_length)
    right_gripper_pos = pad_to_length(right_gripper_pos, seq_length)

    left_gripper_action = pad_to_length(left_gripper_action, seq_length)
    right_gripper_action = pad_to_length(right_gripper_action, seq_length)
    
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


    proprio_data = np.concatenate([
        left_6d_rot, left_ee_ik_target_handle_position, left_gripper_pos,
        right_6d_rot, right_ee_ik_target_handle_position, right_gripper_pos
    ], axis=1)

    action_data = np.concatenate([
        left_6d_rot, left_ee_ik_target_handle_position, left_gripper_action,
        right_6d_rot, right_ee_ik_target_handle_position, right_gripper_action
    ], axis=1)

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
    
    # VALIDATION: Check final processed arrays
    is_valid, error_msg = validate_array_data(states, "states", expected_shape=(seq_length, 20))
    if not is_valid:
        print(f"❌ States validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"States validation failed: {error_msg}", cfg, episode_idx)
        return None, None
    
    is_valid, error_msg = validate_array_data(actions, "actions", expected_shape=(seq_length, 20))
    if not is_valid:
        print(f"❌ Actions validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Actions validation failed: {error_msg}", cfg, episode_idx)
        return None, None

    
    return states, actions


def process_episode_in_chunks(episode_data: dict, cfg: XMIConfig, max_chunk_frames: int = 1000, episode_path: Path = None, episode_idx: int = None) -> list:
    """Process episode data in memory-efficient chunks to handle long episodes."""
    
    # Process XMI transforms first
    states, actions = process_xmi_transforms(episode_data, cfg, episode_path, episode_idx)
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
            
            record = {
                "state": state.tolist(),
                "actions": action.tolist(),
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
    episode_data = load_episode_data(episode_path, cfg, base_dir, idx)
    if not episode_data:
        print(f"  ❌ Failed to load episode {idx}")
        move_problematic_episode(episode_path, "Failed to load episode data", cfg, idx)
        return None
    
    # Process episode in memory-efficient chunks
    try:
        records = process_episode_in_chunks(episode_data, cfg, max_chunk_frames=cfg.max_frames_per_chunk, episode_path=episode_path, episode_idx=idx)
        if not records:
            print(f"  ❌ No valid data in episode {idx}")
            # Episode already moved by process_episode_in_chunks if it was a validation failure
            return None
        
        seq_length = len(records)
        # print(f"  Episode {idx}: {seq_length} frames total")
                
    except Exception as e:
        error_msg = f"Error processing episode {idx}: {e}"
        print(f"  ❌ {error_msg}")
        move_problematic_episode(episode_path, error_msg, cfg, idx)
        return None
    
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
    
    # Write info.json
    features = {
        "state": {
            "dtype": "float32", 
            "shape": [20],  # XMI uses 20-dimensional state space
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": [20],  # XMI uses 20-dimensional action space
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
