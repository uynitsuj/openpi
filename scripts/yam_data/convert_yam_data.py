#!/usr/bin/env python3
"""
Direct YAMS to LeRobot format converter.

This script bypasses the LeRobot dataset creation completely and directly creates
the dataset in the same format as LeRobot, avoiding memory accumulation and
ffmpeg-python import issues.


FOR LARGE DATASETS, HF MAY RATE LIMIT WITH REGULAR PUSH SO INSTEAD SET push_to_hub=False AND USE:

huggingface-cli upload-large-folder <repo-id> <local-path> --repo-type=dataset

"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
from typing import Literal, Optional, List, Tuple
import shutil

from lerobot.constants import HF_LEROBOT_HOME
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro


@dataclass
class YAMSConfig:
    yam_data_path: str | list[str] = field(
        default_factory=lambda: [
            # "/nfs_us/sz_test_01_sim/20250619" # Red Cube Pick Mujoco Sim DATA
            # "/nfs_us/datasets/sim_red_cube_20250630"  # new Red Cube Pick Mujoco Sim DATA saved in local
            # "/home/local_dataset/sim_red_cube" # extended Red Cube Pick Mujoco Sim DATA saved in local
            # "/home/test_sim_data" # Red Cube Pick Mujoco Sim DATA one episode for testing
            "/nfs_us/data/sz_02/20250908" # Random YAM test data
        ]
    )

    repo_name: str = (
        "uynitsuj/yam_random_test_data"  # TODO: Change this before running
    )

    language_instruction: str = "Test"  # Gets overwritten by the task name in episode metadata # TODO: Change this before running

    # YAMS camera keys
    camera_keys: list[str] = field(
        default_factory=lambda: [
            "left_camera-images-rgb",
            "right_camera-images-rgb",
            "top_camera-images-rgb",
        ]
    )

    rotate_left_right_image: bool = (
        False  # TODO: Change this before running # If True, will rotate left and right camera images up side down
    )
    resize_size: int = 224  # image size
    fps: int = 30  # TODO: Change this before running # real data: 30, mujoco sim: 11
    chunk_size: int = 1000  # number of frames per chunk (for memory considerations)
    max_workers: int = 1  # Set lower on machines with less memory -- must be 1 for cartesian
    no_filter_quality: bool = True  # If True, will not filter out low quality episodes
    max_episodes: int | None = None  # If specified, will only process this many episodes
    skip_videos: bool = False  # If True, will not process videos
    single_arm: bool = False  # If True, expect only left arm data (7 DoF), if False expect bimanual data (14 DoF)
    overwrite_existing_data: bool = True  # If True, will overwrite existing data if repo_id exists

    """
    Huggingface hub settings:
    """
    push_to_hub: bool = False  # If True, will push to huggingface hub after processing (Not required, can convert dataset and train using the local directory)
    push_to_hub_only: bool = False  # Only push existing dataset to hub, skip processing

    action_space: Literal["abs_joint", "abs_cartesian"] = (
        "abs_joint"  # "abs_joint" for absolute joint positions, "abs_cartesian" for absolute cartesian positions
    )

    # Memory management settings
    max_frames_per_chunk: int = 1000  # Process episodes in chunks to avoid OOM on long episodes
    
    # Video processing settings
    temporal_subsample_factor: int = 1  # Subsample every N frames (1 = no subsampling)
    crop_images_to_square: bool = True  # Whether to crop images to square
    
    # Validation settings
    max_se3_diff_in_meters: float = 0.5  # Maximum allowed SE3 difference in meters
    max_se3_diff_in_degrees: float = 40  # Maximum allowed SE3 difference in degrees
    video_timestamp_tolerance_s: float = 0.0001  # Maximum allowed deviation from expected frame interval
    
    # Debug settings
    debug: bool = False
    problematic_data_dir: Optional[str] = None  # Directory to move problematic episodes

    robot: object | None = field(
        default=None, repr=False
    )  # YAMSBaseInterface object (Used for FK in the case of abs_cartesian)
    use_hugging_face: bool = False
    
    def __post_init__(self):
        # Set default problematic data directory if not specified
        if self.problematic_data_dir is None:
            self.problematic_data_dir = f"./problematic_episodes_{self.repo_name.replace('/', '_')}"

# Import utility modules
try:
    from .data_utils import calculate_actions
    from .data_utils import calculate_actions_cartesian
    from .data_utils import find_episode_directories
    from .data_utils import is_episode_good_quality
    from .data_utils import load_yams_episode_data_fast
    from .data_utils import process_joint_data
except ImportError:
    from data_utils import calculate_actions
    from data_utils import calculate_actions_cartesian
    from data_utils import find_episode_directories
    from data_utils import is_episode_good_quality
    from data_utils import load_yams_episode_data_fast
    from data_utils import process_joint_data

# Import video processing
try:
    from openpi.utils.video_processor import resize_and_pad_video, get_video_resolution
except ImportError:
    print("Warning: Video processing utilities not available. Video processing will be limited.")
    resize_and_pad_video = None
    get_video_resolution = None


def extract_task_name_from_episode(episode_data: dict, episode_path: Path) -> str:
    """Extract task name from episode metadata or path."""
    # First try to get from metadata
    if episode_data.get("metadata"):
        metadata = episode_data["metadata"]

        # Common metadata fields that might contain task info
        possible_task_fields = [
            "task",
            "task_name",
            "language_instruction",
            "instruction",
            "description",
            "task_description",
        ]

        for field in possible_task_fields:
            if metadata.get(field):
                return str(metadata[field])

    # Fallback: try to extract from episode directory name
    episode_name = episode_path.name

    # If episode name contains identifiable task keywords, extract them
    task_keywords = {
        "load_dishes": "Load dishes into dishwasher",
        "load_dishwasher": "Load dishes into dishwasher",
        "dishes": "Load dishes into dishwasher",
        "dishwasher": "Load dishes into dishwasher",
        "bimanual": "Perform bimanual manipulation task",
        "manipulation": "Perform manipulation task",
        "pick_place": "Pick and place objects",
        "sorting": "Sort objects",
        "stacking": "Stack objects",
        "cleaning": "Clean surfaces",
    }

    episode_lower = episode_name.lower()
    for keyword, task_name in task_keywords.items():
        if keyword in episode_lower:
            return task_name

    # Final fallback
    return "Perform bimanual manipulation task"


# Import LeRobotDataset for hub operations
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import write_episode_stats

    HAS_LEROBOT = True
except ImportError:
    print("Warning: LeRobot not available. Hub push functionality disabled.")
    HAS_LEROBOT = False


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def validate_episode_data(episode_data: dict) -> tuple[bool, str]:
    """
    Validate that episode data contains all required fields and has valid structure.
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(episode_data, dict):
        return False, "Episode data is not a dictionary"
    
    # Check required top-level keys
    required_keys = ['joint_data']
    for key in required_keys:
        if key not in episode_data:
            return False, f"Missing required key: {key}"
        if not isinstance(episode_data[key], dict):
            return False, f"Key '{key}' is not a dictionary"
    
    # Check that arrays have reasonable lengths
    try:
        joint_data = episode_data['joint_data']
        
        # Check for at least some joint data
        joint_keys = list(joint_data.keys())
        import pdb; pdb.set_trace()
        if not joint_keys:
            return False, "No joint data found"
            
        # Get lengths of key arrays
        joint_lengths = []
        for key, data in joint_data.items():
            if hasattr(data, '__len__'):
                joint_lengths.append(len(data))
            else:
                return False, f"joint_data['{key}'] is not array-like"
        
        # Check if all arrays have reasonable and consistent lengths
        if len(set(joint_lengths)) > 1:  # All arrays must have same length
            return False, f"Inconsistent array lengths: {joint_lengths}"
        
        min_length = min(joint_lengths) if joint_lengths else 0
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


def move_problematic_episode(episode_path: Path, error_msg: str, cfg: YAMSConfig, episode_idx: int = None) -> bool:
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


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resize images with padding to maintain aspect ratio."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize single image with padding using PIL."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


def detect_available_encoders():
    """Detect available hardware and software video encoders."""
    import subprocess

    encoders = []

    # Test for available encoders by checking ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        encoder_output = result.stdout

        # Check for hardware encoders (in order of preference)
        hardware_encoders = [
            ("h264_nvenc", "NVIDIA NVENC H.264"),
            ("hevc_nvenc", "NVIDIA NVENC H.265"),
            ("h264_qsv", "Intel Quick Sync H.264"),
            ("hevc_qsv", "Intel Quick Sync H.265"),
            ("h264_amf", "AMD VCE H.264"),
            ("hevc_amf", "AMD VCE H.265"),
            ("h264_videotoolbox", "Apple VideoToolbox H.264"),
            ("hevc_videotoolbox", "Apple VideoToolbox H.265"),
        ]

        for encoder_name, description in hardware_encoders:
            if encoder_name in encoder_output:
                encoders.append((encoder_name, description, "hardware"))

        # Software encoders as fallback
        software_encoders = [
            ("libx264", "Software H.264", "software"),
            ("libx265", "Software H.265", "software"),
        ]

        for encoder_name, description, enc_type in software_encoders:
            if encoder_name in encoder_output:
                encoders.append((encoder_name, description, enc_type))

    except Exception as e:
        print(f"Warning: Could not detect encoders: {e}")
        # Fallback to basic software encoder
        encoders = [("libx264", "Software H.264 (fallback)", "software")]

    return encoders


def get_encoder_settings(encoder_name: str, quality: str = "fast"):
    """Get optimized settings for different encoders."""

    settings = {
        "common": {
            "pix_fmt": "yuv420p",
            "movflags": "+faststart",  # Enable fast start for web playback
        }
    }

    if "nvenc" in encoder_name:
        # NVIDIA NVENC settings
        if quality == "fastest":
            settings.update(
                {
                    "preset": "p1",  # Fastest preset
                    "tune": "ull",  # Ultra-low latency
                    "rc": "vbr",  # Variable bitrate
                    "cq": "28",  # Quality (lower = better, 18-28 typical)
                    "b:v": "3M",  # Target bitrate
                    "maxrate": "6M",  # Max bitrate
                    "bufsize": "6M",  # Buffer size
                    "gpu": "0",  # GPU index
                }
            )
        else:  # 'fast'
            settings.update(
                {
                    "preset": "p4",  # Faster preset
                    "tune": "hq",  # High quality
                    "rc": "vbr",
                    "cq": "23",
                    "b:v": "5M",
                    "maxrate": "10M",
                    "bufsize": "10M",
                    "gpu": "0",
                }
            )

    elif "qsv" in encoder_name:
        # Intel Quick Sync settings
        if quality == "fastest":
            settings.update({"preset": "veryfast", "global_quality": "28", "look_ahead": "0", "b:v": "3M"})
        else:  # 'fast'
            settings.update({"preset": "fast", "global_quality": "23", "look_ahead": "1", "b:v": "5M"})

    elif "amf" in encoder_name:
        # AMD VCE settings
        if quality == "fastest":
            settings.update({"quality": "speed", "rc": "vbr_peak", "qp_i": "28", "qp_p": "30", "b:v": "3M"})
        else:  # 'fast'
            settings.update({"quality": "balanced", "rc": "vbr_peak", "qp_i": "22", "qp_p": "24", "b:v": "5M"})

    elif "videotoolbox" in encoder_name:
        # Apple VideoToolbox settings
        if quality == "fastest":
            settings.update(
                {
                    "q:v": "65",  # Quality (0-100, higher = better)
                    "realtime": "1",  # Real-time encoding
                    "b:v": "3M",
                }
            )
        else:  # 'fast'
            settings.update({"q:v": "55", "b:v": "5M"})

    # Software encoder fallback (libx264/libx265)
    elif quality == "fastest":
        settings.update({"preset": "ultrafast", "crf": "28", "tune": "fastdecode"})
    else:  # 'fast'
        settings.update({"preset": "veryfast", "crf": "23"})

    return settings


def benchmark_encoder(encoder_name: str, test_frames: list[np.ndarray], fps: int):
    """Benchmark an encoder with test frames."""
    import subprocess
    import tempfile
    import time

    if not test_frames:
        return float("inf")

    height, width, _ = test_frames[0].shape

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_input:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
            try:
                # Write test frames
                frame_data = np.stack(test_frames).astype(np.uint8)
                temp_input.write(frame_data.tobytes())
                temp_input.flush()

                # Get encoder settings
                settings = get_encoder_settings(encoder_name, "fastest")

                # Build ffmpeg command
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    temp_input.name,
                    "-vcodec",
                    encoder_name,
                ]

                # Add encoder-specific settings
                for key, value in settings.items():
                    if key not in ["common"]:
                        cmd.extend([f"-{key}", str(value)])

                cmd.append(temp_output.name)

                # Benchmark encoding time
                start_time = time.time()
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
                end_time = time.time()

                if result.returncode == 0:
                    encoding_time = end_time - start_time
                    return encoding_time
                return float("inf")

            except Exception:
                return float("inf")
            finally:
                # Cleanup
                try:
                    os.unlink(temp_input.name)
                    os.unlink(temp_output.name)
                except:
                    pass


def select_best_encoder(test_frames: list[np.ndarray] = None, fps: int = 30):
    """Select the best available encoder, optionally with benchmarking."""
    encoders = detect_available_encoders()

    if not encoders:
        return "libx264", "fast"

    print(f"Available encoders: {[(name, desc) for name, desc, _ in encoders]}")

    # If we have test frames, benchmark the encoders
    if test_frames and len(test_frames) >= 10:
        print("Benchmarking encoders...")
        benchmark_results = []

        # Test up to 3 fastest hardware encoders + software fallback
        test_encoders = [enc for enc in encoders if enc[2] == "hardware"][:3]
        test_encoders.extend([enc for enc in encoders if enc[2] == "software"][:1])

        for encoder_name, description, enc_type in test_encoders:
            print(f"Testing {encoder_name} ({description})...")
            # Use subset of frames for benchmarking
            test_subset = test_frames[: min(10, len(test_frames))]
            encode_time = benchmark_encoder(encoder_name, test_subset, fps)
            if encode_time != float("inf"):
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
            quality = (
                "fastest"
                if any(
                    "nvenc" in best_encoder or "qsv" in best_encoder or "amf" in best_encoder
                    for best_encoder in [best_encoder]
                )
                else "fast"
            )
            return best_encoder, quality

    # Default selection without benchmarking
    # Prefer hardware encoders in order of typical performance
    preferred_order = ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "libx264"]

    for preferred in preferred_order:
        for encoder_name, description, enc_type in encoders:
            if encoder_name == preferred:
                quality = "fastest" if enc_type == "hardware" else "fast"
                print(f"Selected encoder: {encoder_name} ({description}) with {quality} quality")
                return encoder_name, quality

    # Fallback to first available
    encoder_name, description, enc_type = encoders[0]
    quality = "fastest" if enc_type == "hardware" else "fast"
    print(f"Using fallback encoder: {encoder_name} ({description}) with {quality} quality")
    return encoder_name, quality


def encode_video_optimized(
    frames: list[np.ndarray], save_path: Path, fps: int, encoder_name: str = None, quality: str = "fast"
):
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
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_file:
        temp_path = temp_file.name
        # Write frames as raw video data
        frame_data = np.stack(frames).astype(np.uint8)
        temp_file.write(frame_data.tobytes())

    try:
        # Build optimized ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            temp_path,
            "-vcodec",
            encoder_name,
        ]

        # Add encoder-specific settings
        for key, value in settings.items():
            if key not in ["common"]:
                cmd.extend([f"-{key}", str(value)])

        # Add common settings
        cmd.extend(["-r", str(fps)])
        cmd.append(str(save_path))

        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        if result.returncode != 0:
            print(f"Warning: Optimized encoding failed for {save_path}, trying fallback")
            print(f"Error: {result.stderr.decode()}")
            # Fallback to simple encoding
            encode_video_simple(frames, save_path, fps)

    except Exception as e:
        print(f"Exception in optimized encoding for {save_path}: {e}")
        # Fallback to simple encoding
        encode_video_simple(frames, save_path, fps)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def encode_video_simple(frames: list[np.ndarray], save_path: Path, fps: int):
    """Simple fallback encoding function."""
    import subprocess
    import tempfile

    if not frames:
        return

    height, width, _ = frames[0].shape

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_file:
        temp_path = temp_file.name
        frame_data = np.stack(frames).astype(np.uint8)
        temp_file.write(frame_data.tobytes())

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            temp_path,
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(save_path),
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        if result.returncode != 0:
            print(f"Simple encoding also failed for {save_path}: {result.stderr.decode()}")
    except Exception as e:
        print(f"Simple encoding exception for {save_path}: {e}")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def compute_basic_episode_stats(episode_idx: int, episode_info: dict, cfg: YAMSConfig, base_dir: Path) -> dict:
    """Compute basic statistics for an episode to create v2.1 compatible episodes_stats.jsonl"""

    # Load the episode parquet file
    chunk_id = episode_idx // cfg.chunk_size
    parquet_path = base_dir / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_idx:06d}.parquet"

    if cfg.action_space == "abs_joint":
        if cfg.single_arm:
            state_dim = 7
            action_dim = 7
        else:
            state_dim = 14
            action_dim = 14
    elif cfg.action_space == "abs_cartesian":
        assert not cfg.single_arm
        state_dim = 20
        action_dim = 20
    else:
        raise ValueError(f"Invalid action space: {cfg.action_space}, or not implemented yet")

    if not parquet_path.exists():
        # Return minimal stats if parquet doesn't exist
        return {
            "state": {
                "min": np.zeros(state_dim, dtype=np.float32),
                "max": np.zeros(state_dim, dtype=np.float32),
                "mean": np.zeros(state_dim, dtype=np.float32),
                "std": np.ones(state_dim, dtype=np.float32),
                "count": np.array([1], dtype=np.int64),
            },
            "actions": {
                "min": np.zeros(action_dim, dtype=np.float32),
                "max": np.zeros(action_dim, dtype=np.float32),
                "mean": np.zeros(action_dim, dtype=np.float32),
                "std": np.ones(action_dim, dtype=np.float32),
                "count": np.array([1], dtype=np.int64),
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
            data = df[feature_name].values.astype(np.float32)
            if len(data.shape) > 1:
                data = data.flatten()

            # For 1D data, LeRobot expects keepdims=True if original was 1D
            episode_stats[feature_name] = {
                "min": np.array([data.min()], dtype=np.float32),
                "max": np.array([data.max()], dtype=np.float32),
                "mean": np.array([data.mean()], dtype=np.float32),
                "std": np.array([data.std()], dtype=np.float32),
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
        with open(tasks_file) as f:
            for line in f:
                task_data = json.loads(line.strip())
                existing_tasks[task_data["task"]] = task_data["task_index"]

    # Add new tasks if they don't exist
    new_tasks_added = False
    for task in tasks:
        if task not in existing_tasks:
            task_index = len(existing_tasks)
            existing_tasks[task] = task_index
            new_tasks_added = True

            # Append new task immediately
            with open(tasks_file, "a") as f:
                f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

    return existing_tasks


def process_episode_in_chunks(episode_data: dict, cfg: YAMSConfig, max_chunk_frames: int = 1000, episode_path: Path = None, episode_idx: int = None) -> list:
    """Process episode data in memory-efficient chunks to handle long episodes."""
    
    # VALIDATION: Check episode data structure
    is_valid, error_msg = validate_episode_data(episode_data)
    if not is_valid:
        print(f"❌ Episode data validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Episode data validation failed: {error_msg}", cfg, episode_idx)
        return []

    # Process joint data first (this is relatively small)
    full_joint_state, full_joint_action = process_joint_data(episode_data["joint_data"], cfg.single_arm)
    if full_joint_state is None:
        if episode_path is not None:
            move_problematic_episode(episode_path, "Failed to process joint data", cfg, episode_idx)
        return []

    # Determine sequence length
    total_length = len(full_joint_state) - 1  # -1 because we need next state for actions
    if total_length <= 0:
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Episode too short: {total_length} frames", cfg, episode_idx)
        return []

    # Calculate actions for the full episode (joint data is manageable)
    try:
        if cfg.action_space == "abs_joint":
            states, actions = calculate_actions(full_joint_state, full_joint_action, total_length)
        elif cfg.action_space == "abs_cartesian":
            states, actions = calculate_actions_cartesian(full_joint_state, full_joint_action, total_length, cfg.robot)
        else:
            raise ValueError(f"Invalid action space: {cfg.action_space}, or not implemented yet")
    except Exception as e:
        error_msg = f"Failed to calculate actions: {str(e)}"
        print(f"❌ {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, error_msg, cfg, episode_idx)
        return []

    # VALIDATION: Check processed arrays
    state_dim = 7 if cfg.single_arm else 14
    if cfg.action_space == "abs_cartesian":
        state_dim = 20
        
    is_valid, error_msg = validate_array_data(states, "states", expected_shape=(total_length, state_dim))
    if not is_valid:
        print(f"❌ States validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"States validation failed: {error_msg}", cfg, episode_idx)
        return []
    
    is_valid, error_msg = validate_array_data(actions, "actions", expected_shape=(total_length, state_dim))
    if not is_valid:
        print(f"❌ Actions validation failed: {error_msg}")
        if episode_path is not None:
            move_problematic_episode(episode_path, f"Actions validation failed: {error_msg}", cfg, episode_idx)
        return []

    # Apply temporal subsampling if configured
    original_total_length = total_length
    if cfg.temporal_subsample_factor > 1:
        global_subsample_indices = list(range(0, original_total_length, cfg.temporal_subsample_factor))
        states = states[global_subsample_indices]
        actions = actions[global_subsample_indices]
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


def process_yam_episode(
    idx: int,
    episode_path: Path,
    cfg: YAMSConfig,
    episode_base: Path,
    base_dir: Path,
    encoder_name: str = None,
):
    """Process a single YAM episode and save it directly to LeRobot format."""

    # print(f"Processing episode {idx}: {episode_path.name}")

    # Quality filtering
    if not cfg.no_filter_quality and not is_episode_good_quality(episode_path):
        print(f"  Skipping episode {idx}: poor quality")
        return None

    # Load episode data
    episode_data = load_yams_episode_data_fast(episode_path, cfg, idx)
    if not episode_data:
        print(f"  ❌ Failed to load episode {idx}")
        move_problematic_episode(episode_path, "Failed to load episode data", cfg, idx)
        return None

    # Extract task name from episode metadata instead of using hardcoded value
    task_name = extract_task_name_from_episode(episode_data, episode_path)

    # Process episode in memory-efficient chunks
    records = process_episode_in_chunks(episode_data, cfg, max_chunk_frames=cfg.max_frames_per_chunk, episode_path=episode_path, episode_idx=idx)
    if not records:
        print(f"  ❌ No valid data in episode {idx}")
        # Episode already moved by process_episode_in_chunks if it was a validation failure
        return None

    seq_length = len(records)
    print(f"  Episode {idx}: {seq_length} frames total")

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
        move_problematic_episode(episode_path, f"Failed to save parquet: {e}", cfg, idx)
        return None

    # Process videos if not skipping
    # if not cfg.skip_videos and resize_and_pad_video is not None:
    #     chunk_id = idx // cfg.chunk_size
        
    #     # Process each camera video
    #     for cam_key in cfg.camera_keys:
    #         # Look for video files in the episode directory
    #         video_patterns = [
    #             episode_path / f"{cam_key}.mp4",
    #             episode_path / f"{cam_key}.avi",
    #             episode_path / f"{cam_key}.mov"
    #         ]
            
    #         input_video_path = None
    #         for pattern in video_patterns:
    #             if pattern.exists():
    #                 input_video_path = pattern
    #                 break
            
    #         if input_video_path is None:
    #             continue  # Skip if no video file found for this camera
            
    #         # Set up output path
    #         video_dir = base_dir / "videos" / f"chunk-{chunk_id:03d}" / cam_key
    #         video_dir.mkdir(parents=True, exist_ok=True)
    #         output_video_path = video_dir / f"episode_{idx:06d}.mp4"
            
            # try:
            #     # Use the improved video processing from XMI
            #     success = resize_and_pad_video(
            #         input_path=str(input_video_path),
            #         output_path=str(output_video_path),
            #         target_size=cfg.resize_size,
            #         fps=cfg.fps // cfg.temporal_subsample_factor,
            #         frame_stride=cfg.temporal_subsample_factor,
            #         crop_to_square=cfg.crop_images_to_square,
            #         encoder=encoder_name or "h264_nvenc",
            #         overwrite=True
            #     )
                
            #     if not success:
            #         print(f"  ⚠️  Video processing failed for {cam_key} in episode {idx}")
                    
            # except Exception as e:
            #     print(f"  ⚠️  Video processing error for {cam_key} in episode {idx}: {e}")

    # Compute and write episode stats immediately
    episode_stats = compute_basic_episode_stats(idx, {"length": seq_length}, cfg, base_dir)
    if HAS_LEROBOT:
        write_episode_stats(idx, episode_stats, base_dir)

    # Write episode metadata immediately
    episode_metadata = {
        "episode_index": idx,
        "tasks": [task_name],
        "length": seq_length,
        "original_episode_name": episode_path.name,  # Add original episode directory name
        "original_episode_path": str(episode_path),  # Add full original path for reference
    }

    task_mapping = write_episode_metadata_immediately(episode_metadata, [task_name], base_dir)

    # Update task index in the episode metadata
    task_index = task_mapping.get(task_name, 0)
    episode_metadata["task_index"] = task_index

    # Clean up memory
    del episode_data, records
    gc.collect()

    # print(f"  ✅ Completed episode {idx}: {seq_length} frames, task '{task_name}'")

    # Return metadata for final statistics
    return episode_metadata


def find_completed_episodes(base_dir: Path, total_episodes: int, chunk_size: int) -> set[int]:
    """Find episodes that have already been processed by checking for existing parquet files."""
    completed_episodes = set()

    data_dir = base_dir / "data"
    if not data_dir.exists():
        return completed_episodes

    # Check each chunk directory for completed episodes
    for chunk_id in range((total_episodes + chunk_size - 1) // chunk_size):
        chunk_dir = data_dir / f"chunk-{chunk_id:03d}"
        if chunk_dir.exists():
            for parquet_file in chunk_dir.glob("episode_*.parquet"):
                # Extract episode index from filename
                episode_name = parquet_file.stem  # removes .parquet
                if episode_name.startswith("episode_"):
                    try:
                        episode_idx = int(episode_name.split("_")[1])
                        completed_episodes.add(episode_idx)
                    except (ValueError, IndexError):
                        continue

    return completed_episodes


def reconstruct_metadata_from_files(base_dir: Path, completed_episodes: set[int], cfg: YAMSConfig) -> tuple[list, dict]:
    """Reconstruct missing metadata from existing parquet files for backwards compatibility."""
    print("Reconstructing metadata from existing files for backwards compatibility...")

    reconstructed_episodes = []
    reconstructed_tasks = {}
    task_counter = 0

    for episode_idx in sorted(completed_episodes):
        # Load the parquet file to extract metadata
        chunk_id = episode_idx // cfg.chunk_size
        parquet_path = base_dir / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_idx:06d}.parquet"

        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                episode_length = len(df)

                # Try to determine task name from episode directory structure or use default
                # Since we don't have access to original episode path here, use default
                default_task = "Perform bimanual manipulation task"

                # Add task if not already present
                if default_task not in reconstructed_tasks:
                    reconstructed_tasks[default_task] = task_counter
                    task_counter += 1

                task_index = reconstructed_tasks[default_task]

                # Create episode metadata
                episode_metadata = {
                    "episode_index": episode_idx,
                    "tasks": [default_task],
                    "length": episode_length,
                    "task_index": task_index,
                }

                reconstructed_episodes.append(episode_metadata)
                print(f"  Reconstructed metadata for episode {episode_idx}: {episode_length} frames")

            except Exception as e:
                print(f"  Warning: Could not reconstruct metadata for episode {episode_idx}: {e}")

    return reconstructed_episodes, reconstructed_tasks


def filter_episodes_for_resume(
    episode_dirs: list[Path], base_dir: Path, chunk_size: int
) -> tuple[list[Path], list[int]]:
    """Filter episode directories to only process incomplete episodes for resume functionality."""
    total_episodes = len(episode_dirs)
    completed_episodes = find_completed_episodes(base_dir, total_episodes, chunk_size)

    if completed_episodes:
        print(f"Found {len(completed_episodes)} already completed episodes")

        # Check if metadata files exist (for backwards compatibility)
        episodes_file = base_dir / "meta" / "episodes.jsonl"
        tasks_file = base_dir / "meta" / "tasks.jsonl"

        if not episodes_file.exists() or not tasks_file.exists():
            print("Metadata files missing - this appears to be from an old incomplete run")
            print("Reconstructing metadata from existing files...")

            # Reconstruct metadata from parquet files
            reconstructed_episodes, reconstructed_tasks = reconstruct_metadata_from_files(
                base_dir,
                completed_episodes,
                YAMSConfig(),  # Use default config for reconstruction
            )

            # Write the reconstructed metadata
            base_dir.joinpath("meta").mkdir(exist_ok=True)

            # Write episodes.jsonl
            with open(episodes_file, "w") as f:
                for episode in reconstructed_episodes:
                    f.write(json.dumps(episode) + "\n")

            # Write tasks.jsonl
            with open(tasks_file, "w") as f:
                for task, task_index in reconstructed_tasks.items():
                    f.write(json.dumps({"task_index": task_index, "task": task}) + "\n")

            print(f"Reconstructed metadata for {len(reconstructed_episodes)} episodes")

        print(
            f"Resuming from episode {min(set(range(total_episodes)) - completed_episodes) if completed_episodes != set(range(total_episodes)) else total_episodes}"
        )

    # Filter out completed episodes
    remaining_dirs = []
    remaining_indices = []

    for idx, episode_dir in enumerate(episode_dirs):
        if idx not in completed_episodes:
            remaining_dirs.append(episode_dir)
            remaining_indices.append(idx)

    return remaining_dirs, remaining_indices


def main(cfg: YAMSConfig):
    """Main function to convert YAMS data to LeRobot format."""

    print("=== Direct YAMS to LeRobot Converter ===")

    if cfg.action_space == "abs_cartesian":
        try:
            from visualization.yam_base import YAMSBaseInterface
        except Exception as e:
            print(f"Error importing YAMSBaseInterface: {e}, cartesian action space will not be available")

        # kinematics using pyroki (made available in the case of cartesian action space)
        cfg.robot = YAMSBaseInterface(minimal=True)  # YAMSBaseInterface object
    else:
        pass

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
            base_dir / "meta" / "tasks.jsonl",
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
        with open(base_dir / "meta" / "info.json") as f:
            info = json.load(f)

        print("📊 Dataset Info:")
        print(f"  Repository: {cfg.repo_name}")
        print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
        print(f"  Total frames: {info.get('total_frames', 'unknown')}")
        print(f"  Dataset path: {base_dir}")

        # Perform hub push
        try:
            from huggingface_hub import HfApi
            from huggingface_hub import whoami

            # Check authentication
            user_info = whoami()
            print(f"✅ Authenticated as: {user_info['name']}")

            # Create repository if it doesn't exist
            api = HfApi()
            print(f"🏗️  Ensuring repository exists: {cfg.repo_name}")
            repo_url = api.create_repo(repo_id=cfg.repo_name, repo_type="dataset", private=True, exist_ok=True)
            print(f"✅ Repository ready: {repo_url}")

            # Create version tag
            try:
                api.create_tag(repo_id=cfg.repo_name, tag="v2.1", repo_type="dataset")
                print("✅ Version tag created: v2.1")
            except Exception as tag_error:
                print(f"⚠️  Version tag creation failed (may already exist): {tag_error}")

            # Instantiate LeRobotDataset and push
            dataset = LeRobotDataset(repo_id=cfg.repo_name, root=base_dir)
            print(f"✅ LeRobotDataset loaded with {len(dataset)} frames")

            print(f"🚀 Pushing dataset to Hugging Face Hub: {cfg.repo_name}")
            dataset.push_to_hub(
                tags=["yams", "bimanual", "manipulation", "robotics"],
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

    # Handle both single path and list of paths for display
    if isinstance(cfg.yam_data_path, list):
        print("Input paths:")
        for i, path in enumerate(cfg.yam_data_path, 1):
            print(f"  {i}. {path}")
    else:
        print(f"Input path: {cfg.yam_data_path}")

    print(f"Output path: {HF_LEROBOT_HOME/cfg.repo_name}")
    print(f"Repository name: {cfg.repo_name}")
    print(f"Skip videos: {cfg.skip_videos}")
    print(f"Max episodes: {cfg.max_episodes or 'unlimited'}")
    print(f"Max workers: {cfg.max_workers}")

    # Find episodes - handle both single path and list of paths
    if isinstance(cfg.yam_data_path, list):
        input_paths = [Path(path) for path in cfg.yam_data_path]
    else:
        input_paths = Path(cfg.yam_data_path)

    episode_dirs = find_episode_directories(input_paths)
    if cfg.max_episodes:
        episode_dirs = episode_dirs[: cfg.max_episodes]

    print(f"Found {len(episode_dirs)} episodes to process")

    if not episode_dirs:
        print("No episodes found!")
        return

    # Auto-disable hub push for large datasets to avoid rate limiting
    auto_disabled_push = False
    if len(episode_dirs) > 500 and cfg.push_to_hub:
        print(f"⚠️  Large dataset detected ({len(episode_dirs)} episodes > 500)")
        print("   Automatically disabling push_to_hub to avoid HuggingFace rate limiting")
        print("   Manual upload instructions will be provided at the end")
        cfg.push_to_hub = False
        auto_disabled_push = True

    # Prepare folders - include repo_name in path structure
    base_dir = HF_LEROBOT_HOME / cfg.repo_name

    # Check for resume capability
    resume_mode = False
    if base_dir.exists() and not cfg.overwrite_existing_data:
        print("Dataset directory already exists - checking for resume capability...")
        remaining_dirs, remaining_indices = filter_episodes_for_resume(episode_dirs, base_dir, cfg.chunk_size)

        if len(remaining_dirs) < len(episode_dirs):
            resume_mode = True
            print(f"Resume mode: {len(episode_dirs) - len(remaining_dirs)} episodes already completed")
            print(f"Will process {len(remaining_dirs)} remaining episodes")
            episode_dirs = remaining_dirs
            episode_indices = remaining_indices
        else:
            print("No completed episodes found - starting fresh")
            import shutil

            shutil.rmtree(base_dir)
            episode_indices = list(range(len(episode_dirs)))
    else:
        episode_indices = list(range(len(episode_dirs)))

    if not resume_mode:
        # Only create/clear directories if not in resume mode
        (base_dir / "data").mkdir(parents=True, exist_ok=True)
        (base_dir / "meta").mkdir(exist_ok=True)
        if not cfg.skip_videos:
            (base_dir / "videos").mkdir(exist_ok=True)
    else:
        # Ensure directories exist for resume mode
        (base_dir / "data").mkdir(parents=True, exist_ok=True)
        (base_dir / "meta").mkdir(exist_ok=True)
        if not cfg.skip_videos:
            (base_dir / "videos").mkdir(exist_ok=True)

    # Create chunk directories
    num_chunks = (len(episode_dirs) + cfg.chunk_size - 1) // cfg.chunk_size
    episode_base = base_dir / "data"
    for i in range(num_chunks):
        (episode_base / f"chunk-{i:03d}").mkdir(parents=True, exist_ok=True)

    # We'll create tasks.jsonl after processing episodes to get actual task names

    # Process episodes
    all_episodes = []

    # print(f"\nProcessing {len(episode_dirs)} episodes...")

    if cfg.max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = []
            for i, (episode_idx, episode_path) in enumerate(zip(episode_indices, episode_dirs)):
                chunk_id = episode_idx // cfg.chunk_size
                futures.append(
                    executor.submit(
                        process_yam_episode,
                        episode_idx,  # Use original episode index
                        episode_path,
                        cfg,
                        episode_base / f"chunk-{chunk_id:03d}",
                        base_dir,
                        cfg.rotate_left_right_image,
                    )
                )

            for f in tqdm(futures, desc="Processing episodes"):
                result = f.result()
                if result is not None:
                    all_episodes.append(result)
    else:
        # Sequential processing (for debugging)
        for episode_idx, episode_path in zip(episode_indices, tqdm(episode_dirs, desc="Processing episodes")):
            chunk_id = episode_idx // cfg.chunk_size
            result = process_yam_episode(
                episode_idx,  # Use original episode index
                episode_path,
                cfg,
                episode_base / f"chunk-{chunk_id:03d}",
                base_dir,
                cfg.rotate_left_right_image,
            )
            if result is not None:
                all_episodes.append(result)

    print(f"Successfully processed {len(all_episodes)} episodes")

    if not all_episodes:
        print("No new episodes were processed!")
        # Still need to update info.json and complete if we're in resume mode
        if resume_mode:
            print("Updating dataset info for resume completion...")
        else:
            return

    # For resume mode or if we have processed episodes, read all metadata to get totals
    print("Reading all metadata to calculate final statistics...")

    # Read all episodes from episodes.jsonl
    all_combined_episodes = []
    episodes_file = base_dir / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        with open(episodes_file) as f:
            for line in f:
                all_combined_episodes.append(json.loads(line.strip()))

    # Read all tasks from tasks.jsonl
    all_tasks = {}
    tasks_file = base_dir / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file) as f:
            for line in f:
                task_data = json.loads(line.strip())
                all_tasks[task_data["task_index"]] = task_data["task"]

    # Sort episodes by episode_index for consistency
    all_combined_episodes.sort(key=lambda x: x["episode_index"])

    print(f"Dataset contains {len(all_combined_episodes)} total episodes")
    print(f"Dataset contains {len(all_tasks)} unique tasks")

    # Calculate final dataset statistics
    total_frames = sum(e["length"] for e in all_combined_episodes)
    actual_chunks = (len(all_combined_episodes) + cfg.chunk_size - 1) // cfg.chunk_size

    # Write info.json
    if cfg.action_space == "abs_joint":
        if cfg.single_arm:
            state_dim = 7
            action_dim = 7
        else:
            state_dim = 14
            action_dim = 14
    elif cfg.action_space == "abs_cartesian":
        assert not cfg.single_arm
        state_dim = 20
        action_dim = 20
    else:
        raise ValueError(f"Invalid action space: {cfg.action_space}, or not implemented yet")

    features = {
        "state": {
            "dtype": "float32",
            "shape": [state_dim],  # YAMS joint state dimension (single-arm: 7, bimanual: 14)
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": [action_dim],  # YAMS action dimension (single-arm: 7, bimanual: 14)
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
            features[cam_key] = {
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
                    "has_audio": False,
                },
            }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "yams",
        "total_episodes": len(all_combined_episodes),
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(cfg.camera_keys) * len(all_combined_episodes) if not cfg.skip_videos else 0,
        "total_chunks": actual_chunks,
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps / cfg.temporal_subsample_factor,
        "splits": {"train": f"0:{len(all_combined_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    with open(base_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n=== Conversion Complete ===")
    print(f"Dataset saved to: {HF_LEROBOT_HOME/cfg.repo_name}")
    if resume_mode:
        print(f"Processed {len(all_episodes)} new episodes")
        print(f"Total episodes in dataset: {len(all_combined_episodes)}")
    else:
        print(f"Total episodes: {len(all_combined_episodes)}")
    print(f"Total frames: {total_frames}")
    print(f"Total chunks: {actual_chunks}")
    
    # Print summary of problematic episodes
    print_problematic_episodes_summary(cfg)
    
    if not cfg.use_hugging_face:
        exit()

    # Create repository if it doesn't exist
    try:
        from huggingface_hub import HfApi
        from huggingface_hub import whoami

        # Check authentication
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")

        # Create repository (will not fail if it already exists)
        api = HfApi()
        print(f"🏗️  Ensuring repository exists: {cfg.repo_name}")
        repo_url = api.create_repo(
            repo_id=cfg.repo_name,
            repo_type="dataset",
            private=True,  # Make it private
            exist_ok=True,  # Won't fail if repo already exists
        )
        print(f"✅ Repository ready: {repo_url}")

        # Create version tag required by LeRobot
        try:
            api.create_tag(
                repo_id=cfg.repo_name,
                tag="v2.1",  # Match the codebase_version in info.json
                repo_type="dataset",
            )
            print("✅ Version tag created: v2.1")
        except Exception as tag_error:
            print(f"⚠️  Version tag creation failed (may already exist): {tag_error}")

    except Exception as e:
        print(f"❌ Failed to create/verify repository: {e}")
        print("Cannot proceed with hub push without repository access.")
        return

    # Provide manual upload instructions if push was auto-disabled
    if auto_disabled_push:
        print("\n🚀 MANUAL UPLOAD REQUIRED (Large Dataset)")
        print("=" * 60)
        print(f"Due to the large dataset size ({len(all_combined_episodes)} episodes),")
        print("automatic hub push was disabled to avoid HuggingFace rate limiting.")
        print()
        print("To upload your dataset manually, run:")
        print()
        print(f"  huggingface-cli upload-large-folder {cfg.repo_name} {base_dir} --repo-type=dataset")
        print()
        print("This will upload the dataset in chunks and handle rate limiting automatically.")
        print("=" * 60)

    # Push to hub if enabled
    if cfg.push_to_hub and HAS_LEROBOT:
        print("\nPreparing to push dataset to Hugging Face Hub...")

        # Extract repo name from full repo_id (e.g., "uynitsuj/yam_bimanual_load_dishes" -> "yam_bimanual_load_dishes")
        repo_name_only = cfg.repo_name.split("/")[-1]
        dataset_root = base_dir  # This points to the actual dataset directory

        print(f"Dataset root: {dataset_root}")
        print(f"Repository ID: {cfg.repo_name}")

        # Verify dataset structure exists
        required_files = [
            dataset_root / "meta" / "info.json",
            dataset_root / "meta" / "episodes.jsonl",
            dataset_root / "meta" / "tasks.jsonl",
        ]

        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("Cannot push incomplete dataset to hub.")
        else:
            try:
                # Instantiate LeRobotDataset from the correct dataset root directory
                dataset = LeRobotDataset(repo_id=cfg.repo_name, root=dataset_root)
                print(f"✅ LeRobotDataset created successfully with {len(dataset)} frames")

                print(f"🚀 Pushing dataset to Hugging Face Hub: {cfg.repo_name}")
                dataset.push_to_hub(
                    tags=["yams", "bimanual", "manipulation", "robotics"],
                    private=True,  # Repository was created as private
                    push_videos=not cfg.skip_videos,
                    license="apache-2.0",
                )
                print(f"✅ Dataset successfully pushed to hub: {cfg.repo_name}")
                print(f"🔗 View at: https://huggingface.co/datasets/{cfg.repo_name}")

            except Exception as e:
                print(f"❌ Failed to push to hub: {e}")
                print("Dataset was created successfully locally, but hub push failed.")
                print("You can manually push later with:")
                print(f"  dataset = LeRobotDataset(repo_id='{cfg.repo_name}', root='{dataset_root}')")
                print("  dataset.push_to_hub()")

    elif cfg.push_to_hub and not HAS_LEROBOT:
        print("❌ Cannot push to hub: LeRobot not available")
        print("Install lerobot package to enable hub push functionality")


def print_problematic_episodes_summary(cfg: YAMSConfig):
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
    cfg = tyro.cli(YAMSConfig)
    main(cfg)
