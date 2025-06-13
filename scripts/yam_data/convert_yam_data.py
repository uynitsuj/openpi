#!/usr/bin/env python3
"""
Direct YAMS to LeRobot format converter.

This script bypasses the LeRobot dataset creation completely and directly creates
the dataset in the same format as LeRobot, avoiding memory accumulation and 
ffmpeg-python import issues.
"""

import json
import multiprocessing
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Optional
import numpy as np
import tyro
import gc

@dataclass
class YAMSConfig:
    yam_data_path: str = "/home/justinyu/nfs_us/nfs/data/sz_05/20250425"
    output_dir: Path = Path("/home/justinyu/nfs_us/justinyu/yam_lerobot_datasets")
    repo_name: str = "uynitsuj/yam_bimanual_load_dishes"
    language_instruction: str = "Perform bimanual manipulation task" # Default task name; gets overwritten by task name in metadata
    
    # YAMS camera keys
    camera_keys: List[str] = field(default_factory=lambda: [
        "left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"
    ])
    
    resize_size: int = 224
    fps: int = 30
    chunk_size: int = 1000
    max_workers: int = 8 # Set lower on machines with less memory
    filter_quality: bool = True
    max_episodes: Optional[int] = None
    skip_videos: bool = False
    push_to_hub: bool = True
    
    # Video encoding settings
    benchmark_encoders: bool = True  # Benchmark encoders on first episode
    encoder_name: Optional[str] = None  # Force specific encoder, or None for auto-selection
    encoding_quality: str = 'fastest'  # 'fastest' or 'fast'


# Import utility modules
try:
    from .data_utils import (
        find_episode_directories, 
        is_episode_good_quality,
        load_yams_episode_data_fast,
        process_joint_data,
        calculate_actions
    )
except ImportError:
    from data_utils import (
        find_episode_directories, 
        is_episode_good_quality,
        load_yams_episode_data_fast,
        process_joint_data,
        calculate_actions
    )

def extract_task_name_from_episode(episode_data: dict, episode_path: Path) -> str:
    """Extract task name from episode metadata or path."""
    # First try to get from metadata
    if 'metadata' in episode_data and episode_data['metadata']:
        metadata = episode_data['metadata']
        
        # Common metadata fields that might contain task info
        possible_task_fields = ['task', 'task_name', 'language_instruction', 'instruction', 'description', 'task_description']
        
        for field in possible_task_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field])
    
    # Fallback: try to extract from episode directory name
    episode_name = episode_path.name
    
    # If episode name contains identifiable task keywords, extract them
    task_keywords = {
        'load_dishes': 'Load dishes into dishwasher',
        'load_dishwasher': 'Load dishes into dishwasher',
        'dishes': 'Load dishes into dishwasher',
        'dishwasher': 'Load dishes into dishwasher',
        'bimanual': 'Perform bimanual manipulation task',
        'manipulation': 'Perform manipulation task',
        'pick_place': 'Pick and place objects',
        'sorting': 'Sort objects',
        'stacking': 'Stack objects',
        'cleaning': 'Clean surfaces',
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
    HAS_LEROBOT = True
except ImportError:
    print("Warning: LeRobot not available. Hub push functionality disabled.")
    HAS_LEROBOT = False

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


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


def get_encoder_settings(encoder_name: str, quality: str = 'fast'):
    """Get optimized settings for different encoders."""
    
    settings = {
        'common': {
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'  # Enable fast start for web playback
        }
    }
    
    if 'nvenc' in encoder_name:
        # NVIDIA NVENC settings
        if quality == 'fastest':
            settings.update({
                'preset': 'p1',      # Fastest preset
                'tune': 'ull',       # Ultra-low latency
                'rc': 'vbr',         # Variable bitrate
                'cq': '28',          # Quality (lower = better, 18-28 typical)
                'b:v': '3M',         # Target bitrate
                'maxrate': '6M',     # Max bitrate
                'bufsize': '6M',     # Buffer size
                'gpu': '0'           # GPU index
            })
        else:  # 'fast'
            settings.update({
                'preset': 'p4',      # Faster preset
                'tune': 'hq',        # High quality
                'rc': 'vbr',
                'cq': '23',
                'b:v': '5M',
                'maxrate': '10M',
                'bufsize': '10M',
                'gpu': '0'
            })
    
    elif 'qsv' in encoder_name:
        # Intel Quick Sync settings
        if quality == 'fastest':
            settings.update({
                'preset': 'veryfast',
                'global_quality': '28',
                'look_ahead': '0',
                'b:v': '3M'
            })
        else:  # 'fast'
            settings.update({
                'preset': 'fast',
                'global_quality': '23',
                'look_ahead': '1',
                'b:v': '5M'
            })
    
    elif 'amf' in encoder_name:
        # AMD VCE settings
        if quality == 'fastest':
            settings.update({
                'quality': 'speed',
                'rc': 'vbr_peak',
                'qp_i': '28',
                'qp_p': '30',
                'b:v': '3M'
            })
        else:  # 'fast'
            settings.update({
                'quality': 'balanced',
                'rc': 'vbr_peak',
                'qp_i': '22',
                'qp_p': '24',
                'b:v': '5M'
            })
    
    elif 'videotoolbox' in encoder_name:
        # Apple VideoToolbox settings
        if quality == 'fastest':
            settings.update({
                'q:v': '65',         # Quality (0-100, higher = better)
                'realtime': '1',     # Real-time encoding
                'b:v': '3M'
            })
        else:  # 'fast'
            settings.update({
                'q:v': '55',
                'b:v': '5M'
            })
    
    else:
        # Software encoder fallback (libx264/libx265)
        if quality == 'fastest':
            settings.update({
                'preset': 'ultrafast',
                'crf': '28',
                'tune': 'fastdecode'
            })
        else:  # 'fast'
            settings.update({
                'preset': 'veryfast',
                'crf': '23'
            })
    
    return settings


def benchmark_encoder(encoder_name: str, test_frames: List[np.ndarray], fps: int):
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
                    '-framerate', str(fps),
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


def select_best_encoder(test_frames: List[np.ndarray] = None, fps: int = 30):
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
            encode_time = benchmark_encoder(encoder_name, test_subset, fps)
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


def encode_video_optimized(frames: List[np.ndarray], save_path: Path, fps: int, 
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
        # Build optimized ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', temp_path,
            '-vcodec', encoder_name,
        ]
        
        # Add encoder-specific settings
        for key, value in settings.items():
            if key not in ['common']:
                cmd.extend([f'-{key}', str(value)])
        
        # Add common settings
        cmd.extend(['-r', str(fps)])
        cmd.append(str(save_path))
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
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


def encode_video_simple(frames: List[np.ndarray], save_path: Path, fps: int):
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
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', temp_path,
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
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


def process_yam_episode(
    idx: int, episode_path: Path, language_instruction: str, cfg: YAMSConfig, episode_base: Path,
    base_dir: Path, encoder_name: str = None, encoding_quality: str = 'fast'
):
    """Process a single YAM episode and save it directly to LeRobot format."""
    
    # Quality filtering
    if cfg.filter_quality and not is_episode_good_quality(episode_path):
        return None
    
    # Load episode data
    episode_data = load_yams_episode_data_fast(episode_path)
    if not episode_data:
        return None
    
    # Extract task name from episode metadata instead of using hardcoded value
    task_name = extract_task_name_from_episode(episode_data, episode_path)
    
    # Process joint data
    full_joint_state = process_joint_data(episode_data['joint_data'])
    if full_joint_state is None:
        return None
    
    # Determine sequence length
    seq_length = len(full_joint_state) - 1  # -1 because we need next state for actions
    if seq_length <= 0:
        return None
    
    # Calculate actions
    joint_states, joint_actions = calculate_actions(full_joint_state, seq_length)
    
    # Process images if not skipping videos
    image_data = {}
    if not cfg.skip_videos and 'images' in episode_data:
        for cam_key in cfg.camera_keys:
            if cam_key in episode_data['images']:
                images = episode_data['images'][cam_key]
                
                if len(images) > seq_length:
                    images = images[:seq_length]
                elif len(images) < seq_length:
                    # Repeat last frame if needed
                    last_image = images[-1] if images else np.zeros((cfg.resize_size, cfg.resize_size, 3), dtype=np.uint8)
                    images.extend([last_image] * (seq_length - len(images)))
                
                # Resize images
                resized_images = []
                for img in images:
                    if isinstance(img, np.ndarray):
                        resized_img = resize_with_pad(img, cfg.resize_size, cfg.resize_size)
                        resized_images.append(convert_to_uint8(resized_img))
                
                image_data[cam_key] = resized_images
    
    # Save parquet (joint positions + actions per frame)
    records = []
    for step in range(seq_length):
        joint_pos = joint_states[step]
        action = joint_actions[step]
        
        record = {
            "state": joint_pos.tolist(),
            "actions": action.tolist(),
            "timestamp": [step / cfg.fps],
            "frame_index": [step],
            "episode_index": [idx],
            "index": [step],
            "task_index": [0],
        }
        records.append(record)
    
    episode_path_out = episode_base / f"episode_{idx:06d}.parquet"
    pd.DataFrame(records).to_parquet(episode_path_out)
    
    # Save videos if not skipping
    if not cfg.skip_videos and image_data:
        chunk_id = idx // cfg.chunk_size
        for cam_key in cfg.camera_keys:
            if cam_key in image_data:
                video_dir = base_dir / "videos" / f"chunk-{chunk_id:03d}" / cam_key
                video_dir.mkdir(parents=True, exist_ok=True)
                save_path = video_dir / f"episode_{idx:06d}.mp4"
                
                frames = image_data[cam_key]
                if frames:
                    if encoder_name:
                        encode_video_optimized(frames, save_path, cfg.fps, encoder_name, encoding_quality)
                    else:
                        encode_video_simple(frames, save_path, cfg.fps)
    
    # Clean up memory
    del episode_data, full_joint_state, joint_states, joint_actions, image_data
    gc.collect()
    
    # Return metadata for episode
    return {
        "episode_index": idx,
        "tasks": [task_name],
        "length": seq_length,
    }


def main(cfg: YAMSConfig):
    """Main function to convert YAMS data to LeRobot format."""
    
    print("=== Direct YAMS to LeRobot Converter ===")
    print(f"Input path: {cfg.yam_data_path}")
    print(f"Output path: {cfg.output_dir}")
    print(f"Repository name: {cfg.repo_name}")
    print(f"Skip videos: {cfg.skip_videos}")
    print(f"Max episodes: {cfg.max_episodes or 'unlimited'}")
    print(f"Max workers: {cfg.max_workers}")
    
    # Find episodes
    episode_dirs = find_episode_directories(Path(cfg.yam_data_path))
    if cfg.max_episodes:
        episode_dirs = episode_dirs[:cfg.max_episodes]
    
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if not episode_dirs:
        print("No episodes found!")
        return
    
    # Prepare folders - include repo_name in path structure
    base_dir = cfg.output_dir / cfg.repo_name
    if base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)
    
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
    
    # Select best encoder for video encoding
    best_encoder = None
    encoding_quality = cfg.encoding_quality
    
    if not cfg.skip_videos:
        print(f"\n=== Video Encoder Setup ===")
        if cfg.encoder_name:
            best_encoder = cfg.encoder_name
            print(f"Using forced encoder: {best_encoder}")
        else:
            # Auto-detect best encoder
            if cfg.benchmark_encoders and len(episode_dirs) > 0:
                print("Loading first episode for encoder benchmarking...")
                # Load first episode to get sample frames for benchmarking
                first_episode_data = load_yams_episode_data_fast(episode_dirs[0])
                if first_episode_data and 'images' in first_episode_data:
                    sample_frames = []
                    for cam_key in cfg.camera_keys:
                        if cam_key in first_episode_data['images']:
                            images = first_episode_data['images'][cam_key][:10]  # First 10 frames
                            for img in images:
                                if isinstance(img, np.ndarray):
                                    resized_img = resize_with_pad(img, cfg.resize_size, cfg.resize_size)
                                    sample_frames.append(convert_to_uint8(resized_img))
                            break  # Use first available camera for benchmarking
                    
                    if sample_frames:
                        best_encoder, encoding_quality = select_best_encoder(sample_frames, cfg.fps)
                    else:
                        best_encoder, encoding_quality = select_best_encoder()
                else:
                    best_encoder, encoding_quality = select_best_encoder()
            else:
                best_encoder, encoding_quality = select_best_encoder()
        
        print(f"Using encoder: {best_encoder} with {encoding_quality} quality")
    
    # Process episodes
    all_episodes = []
    
    print(f"\nProcessing {len(episode_dirs)} episodes...")
    
    if cfg.max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = []
            for idx, episode_path in enumerate(episode_dirs):
                chunk_id = idx // cfg.chunk_size
                futures.append(
                    executor.submit(
                        process_yam_episode,
                        idx,
                        episode_path,
                        cfg.language_instruction,
                        cfg,
                        episode_base / f"chunk-{chunk_id:03d}",
                        base_dir,
                        best_encoder,
                        encoding_quality
                    )
                )
            
            for f in tqdm(futures, desc="Processing episodes"):
                result = f.result()
                if result is not None:
                    all_episodes.append(result)
    else:
        # Sequential processing (for debugging)
        for idx, episode_path in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
            chunk_id = idx // cfg.chunk_size
            result = process_yam_episode(
                idx,
                episode_path, 
                cfg.language_instruction,
                cfg,
                episode_base / f"chunk-{chunk_id:03d}",
                base_dir,
                best_encoder,
                encoding_quality
            )
            if result is not None:
                all_episodes.append(result)
    
    print(f"Successfully processed {len(all_episodes)} episodes")
    
    if not all_episodes:
        print("No episodes were successfully processed!")
        return
    
    # Collect unique task names and create task mapping
    unique_tasks = set()
    for episode in all_episodes:
        unique_tasks.update(episode['tasks'])
    
    unique_tasks = sorted(list(unique_tasks))  # Sort for consistency
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}
    
    print(f"Found {len(unique_tasks)} unique tasks:")
    for idx, task in enumerate(unique_tasks):
        print(f"  {idx}: {task}")
    
    # Write tasks.jsonl with actual task names
    with open(base_dir / "meta" / "tasks.jsonl", "w") as f:
        for idx, task in enumerate(unique_tasks):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    
    # Update episodes with correct task indices and write episodes.jsonl
    with open(base_dir / "meta" / "episodes.jsonl", "w") as f:
        for epi in all_episodes:
            # Convert task names to task indices
            task_indices = [task_to_index[task] for task in epi['tasks']]
            epi_updated = epi.copy()
            epi_updated['task_index'] = task_indices[0] if task_indices else 0  # Use first task index
            f.write(json.dumps(epi_updated) + "\n")
    
    # Calculate dataset statistics
    total_frames = sum(e["length"] for e in all_episodes)
    actual_chunks = (len(all_episodes) + cfg.chunk_size - 1) // cfg.chunk_size
    
    # Write info.json
    features = {
        "state": {
            "dtype": "float32", 
            "shape": [14],  # YAMS joint state dimension (6 joints + 1 gripper per arm × 2 arms)
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": [14],  # YAMS action dimension (6 joints + 1 gripper per arm × 2 arms)
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
                    "video.fps": cfg.fps,
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
        "codebase_version": "v2.0",
        "robot_type": "yams",
        "total_episodes": len(all_episodes),
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "total_videos": len(cfg.camera_keys) * len(all_episodes) if not cfg.skip_videos else 0,
        "total_chunks": actual_chunks,
        "chunks_size": cfg.chunk_size,
        "fps": cfg.fps,
        "splits": {"train": f"0:{len(all_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features
    }
    
    with open(base_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Dataset saved to: {cfg.output_dir}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total frames: {total_frames}")
    print(f"Total chunks: {actual_chunks}")

    # Push to hub if enabled
    if cfg.push_to_hub and HAS_LEROBOT:
        print(f"\nPreparing to push dataset to Hugging Face Hub...")
        
        # Extract repo name from full repo_id (e.g., "uynitsuj/yam_bimanual_load_dishes" -> "yam_bimanual_load_dishes")
        repo_name_only = cfg.repo_name.split("/")[-1]
        dataset_root = base_dir  # This points to the actual dataset directory
        
        print(f"Dataset root: {dataset_root}")
        print(f"Repository ID: {cfg.repo_name}")
        
        # Create repository if it doesn't exist
        try:
            from huggingface_hub import HfApi, whoami
            
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
                exist_ok=True  # Won't fail if repo already exists
            )
            print(f"✅ Repository ready: {repo_url}")
            
            # Create version tag required by LeRobot
            try:
                api.create_tag(
                    repo_id=cfg.repo_name,
                    tag="v2.0",  # Match the codebase_version in info.json
                    repo_type="dataset"
                )
                print(f"✅ Version tag created: v2.0")
            except Exception as tag_error:
                print(f"⚠️  Version tag creation failed (may already exist): {tag_error}")
            
        except Exception as e:
            print(f"❌ Failed to create/verify repository: {e}")
            print("Cannot proceed with hub push without repository access.")
            return
        
        # Verify dataset structure exists
        required_files = [
            dataset_root / "meta" / "info.json",
            dataset_root / "meta" / "episodes.jsonl", 
            dataset_root / "meta" / "tasks.jsonl"
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
                print(f"You can manually push later with:")
                print(f"  dataset = LeRobotDataset(repo_id='{cfg.repo_name}', root='{dataset_root}')")
                print(f"  dataset.push_to_hub()")
                
    elif cfg.push_to_hub and not HAS_LEROBOT:
        print("❌ Cannot push to hub: LeRobot not available")
        print("Install lerobot package to enable hub push functionality")


if __name__ == "__main__":
    cfg = tyro.cli(YAMSConfig)
    main(cfg)
