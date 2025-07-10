#!/usr/bin/env python3
"""
Visualize first frames from YAM dataset episodes.

This script takes YAM data directories and creates visualizations of the first frame
from each episode, helping to understand the dataset diversity and initial conditions.

Usage examples:
python visualize_dataset_first_frame.py --yam_data_path /path/to/yam/data --output_dir ./first_frame_viz
python visualize_dataset_first_frame.py --yam_data_path /path/to/yam/data --create_grid --grid_size 8 8
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm
import tyro


@dataclass
class FirstFrameVisualizationArgs:
    """Arguments for YAM dataset first frame visualization."""

    yam_data_path: str | list[str] = field(
        default_factory=lambda: [
            "/home/justinyu/nfs_us/philipp/internal_justin/061825_annotated_dishes/unload_dishes_from_tabletop_dish_rack"
        ]
    )
    """Path(s) to YAM data directories containing episodes."""

    output_dir: str = "first_frame_visualization"
    """Directory to save visualization outputs."""

    camera_key: str = "top_camera-images-rgb"
    """Camera key to use for visualization."""

    max_episodes: int | None = None
    """Maximum number of episodes to process. If None, process all episodes."""

    resize_size: int = 480
    """Size to resize images to for visualization."""

    # Grid visualization options
    create_grid: bool = False
    """Whether to create a grid visualization of first frames."""

    grid_size: tuple[int, int] = (8, 8)
    """Grid dimensions (rows, cols) for the grid visualization."""

    grid_output_path: str = "first_frames_grid.jpg"
    """Filename for the grid visualization output."""

    # Video visualization options
    create_video: bool = True
    """Whether to create a video of first frames."""

    video_output_path: str = "first_frames_sequence.mp4"
    """Filename for the video visualization output."""

    video_fps: int = 20
    """FPS for the first frames video (slower to see each frame)."""

    # Individual frames options
    save_individual_frames: bool = False
    """Whether to save individual first frames."""

    individual_frames_dir: str = "individual_first_frames"
    """Directory name for individual frame outputs."""

    # Sampling options
    random_sample: bool = False
    """Whether to randomly sample episodes instead of taking first N."""

    random_seed: int = 42
    """Random seed for reproducible sampling."""

    # Quality filtering
    filter_quality: bool = False
    """Whether to filter out low quality episodes."""


# Simple episode directory finding function
def find_episode_directories(input_paths: list[Path]) -> list[Path]:
    """Find all episode directories in the given paths."""
    episode_dirs = []
    for input_path in input_paths:
        if input_path.is_dir():
            # Look for directories that look like episodes (contain camera MP4 files)
            for item in input_path.iterdir():
                if item.is_dir() and item.name.startswith("episode_"):
                    # Check if it has camera MP4 files
                    has_camera = any(
                        (item / f"{cam}.mp4").exists()
                        for cam in ["top_camera-images-rgb", "left_camera-images-rgb", "right_camera-images-rgb"]
                    )
                    if has_camera:
                        episode_dirs.append(item)
    return sorted(episode_dirs)


def is_episode_good_quality(episode_path: Path) -> bool:
    """Simplified quality check - just check if directory exists and has camera files."""
    camera_file = episode_path / "top_camera-images-rgb.mp4"
    return camera_file.exists()


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image with padding to maintain aspect ratio."""
    if len(image.shape) == 2:
        # Grayscale image
        h, w = image.shape
    else:
        # Color image
        h, w = image.shape[:2]

    # Calculate scaling factor
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded image
    if len(image.shape) == 2:
        padded = np.zeros((height, width), dtype=image.dtype)
    else:
        padded = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    # Calculate padding offsets
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2

    # Place resized image in center
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return padded


def load_first_frame(episode_path: Path, camera_key: str, resize_size: int) -> np.ndarray | None:
    """Load and resize the first frame from an episode MP4 file - FAST version."""
    try:
        # Get the camera MP4 file path
        camera_mp4_path = episode_path / f"{camera_key}.mp4"
        if not camera_mp4_path.exists():
            return None

        # Open video file and read first frame
        cap = cv2.VideoCapture(str(camera_mp4_path))
        if not cap.isOpened():
            return None

        # Read the first frame
        ret, first_frame = cap.read()
        cap.release()  # Important: release the video capture object

        if not ret or first_frame is None:
            return None

        # Resize with padding
        resized_frame = resize_with_pad(first_frame, resize_size, resize_size)

        return resized_frame

    except Exception:
        # Suppress verbose errors - uncomment for debugging
        # print(f"Error loading first frame from {episode_path}: {e}")
        return None


def create_grid_visualization(frames: list[np.ndarray], grid_size: tuple[int, int], output_path: Path) -> None:
    """Create a grid visualization of first frames."""
    rows, cols = grid_size
    total_slots = rows * cols

    if len(frames) > total_slots:
        print(f"Warning: {len(frames)} frames but only {total_slots} grid slots. Using first {total_slots} frames.")
        frames = frames[:total_slots]

    # Get frame dimensions
    if not frames:
        print("No frames to create grid visualization")
        return

    frame_h, frame_w = frames[0].shape[:2]
    has_color = len(frames[0].shape) == 3

    # Create grid image
    if has_color:
        grid_img = np.zeros((rows * frame_h, cols * frame_w, 3), dtype=np.uint8)
    else:
        grid_img = np.zeros((rows * frame_h, cols * frame_w), dtype=np.uint8)

    # Fill grid with frames
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols

        y_start, y_end = row * frame_h, (row + 1) * frame_h
        x_start, x_end = col * frame_w, (col + 1) * frame_w

        grid_img[y_start:y_end, x_start:x_end] = frame

    # Save grid image
    cv2.imwrite(str(output_path), grid_img)
    print(f"Grid visualization saved to: {output_path}")


def create_video_visualization(frames: list[np.ndarray], fps: int, output_path: Path) -> None:
    """Create a video visualization of first frames."""
    if not frames:
        print("No frames to create video visualization")
        return

    height, width = frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=(len(frames[0].shape) == 3))

    # Write frames to video
    for frame in tqdm(frames, desc="Creating video"):
        video_writer.write(frame)

    video_writer.release()
    print(f"Video visualization saved to: {output_path}")


def save_individual_frames(frames: list[np.ndarray], episode_paths: list[Path], output_dir: Path) -> None:
    """Save individual first frames with episode names."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame, episode_path in zip(frames, episode_paths):
        frame_filename = f"{episode_path.name}_first_frame.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

    print(f"Individual frames saved to: {output_dir}")


def main(args: FirstFrameVisualizationArgs) -> None:
    """Main function to create first frame visualizations."""

    print("=== YAM Dataset First Frame Visualization ===")
    print(f"Camera key: {args.camera_key}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle both single path and list of paths
    if isinstance(args.yam_data_path, list):
        input_paths = [Path(path) for path in args.yam_data_path]
        print(f"Input paths: {args.yam_data_path}")
    else:
        input_paths = [Path(args.yam_data_path)]
        print(f"Input path: {args.yam_data_path}")

    # Find episode directories
    episode_dirs = find_episode_directories(input_paths)
    print(f"Found {len(episode_dirs)} total episodes")

    if not episode_dirs:
        print("No episodes found!")
        return

    # Filter by quality if requested
    if args.filter_quality:
        print("Filtering episodes by quality...")
        good_episodes = []
        for episode_dir in tqdm(episode_dirs, desc="Quality filtering"):
            if is_episode_good_quality(episode_dir):
                good_episodes.append(episode_dir)
        episode_dirs = good_episodes
        print(f"After quality filtering: {len(episode_dirs)} episodes")

    # Sample episodes if requested
    if args.max_episodes and len(episode_dirs) > args.max_episodes:
        if args.random_sample:
            random.seed(args.random_seed)
            episode_dirs = random.sample(episode_dirs, args.max_episodes)
            print(f"Randomly sampled {args.max_episodes} episodes")
        else:
            episode_dirs = episode_dirs[: args.max_episodes]
            print(f"Using first {args.max_episodes} episodes")

    print(f"Processing {len(episode_dirs)} episodes...")

    # Load first frames
    frames = []
    valid_episode_paths = []

    for episode_path in tqdm(episode_dirs, desc="Loading first frames"):
        frame = load_first_frame(episode_path, args.camera_key, args.resize_size)
        if frame is not None:
            frames.append(frame)
            valid_episode_paths.append(episode_path)
        # Removed the verbose error message to reduce clutter

    print(f"Successfully loaded {len(frames)} first frames")

    if not frames:
        print("No valid first frames found!")
        return

    # Create visualizations
    if args.create_grid:
        print("Creating grid visualization...")
        grid_output = output_dir / args.grid_output_path
        create_grid_visualization(frames, args.grid_size, grid_output)

    if args.create_video:
        print("Creating video visualization...")
        video_output = output_dir / args.video_output_path
        create_video_visualization(frames, args.video_fps, video_output)

    if args.save_individual_frames:
        print("Saving individual frames...")
        individual_dir = output_dir / args.individual_frames_dir
        save_individual_frames(frames, valid_episode_paths, individual_dir)

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total episodes found: {len(episode_dirs)}")
    print(f"Successfully processed: {len(frames)}")
    print(f"Success rate: {len(frames)/len(episode_dirs)*100:.1f}%")
    print(f"Frame size: {args.resize_size}x{args.resize_size}")
    print(f"Camera: {args.camera_key}")

    if args.create_grid:
        grid_frames_used = min(len(frames), args.grid_size[0] * args.grid_size[1])
        print(f"Grid visualization: {grid_frames_used}/{len(frames)} frames used")

    print(f"Output directory: {output_dir}")
    print("Visualization complete!")


if __name__ == "__main__":
    args = tyro.cli(FirstFrameVisualizationArgs)
    main(args)
