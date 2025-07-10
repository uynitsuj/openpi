"""
Minimal example script for converting YAMS dataset to LeRobot format using standard LeRobot dataset creation. (SLOWER)

This script uses LeRobot's built-in dataset creation and frame addition methods,
letting LeRobot handle video encoding and dataset management.

Usage:
python convert_yam_data_default.py

If you want to push your dataset to the Hugging Face Hub, add --push_to_hub flag.
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


@dataclass
class YAMSConfig:
    yam_data_path: str | list[str] = field(
        default_factory=lambda: [
            "/home/justinyu/nfs_us/nfs/data/sz_05/20250416",
            "/home/justinyu/nfs_us/nfs/data/sz_05/20250425",
            "/home/justinyu/nfs_us/nfs/data/sz_04/20250415",
            "/home/justinyu/nfs_us/nfs/data/sz_04/20250412",
            "/home/justinyu/nfs_us/nfs/data/sz_04/20250411",
            "/home/justinyu/nfs_us/nfs/data/sz_04/20250410",
            "/home/justinyu/nfs_us/nfs/data/sz_03/20250423",
            "/home/justinyu/nfs_us/nfs/data/sz_03/20250417",
        ]
    )
    output_dir: Path = Path("/home/justinyu/nfs_us/justinyu/yam_lerobot_datasets")
    repo_name: str = "uynitsuj/yam_bimanual_load_dishes_large_absolute"
    language_instruction: str = (
        "Perform bimanual manipulation task"  # Default task name; gets overwritten by task name in metadata
    )

    # YAMS camera keys
    camera_keys: list[str] = field(
        default_factory=lambda: ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
    )

    resize_size: int = 224
    fps: int = 30
    max_episodes: int | None = None
    filter_quality: bool = True
    push_to_hub: bool = True


# Import utility modules
try:
    from .data_utils import calculate_actions
    from .data_utils import find_episode_directories
    from .data_utils import is_episode_good_quality
    from .data_utils import load_yams_episode_data_fast
    from .data_utils import process_joint_data
except ImportError:
    from data_utils import calculate_actions
    from data_utils import find_episode_directories
    from data_utils import is_episode_good_quality
    from data_utils import load_yams_episode_data_fast
    from data_utils import process_joint_data


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


def process_episode_data(episode_data: dict, episode_path: Path, cfg: YAMSConfig) -> tuple[list, dict, str]:
    """Process episode data and return records, image data, and task name."""

    # Extract task name from episode metadata or path
    task_name = extract_task_name_from_episode(episode_data, episode_path)

    # Process joint data
    joint_state = process_joint_data(episode_data["joint_data"])
    if joint_state is None:
        return [], {}, task_name

    # Determine sequence length
    total_length = len(joint_state) - 1  # -1 because we need next state for actions
    if total_length <= 0:
        return [], {}, task_name

    # Calculate actions
    joint_states, joint_actions = calculate_actions(joint_state, total_length)

    # Process images
    image_data = {}
    if "images" in episode_data:
        for cam_key in cfg.camera_keys:
            if cam_key in episode_data["images"]:
                images = episode_data["images"][cam_key][:total_length]  # Match joint data length

                # Resize images
                resized_images = []
                for img in images:
                    if isinstance(img, np.ndarray):
                        resized_img = resize_with_pad(img, cfg.resize_size, cfg.resize_size)
                        resized_images.append(convert_to_uint8(resized_img))

                if resized_images:
                    image_data[cam_key] = resized_images

    # Create frame records (following Libero format - no task or timestamp in frames)
    records = []
    for step in range(total_length):
        joint_pos = joint_states[step]
        action = joint_actions[step]

        record = {
            "state": joint_pos,
            "actions": action,
            "task": task_name,
        }

        # Add camera images with LeRobot naming convention
        for cam_key in cfg.camera_keys:
            if cam_key in image_data and step < len(image_data[cam_key]):
                lerobot_key = cam_key
                record[lerobot_key] = image_data[cam_key][step]

        records.append(record)

    return records, image_data, task_name


def main(cfg: YAMSConfig):
    # Clean up any existing dataset in the output directory
    output_path = cfg.output_dir / cfg.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    print("=== YAMS to LeRobot Converter (Standard Method) ===")

    # Handle both single path and list of paths for display
    if isinstance(cfg.yam_data_path, list):
        print("Input paths:")
        for i, path in enumerate(cfg.yam_data_path, 1):
            print(f"  {i}. {path}")
    else:
        print(f"Input path: {cfg.yam_data_path}")

    print(f"Output path: {cfg.output_dir}")
    print(f"Repository name: {cfg.repo_name}")
    print(f"Max episodes: {cfg.max_episodes or 'unlimited'}")

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

    # Create LeRobot dataset with proper features
    # Note: LeRobot expects specific naming conventions
    features = {
        "state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["actions"],
        },
    }

    # Add camera features with proper LeRobot naming convention
    for cam_key in cfg.camera_keys:
        # Convert YAMS camera key to LeRobot format
        lerobot_key = cam_key
        features[lerobot_key] = {
            "dtype": "image",
            "shape": (cfg.resize_size, cfg.resize_size, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_name,
        robot_type="yams_bimanual",
        fps=cfg.fps,
        features=features,
        root=cfg.output_dir / cfg.repo_name,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Created dataset with {len(features)} features")
    print(f"Camera keys: {cfg.camera_keys}")

    # Process episodes
    successful_episodes = 0
    total_frames = 0

    for episode_idx, episode_path in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
        print(f"\nProcessing episode {episode_idx}: {episode_path.name}")

        # Quality filtering
        if cfg.filter_quality and not is_episode_good_quality(episode_path):
            print(f"  Skipping episode {episode_idx}: poor quality")
            continue

        # Load episode data
        episode_data = load_yams_episode_data_fast(episode_path)
        if not episode_data:
            print(f"  Failed to load episode {episode_idx}")
            continue

        # Extract task name from episode
        task_name = extract_task_name_from_episode(episode_data, episode_path)
        print(f"  Task: {task_name}")

        # Process episode data
        try:
            records, image_data, extracted_task = process_episode_data(episode_data, episode_path, cfg)
            if not records:
                print(f"  No valid data in episode {episode_idx}")
                continue

            print(f"  Episode {episode_idx}: {len(records)} frames")

            # Add frames to dataset
            for frame_idx, record in enumerate(records):
                try:
                    dataset.add_frame(record)
                except Exception as e:
                    print(f"    Warning: Failed to add frame {frame_idx}: {e}")
                    continue

            # Save episode with task
            try:
                dataset.save_episode(task=task_name)
            except TypeError:
                # If task parameter not supported, save without it
                dataset.save_episode()

            successful_episodes += 1
            total_frames += len(records)
            print(f"  ✅ Episode {episode_idx} completed ({len(records)} frames)")

        except Exception as e:
            print(f"  ❌ Error processing episode {episode_idx}: {e}")
            continue

    print("\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_episodes}/{len(episode_dirs)} episodes")
    print(f"Total frames: {total_frames}")

    if successful_episodes == 0:
        print("No episodes were successfully processed!")
        return

    # Consolidate the dataset
    # print("Consolidating dataset...")
    # dataset.consolidate(run_compute_stats=True)

    print(f"Dataset saved to: {output_path}")
    print(f"Dataset contains {len(dataset)} frames across {successful_episodes} episodes")

    # Optionally push to the Hugging Face Hub
    if cfg.push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {cfg.repo_name}")
        dataset.push_to_hub(
            tags=["yams", "bimanual", "manipulation", "robotics"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ Dataset successfully pushed to hub: {cfg.repo_name}")
        print(f"🔗 View at: https://huggingface.co/datasets/{cfg.repo_name}")


if __name__ == "__main__":
    cfg = tyro.cli(YAMSConfig)
    main(cfg)
