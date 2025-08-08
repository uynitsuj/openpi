"""
Data processing utilities for YAMS data conversion.

This module contains functions for loading, processing, and validating YAMS episode data.
"""

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .video_utils import extract_video_frames_fast
    from .video_utils import resize_frames_vectorized
except ImportError:
    from video_utils import extract_video_frames_fast
    from video_utils import resize_frames_vectorized

# YAMS data configuration
CAMERA_KEYS = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

CAMERA_KEY_MAPPING = {
    "left_camera-images-rgb": "exterior_image_1_left",
    "right_camera-images-rgb": "exterior_image_2_right",
    "top_camera-images-rgb": "exterior_image_3_top",
}


def find_episode_directories(parent_dir: Path | list[Path]) -> list[Path]:
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


def load_yams_episode_data_fast(episode_path: Path) -> dict | None:
    """Load data for a specific YAMS episode with maximum optimizations."""
    episode_data = {"metadata": {}, "joint_data": {}, "images": {}}

    try:
        # Load metadata first (small file)
        metadata_file = episode_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                episode_data["metadata"] = json.load(f)

        # Load joint data using memory mapping for large files
        essential_joint_files = [
            "left-joint_pos.npy",
            "right-joint_pos.npy",
            "left-gripper_pos.npy",
            "right-gripper_pos.npy",
            "action-left-pos.npy",
            "action-right-pos.npy",
        ]

        # Load all joint files at once
        joint_data = {}
        for joint_file in essential_joint_files:
            file_path = episode_path / joint_file
            if file_path.exists():
                # Use memory mapping for faster loading of large arrays
                joint_data[joint_file.replace(".npy", "")] = np.load(file_path, mmap_mode="r")

        episode_data["joint_data"] = joint_data

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
                        episode_data["images"][camera_prefix] = frames
                    break  # Found video for this camera, move to next

        # Force garbage collection after loading large video data
        gc.collect()

        return episode_data

    except Exception as e:
        print(f"Error loading YAMS episode data from {episode_path}: {e}")
        return None


def process_joint_data(joint_data: dict, single_arm: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Process and combine joint data from YAMS episode. Handles both single-arm and bimanual data."""

    # Check that we have left arm data (always required)
    if not all(key in joint_data for key in ["left-joint_pos", "left-gripper_pos"]):
        raise ValueError("Missing left arm joint data (left-joint_pos, left-gripper_pos)")

    # Get left arm data
    left_joint_pos = joint_data["left-joint_pos"]  # Shape: (N, 6)
    left_gripper_pos = joint_data["left-gripper_pos"]  # Shape: (N, 1)
    seq_length = len(left_joint_pos)

    if single_arm:
        # Single-arm case: 7-dimensional state (left arm only)
        print("  Processing single-arm data (left arm only)")
        full_joint_state = np.empty((seq_length, 7), dtype=np.float32)
        full_joint_state[:, :6] = left_joint_pos
        full_joint_state[:, 6:7] = left_gripper_pos[:, 0:1]

        # Handle actions for single-arm
        if "action-left-pos" in joint_data:
            left_action_pos = joint_data["action-left-pos"]  # Shape: (N, 7)
            return full_joint_state, left_action_pos
    else:
        # Bimanual case: check for right arm data
        if not all(key in joint_data for key in ["right-joint_pos", "right-gripper_pos"]):
            raise ValueError("Bimanual mode requires right arm joint data (right-joint_pos, right-gripper_pos)")

        print("  Processing bimanual data (left + right arms)")
        right_joint_pos = joint_data["right-joint_pos"]  # Shape: (N, 6)
        right_gripper_pos = joint_data["right-gripper_pos"]  # Shape: (N, 1)

        full_joint_state = np.empty((seq_length, 14), dtype=np.float32)
        full_joint_state[:, :6] = left_joint_pos
        full_joint_state[:, 6:7] = left_gripper_pos[:, 0:1]
        full_joint_state[:, 7:13] = right_joint_pos
        full_joint_state[:, 13:14] = right_gripper_pos[:, 0:1]

        # Handle actions for bimanual
        if "action-left-pos" in joint_data and "action-right-pos" in joint_data:
            full_joint_action = np.empty((seq_length, 14), dtype=np.float32)
            left_action_pos = joint_data["action-left-pos"]  # Shape: (N, 7)
            right_action_pos = joint_data["action-right-pos"]  # Shape: (N, 7)
            full_joint_action[:, :7] = left_action_pos
            full_joint_action[:, 7:14] = right_action_pos
            return full_joint_state, full_joint_action

    return full_joint_state, None


def process_images(images: dict, seq_length: int, resize_size: int, skip_videos: bool = False) -> dict:
    """Process and resize images from YAMS episode."""
    processed_images = {}

    if not skip_videos:
        for camera_key in CAMERA_KEYS:
            if camera_key in images:
                camera_frames = images[camera_key][:seq_length]

                # Batch resize all frames at once using fastest method
                resized_frames = resize_frames_vectorized(camera_frames, resize_size)
                processed_images[camera_key] = resized_frames

    return processed_images


def calculate_actions(full_joint_state: np.ndarray, full_joint_action: np.ndarray | None, seq_length: int):
    joint_states = full_joint_state[:seq_length]
    if full_joint_action is None:
        joint_actions = joint_states.copy()
    else:  # If no action data, use joint states as actions
        joint_actions = full_joint_action[:seq_length]

    return joint_states, joint_actions


def calculate_actions_cartesian(
    full_joint_state: np.ndarray, full_joint_action: np.ndarray | None, seq_length: int, robot: Any
):
    # Check configuration and raise error if not bimanual
    joint_state_dim = full_joint_state.shape[1]
    if joint_state_dim != 14:
        raise ValueError(
            f"Unexpected joint state dimension: {joint_state_dim}. "
            "Expected 14 for bimanual configuration (6 left joints + 1 left gripper + 6 right joints + 1 right gripper)."
        )

    if full_joint_action is not None:
        joint_action_dim = full_joint_action.shape[1]
        if joint_action_dim != 14:
            raise ValueError(
                f"Unexpected joint action dimension: {joint_action_dim}. " "Expected 14 for bimanual configuration."
            )

    from openpi.utils.matrix_utils import quat_to_rot_6d

    joint_states = full_joint_state[:seq_length]

    if full_joint_action is None:
        joint_actions = joint_states
    else:  # If no action data, use joint states as actions
        joint_actions = full_joint_action[:seq_length]

    # joint states ordering: left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos
    # T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[:, :6], joint_states[:, 7:13]) # BUGGY, INTRODUCES A POSE DELAY
    cartesian_states_list = []
    cartesian_actions_list = []

    for i in range(
        seq_length
    ):  # For some reason running batched FK is buggy -- camera lags behind pose so delay is introduced somewhere somehow
        T_left_ee_state, T_right_ee_state = robot.solve_fk_base(joint_states[i, :6], joint_states[i, 7:13])
        T_left_ee_6d_state = quat_to_rot_6d(T_left_ee_state.rotation().wxyz[None, :], scalar_first=True)
        T_right_ee_6d_state = quat_to_rot_6d(T_right_ee_state.rotation().wxyz[None, :], scalar_first=True)
        T_left_ee_pos_state = T_left_ee_state.wxyz_xyz[-3:].reshape(1, -1)
        T_right_ee_pos_state = T_right_ee_state.wxyz_xyz[-3:].reshape(1, -1)
        left_gripper_pos_state = joint_states[i, 6].reshape(-1, 1)
        right_gripper_pos_state = joint_states[i, 13].reshape(-1, 1)
        cartesian_states = np.concatenate(
            [
                T_left_ee_6d_state,
                T_left_ee_pos_state,
                left_gripper_pos_state,
                T_right_ee_6d_state,
                T_right_ee_pos_state,
                right_gripper_pos_state,
            ],
            axis=1,
        )[0]
        cartesian_states_list.append(cartesian_states)

        if full_joint_action is not None:
            T_left_ee_action, T_right_ee_action = robot.solve_fk_base(joint_actions[i, :6], joint_actions[i, 7:13])
            T_left_ee_6d_action = quat_to_rot_6d(T_left_ee_action.rotation().wxyz[None, :], scalar_first=True)
            T_right_ee_6d_action = quat_to_rot_6d(T_right_ee_action.rotation().wxyz[None, :], scalar_first=True)
            T_left_ee_pos_action = T_left_ee_action.wxyz_xyz[-3:].reshape(1, -1)
            T_right_ee_pos_action = T_right_ee_action.wxyz_xyz[-3:].reshape(1, -1)
            left_gripper_pos_action = joint_actions[i, 6].reshape(-1, 1)
            right_gripper_pos_action = joint_actions[i, 13].reshape(-1, 1)
            cartesian_actions = np.concatenate(
                [
                    T_left_ee_6d_action,
                    T_left_ee_pos_action,
                    left_gripper_pos_action,
                    T_right_ee_6d_action,
                    T_right_ee_pos_action,
                    right_gripper_pos_action,
                ],
                axis=1,
            )[0]
            cartesian_actions_list.append(cartesian_actions)
        else:
            cartesian_actions_list.append(cartesian_states)

    cartesian_states = np.stack(cartesian_states_list, axis=0)
    cartesian_actions = np.stack(cartesian_actions_list, axis=0)

    return cartesian_states, cartesian_actions


def calculate_actions_delta_cartesian(full_joint_state: np.ndarray, seq_length: int, robot: Any):
    # Check configuration and raise error if not bimanual
    joint_state_dim = full_joint_state.shape[1]
    if joint_state_dim != 14:
        raise ValueError(
            f"Unexpected joint state dimension: {joint_state_dim}. "
            "Expected 14 for bimanual configuration (6 left joints + 1 left gripper + 6 right joints + 1 right gripper). "
            "Delta cartesian action space is currently only supported for bimanual (dual arm) configurations."
        )

    from openpi.utils.matrix_utils import quat_to_rot_6d

    joint_states = full_joint_state[:seq_length]

    # joint states ordering: left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos
    # T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[:, :6], joint_states[:, 7:13]) # BUGGY, INTRODUCES A POSE DELAY
    cartesian_states_list = []
    cartesian_actions_list = []

    for i in range(
        seq_length
    ):  # For some reason running batched FK is buggy -- camera lags behind pose so delay is introduced somewhere somehow
        T_left_ee, T_right_ee = robot.solve_fk_base(joint_states[i, :6], joint_states[i, 7:13])
        T_left_ee_6d = quat_to_rot_6d(T_left_ee.rotation().wxyz[None, :], scalar_first=True)
        T_right_ee_6d = quat_to_rot_6d(T_right_ee.rotation().wxyz[None, :], scalar_first=True)
        T_left_ee_pos = T_left_ee.wxyz_xyz[-3:].reshape(1, -1)
        T_right_ee_pos = T_right_ee.wxyz_xyz[-3:].reshape(1, -1)
        left_gripper_pos = joint_states[i, 6].reshape(-1, 1)
        right_gripper_pos = joint_states[i, 13].reshape(-1, 1)
        cartesian_states = np.concatenate(
            [T_left_ee_6d, T_left_ee_pos, left_gripper_pos, T_right_ee_6d, T_right_ee_pos, right_gripper_pos], axis=1
        )[0]
        cartesian_states_list.append(cartesian_states)
    cartesian_abs_states = np.stack(cartesian_states_list, axis=0)

    for i in range(seq_length):
        cartesian_actions_t = cartesian_abs_states[i + 1] - cartesian_abs_states[i]
        # keep grippers absolute
        cartesian_actions_t[6] = cartesian_abs_states[i + 1, 6]
        cartesian_actions_t[13] = cartesian_abs_states[i + 1, 13]

        assert len(cartesian_actions_t.shape) == 1
        cartesian_actions_list.append(cartesian_actions_t)

    cartesian_delta_actions = np.stack(cartesian_actions_list, axis=0)
    return cartesian_abs_states, cartesian_delta_actions


def create_frame_data(
    joint_states: np.ndarray,
    joint_actions: np.ndarray,
    processed_images: dict,
    seq_length: int,
    skip_videos: bool = False,
) -> list[dict]:
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
