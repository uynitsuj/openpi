#!/usr/bin/env python3
"""
XMI Trajectory Viewer for LeRobot formatted datasets.

This viewer loads XMI data that has been converted to LeRobot format
and provides 3D visualization with transform frames and camera feeds.
Loads dataset files directly without using LeRobotDataset class to avoid hub issues.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tyro
import viser
import viser.extras
import viser.transforms as vtf
import time
import cv2
import json
import pandas as pd
from typing import Literal
import jsonlines
from openpi.utils.matrix_utils import rot_6d_to_quat


class XMILeRobotViewer:
    def __init__(self, dataset_path: str, action_horizon: int = 15):
        """
        Initialize XMI LeRobot trajectory viewer.
        
        Args:
            dataset_path: Path to LeRobot dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.action_horizon = action_horizon
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Load dataset metadata directly from files
        print(f"Loading LeRobot dataset from: {dataset_path}")
        self._load_dataset_metadata()
        
        print(f"Dataset loaded: {self.total_frames} total frames")
        print(f"Episodes: {len(self.episode_indices)}")
        print(f"Features: {list(self.features.keys())}")
        
        # Parse dataset info
        self._parse_dataset_info()
        
        # Initialize episode data
        self.current_frame_in_episode = 0
        self.episode_data = {}
        self.current_episode_name = "Loading..."
        
        # Set up viser server
        self.viser_server = viser.ViserServer()
        
        # Load first available episode
        available_episodes = sorted(self.episode_indices.keys())
        self.current_episode_idx = available_episodes[0] if available_episodes else 0
        self._load_episode_data(self.current_episode_idx)
        
        # Set up visualization
        self._setup_viser_scene()
        self._setup_viser_gui()
    
    def _load_dataset_metadata(self):
        """Load dataset metadata directly from LeRobot files."""
        # Load info.json
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {info_path}")
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.features = self.info['features']
        self.total_frames = self.info['total_frames']
        
        # Load episodes.jsonl
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
        
        self.episodes = []
        with jsonlines.open(episodes_path) as reader:
            for episode in reader:
                self.episodes.append(episode)
        
        # Create episode indices mapping (episode_index -> episode info)
        # Sort episodes by episode_index to ensure proper ordering
        sorted_episodes = sorted(self.episodes, key=lambda x: x.get('episode_index', 0))
        
        self.episode_indices = {}
        current_frame = 0
        self.max_episode_index = 0
        
        for episode in sorted_episodes:
            episode_idx = episode['episode_index']
            episode_length = episode['length']
            self.episode_indices[episode_idx] = {
                'episode_index': episode_idx,
                'from': current_frame,
                'to': current_frame + episode_length,
                'length': episode_length
            }
            current_frame += episode_length
            self.max_episode_index = max(self.max_episode_index, episode_idx)
        
        # Load tasks.jsonl
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        self.tasks = {}
        if tasks_path.exists():
            with jsonlines.open(tasks_path) as reader:
                for task in reader:
                    self.tasks[task['task_index']] = task['task']
        
        print(f"Loaded {len(self.episodes)} episodes and {len(self.tasks)} tasks")
    
    def _parse_dataset_info(self):
        """Parse dataset metadata and features."""
        # Extract camera keys (video features)
        self.camera_keys = []
        for key, feature in self.features.items():
            if feature.get('dtype') == 'video':
                self.camera_keys.append(key)
        
        print(f"Found camera keys: {self.camera_keys}")
        
        # Check for expected XMI features
        expected_features = ['state', 'actions']
        for feature in expected_features:
            if feature not in self.features:
                print(f"Warning: Expected feature '{feature}' not found in dataset")
        
        # Get action/state dimensions
        self.state_dim = self.features.get('state', {}).get('shape', [0])[0]
        self.action_dim = self.features.get('actions', {}).get('shape', [0])[0]
        
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        
        # Validate XMI dimensions (should be 20: 6D rot + 3D pos + 1D gripper per arm)
        if self.state_dim != 20 or self.action_dim != 20:
            print(f"Warning: Expected 20D state/action for XMI, got {self.state_dim}D/{self.action_dim}D")
    
    def _load_episode_data(self, episode_idx: int):
        """Load data for a specific episode."""
        if episode_idx not in self.episode_indices:
            print(f"Episode {episode_idx} not found in dataset")
            return
        
        self.current_episode_idx = episode_idx
        episode_info = self.episode_indices[episode_idx]
        
        # Find the chunk for this episode to construct episode path
        chunk_id = episode_idx // self.info.get('chunks_size', 1000)
        
        # Construct episode path for printing
        parquet_path_str = self.info["data_path"].format(
            episode_chunk=chunk_id,
            episode_index=episode_idx,
        )
        episode_path = self.dataset_path / parquet_path_str
        
        print(f"\n--- Loading Episode {episode_idx} ---")
        print(f"Episode path: {episode_path}")
        print(f"Frames: {episode_info['from']} to {episode_info['to']} ({episode_info['length']} frames) [metadata]")
        
        # Load episode data from parquet files
        self.episode_data = {}
        
        # Load parquet file for this episode (reuse chunk_id and path from above)
        parquet_path = episode_path
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            
            # Extract states and actions
            if 'state' in df.columns:
                # Convert list columns to numpy arrays
                states = np.array([np.array(state) for state in df['state']])
                self.episode_data['states'] = states
            
            if 'actions' in df.columns:
                actions = np.array([np.array(action) for action in df['actions']])
                self.episode_data['actions'] = actions
            
            print(f"Loaded parquet data: {len(df)} frames")
        else:
            print(f"Warning: Parquet file not found: {parquet_path}")
        
        # Determine actual episode length from loaded data
        episode_length = 0
        if 'states' in self.episode_data:
            episode_length = len(self.episode_data['states'])
        elif 'actions' in self.episode_data:
            episode_length = len(self.episode_data['actions'])
        elif parquet_path.exists():
            # Fallback to dataframe length if states/actions not processed
            episode_length = len(df)
        else:
            # Last resort: use metadata
            episode_length = episode_info['length']
        
        # Load video data
        for camera_key in self.camera_keys:
            video_path_str = self.info["video_path"].format(
                episode_chunk=chunk_id,
                video_key=camera_key,
                episode_index=episode_idx,
            )
            video_path = self.dataset_path / video_path_str
            
            if video_path.exists():
                # Load video frames
                frames = self._load_video_frames(video_path)
                if frames is not None:
                    self.episode_data[camera_key] = frames
                    print(f"Loaded {camera_key}: {len(frames)} frames, shape {frames[0].shape}")
            else:
                print(f"Warning: Video file not found: {video_path}")
        
        # Get task information and episode name
        # Find the episode metadata that matches this episode_index
        episode_data = None
        for ep_data in self.episodes:
            if ep_data.get('episode_index') == episode_idx:
                episode_data = ep_data
                break
        
        if episode_data is None:
            print(f"Warning: No metadata found for episode_index {episode_idx}")
            episode_data = {'episode_index': episode_idx, 'length': episode_length}
            
        task_idx = episode_data.get('task_index', 0)
        if task_idx in self.tasks:
            self.current_task = self.tasks[task_idx]
        else:
            # Try to extract task from 'tasks' field if available
            tasks_list = episode_data.get('tasks', [])
            if tasks_list:
                self.current_task = tasks_list[0]  # Use first task
            else:
                self.current_task = f"Task {task_idx}"
        
        # Extract episode name (use timestamp or other identifier if available)
        self.current_episode_name = episode_data.get('episode_name', None)
        if self.current_episode_name is None:
            # Try to extract name from path or create a meaningful name
            if hasattr(episode_data, 'timestamp'):
                self.current_episode_name = f"episode_{episode_data['timestamp']}"
            else:
                # Use the parquet filename as episode name
                self.current_episode_name = parquet_path.stem
        
        self.episode_length = episode_length
        self.current_frame_in_episode = 0
        
        print(f"Episode name: {self.current_episode_name}")
        print(f"Task: {self.current_task}")
        print(f"Actual episode length: {self.episode_length} frames (metadata claimed: {episode_info['length']})")
        if self.episode_length != episode_info['length']:
            print(f"WARNING: Episode length mismatch! Using actual data length: {self.episode_length}")
        print("---")
    
    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """Load frames from a video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if frames:
                return np.array(frames)
            else:
                return None
                
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def _setup_viser_scene(self):
        """Set up the 3D scene."""
        self.viser_server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1)
        
        # Add RBY1 base frame for reference
        self.rby1_base_frame = self.viser_server.scene.add_frame(
            "rby1_base", axes_length=0.15, axes_radius=0.004, origin_radius=0.03
        )
        
        # Add end-effector frames
        self.left_ee_frame = self.viser_server.scene.add_frame(
            "/left_ee", axes_length=0.1, axes_radius=0.005, origin_radius=0.02
        )
        self.right_ee_frame = self.viser_server.scene.add_frame(
            "/right_ee", axes_length=0.1, axes_radius=0.005, origin_radius=0.02
        )

        self.left_ee_action_frames = []
        self.right_ee_action_frames = []
        for i in range(self.action_horizon):
            self.left_ee_action_frames.append(self.viser_server.scene.add_frame(
                f"/left_ee_action_{i}", axes_length=0.07, axes_radius=0.003, origin_radius=0.01
            ))
            self.right_ee_action_frames.append(self.viser_server.scene.add_frame(
                f"/right_ee_action_{i}", axes_length=0.07, axes_radius=0.003, origin_radius=0.01
            ))

        # Add camera frustums for visualization
        self.camera_frustums = {}
        # Define some default poses for cameras relative to the RBY1 base frame
        # camera_poses = {
        #     "exterior_image_1_left": vtf.SE3.from_rotation_and_translation(
        #         vtf.SO3.from_rpy_radians(0.0, np.pi / 6, -np.pi / 2), 
        #         np.array([0.4, 0.4, 0.5])
        #     ),
        #     "exterior_image_2_right": vtf.SE3.from_rotation_and_translation(
        #         vtf.SO3.from_rpy_radians(0.0, np.pi / 6, np.pi / 2), 
        #         np.array([0.4, -0.4, 0.5])
        #     ),
        #     "exterior_image_3_top": vtf.SE3.from_rotation_and_translation(
        #         vtf.SO3.from_rpy_radians(0.0, np.pi / 2, 0.0), 
        #         np.array([0.0, 0.0, 0.8])
        #     ),
        # }

        for camera_key in self.camera_keys:
            if "left" in camera_key:
                frustum_label = f"/left_ee/{camera_key}"
                frustum = self.viser_server.scene.add_camera_frustum(
                    frustum_label,
                    fov=np.pi / 3,
                    aspect=4 / 3,
                    scale=0.1,
                    line_width=2.0,
                    wxyz = vtf.SO3.from_rpy_radians(0.0, np.pi, -np.pi).wxyz,
                )
            if "right" in camera_key:
                frustum_label = f"/right_ee/{camera_key}"
                frustum = self.viser_server.scene.add_camera_frustum(
                    frustum_label,
                    fov=np.pi / 3,
                    aspect=4 / 3,
                    scale=0.1,
                    line_width=2.0,
                    wxyz = vtf.SO3.from_rpy_radians(0.0, np.pi, -np.pi).wxyz,
                )
            if "top" in camera_key:
                frustum_label = f"/top_ee/{camera_key}"
                frustum = self.viser_server.scene.add_camera_frustum(
                    frustum_label,
                    fov=np.pi / 3,
                    aspect=4 / 3,
                    scale=0.1,
                    line_width=2.0,
                )
            self.camera_frustums[camera_key] = frustum


            # if camera_key in camera_poses:
            #     frustum.position = camera_poses[camera_key].wxyz_xyz[-3:]
            #     frustum.wxyz = camera_poses[camera_key].rotation().wxyz
            # self.camera_frustums[camera_key] = frustum

    def _setup_viser_gui(self):
        """Set up GUI controls."""
        
        with self.viser_server.gui.add_folder("Episode Selection"):
            # Get available episode indices (may not be sequential)
            available_episodes = sorted(self.episode_indices.keys())
            self.episode_selector = self.viser_server.gui.add_slider(
                "Episode",
                min=min(available_episodes),
                max=max(available_episodes),
                step=1,
                initial_value=min(available_episodes),
            )
            self.episode_info = self.viser_server.gui.add_text(
                "Episode Info", f"Episode {min(available_episodes)}/{max(available_episodes)}"
            )
            self.episode_name = self.viser_server.gui.add_text(
                "Episode Name", "Loading..."
            )
            self.task_info = self.viser_server.gui.add_text(
                "Task", getattr(self, "current_task", "Loading...")
            )
        
        with self.viser_server.gui.add_folder("Frame Navigation"):
            self.frame_slider = self.viser_server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, self.episode_length - 1),
                step=1,
                initial_value=0,
            )
            self.frame_info = self.viser_server.gui.add_text(
                "Frame Info", f"Frame 0/{self.episode_length - 1}"
            )
            self.play_button = self.viser_server.gui.add_button(
                "Play", icon=viser.Icon.PLAYER_PLAY_FILLED
            )
            self.pause_button = self.viser_server.gui.add_button(
                "Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED, visible=False
            )
            self.step_back_button = self.viser_server.gui.add_button(
                "Step Back", icon=viser.Icon.ARROW_BIG_LEFT_FILLED
            )
            self.step_forward_button = self.viser_server.gui.add_button(
                "Step Forward", icon=viser.Icon.ARROW_BIG_RIGHT_FILLED
            )

        with self.viser_server.gui.add_folder("XMI State"):
            self.left_gripper_pos = self.viser_server.gui.add_text(
                "Left Gripper", "0.0"
            )
            self.right_gripper_pos = self.viser_server.gui.add_text(
                "Right Gripper", "0.0"
            )
            self.left_ee_pos_info = self.viser_server.gui.add_text(
                "Left EE Position", "0.000, 0.000, 0.000"
            )
            self.right_ee_pos_info = self.viser_server.gui.add_text(
                "Right EE Position", "0.000, 0.000, 0.000"
            )
            self.left_ee_rot_info = self.viser_server.gui.add_text(
                "Left EE Rotation (wxyz)", "1.000, 0.000, 0.000, 0.000"
            )
            self.right_ee_rot_info = self.viser_server.gui.add_text(
                "Right EE Rotation (wxyz)", "1.000, 0.000, 0.000, 0.000"
            )
        with self.viser_server.gui.add_folder("XMI Action"):
            self.left_gripper_action = self.viser_server.gui.add_text(
                "Left Gripper", "0.0"
            )
            self.right_gripper_action = self.viser_server.gui.add_text(
                "Right Gripper", "0.0"
            )

        self.camera_displays = {}
        if self.camera_keys:
            with self.viser_server.gui.add_folder("Camera Feeds"):
                for camera_key in self.camera_keys:
                    if camera_key in self.episode_data and len(self.episode_data[camera_key]) > 0:
                        initial_img = self.episode_data[camera_key][0]
                        display_name = camera_key.replace("_", " ").title()
                        self.camera_displays[camera_key] = self.viser_server.gui.add_image(
                            image=initial_img, label=display_name
                        )

        with self.viser_server.gui.add_folder("Visualization"):
            self.show_ee_frames = self.viser_server.gui.add_checkbox(
                "Show End-Effector Frames", True
            )
            self.show_base_frame = self.viser_server.gui.add_checkbox(
                "Show RBY1 Base Frame", True
            )
            self.show_camera_frustums = self.viser_server.gui.add_checkbox(
                "Show Camera Frustums", True
            )

        @self.episode_selector.on_update
        def _(_):
            self._load_episode_data(int(self.episode_selector.value))
            self._update_gui_after_episode_change()

        @self.frame_slider.on_update
        def _(_):
            self.current_frame_in_episode = int(self.frame_slider.value)
            self._update_visualization()

        @self.play_button.on_click
        def _(_):
            self.play_button.visible = False
            self.pause_button.visible = True

        @self.pause_button.on_click
        def _(_):
            self.play_button.visible = True
            self.pause_button.visible = False

        @self.step_back_button.on_click
        def _(_):
            if self.frame_slider.value > 0:
                self.frame_slider.value -= 1

        @self.step_forward_button.on_click
        def _(_):
            if self.frame_slider.value < self.frame_slider.max:
                self.frame_slider.value += 1

        @self.show_ee_frames.on_update
        def _(_):
            self.left_ee_frame.visible = self.show_ee_frames.value
            self.right_ee_frame.visible = self.show_ee_frames.value

        @self.show_base_frame.on_update
        def _(_):
            self.rby1_base_frame.visible = self.show_base_frame.value

        @self.show_camera_frustums.on_update
        def _(_):
            for frustum in self.camera_frustums.values():
                frustum.visible = self.show_camera_frustums.value

    def _update_gui_after_episode_change(self):
        """Update GUI after changing episodes."""
        self.frame_slider.max = max(1, self.episode_length - 1)
        self.frame_slider.value = 0
        self.current_frame_in_episode = 0
        
        available_episodes = sorted(self.episode_indices.keys())
        self.episode_info.value = f"Episode {self.current_episode_idx}/{max(available_episodes)}"
        self.episode_name.value = getattr(self, "current_episode_name", "Unknown")
        self.task_info.value = self.current_task
        
        # Update camera displays
        for camera_key, display in self.camera_displays.items():
            if camera_key in self.episode_data and len(self.episode_data[camera_key]) > 0:
                display.image = self.episode_data[camera_key][0]
        
        self._update_visualization()
    
    def _parse_xmi_state(self, state: np.ndarray) -> Tuple[vtf.SE3, float, vtf.SE3, float]:
        """
        Parse XMI state vector into end-effector poses and gripper positions.
        XMI format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
        """
        if len(state) != 20:
            raise ValueError(f"Expected 20D state for XMI, got {len(state)}D")
        
        # Extract left arm data
        left_6d_rot = state[0:6]
        left_3d_pos = state[6:9]
        left_gripper = state[9]
        
        # Extract right arm data
        right_6d_rot = state[10:16]
        right_3d_pos = state[16:19]
        right_gripper = state[19]
        
        # Convert 6D rotations to quaternions (returns [w, x, y, z])
        left_quat_wxyz = rot_6d_to_quat(left_6d_rot.reshape(1, 6))[0]  # Remove batch dimension
        right_quat_wxyz = rot_6d_to_quat(right_6d_rot.reshape(1, 6))[0]  # Remove batch dimension
        
        # Create SE3 transforms
        left_se3 = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(wxyz=left_quat_wxyz), left_3d_pos
        )
        right_se3 = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(wxyz=right_quat_wxyz), right_3d_pos
        )
        
        return left_se3, left_gripper, right_se3, right_gripper
    
    def _update_visualization(self):
        """Update the 3D visualization."""
        frame_idx = self.current_frame_in_episode
        self.frame_info.value = f"Frame {frame_idx}/{self.episode_length-1}"

        if 'states' in self.episode_data and frame_idx < len(self.episode_data['states']):
            state = self.episode_data['states'][frame_idx]
            action = self.episode_data['actions'][frame_idx]
            action_horizon = self.episode_data['actions'][frame_idx:frame_idx+min(self.action_horizon, self.episode_length-frame_idx)]

            if len(action_horizon) < self.action_horizon:
                # pad with last action
                action_horizon = np.concatenate([action_horizon, action_horizon[-1:].repeat(self.action_horizon - len(action_horizon), axis=0)], axis=0)
                assert len(action_horizon) == self.action_horizon

            for i in range(self.action_horizon):
                left_se3_action, left_gripper_action, right_se3_action, right_gripper_action = self._parse_xmi_state(action_horizon[i])

                left_ee_action_frame = self.left_ee_action_frames[i]
                right_ee_action_frame = self.right_ee_action_frames[i]

                left_ee_action_frame.position = left_se3_action.wxyz_xyz[-3:]
                left_ee_action_frame.wxyz = left_se3_action.rotation().wxyz

                right_ee_action_frame.position = right_se3_action.wxyz_xyz[-3:]
                right_ee_action_frame.wxyz = right_se3_action.rotation().wxyz
                
            
            try:
                left_se3, left_gripper, right_se3, right_gripper = self._parse_xmi_state(state)
                _, left_gripper_action, _, right_gripper_action = self._parse_xmi_state(action)


                # Update gripper displays
                self.left_gripper_pos.value = f"{left_gripper:.3f}"
                self.right_gripper_pos.value = f"{right_gripper:.3f}"
                self.left_gripper_action.value = f"{left_gripper_action:.3f}"
                self.right_gripper_action.value = f"{right_gripper_action:.3f}"

                
                # Update position displays
                left_pos = left_se3.translation()
                right_pos = right_se3.translation()
                self.left_ee_pos_info.value = f"{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}"
                self.right_ee_pos_info.value = f"{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}"
                
                # Update rotation displays (quaternion wxyz)
                left_quat = left_se3.rotation().wxyz
                right_quat = right_se3.rotation().wxyz
                self.left_ee_rot_info.value = f"{left_quat[0]:.3f}, {left_quat[1]:.3f}, {left_quat[2]:.3f}, {left_quat[3]:.3f}"
                self.right_ee_rot_info.value = f"{right_quat[0]:.3f}, {right_quat[1]:.3f}, {right_quat[2]:.3f}, {right_quat[3]:.3f}"
                
                # Update end-effector frame positions and orientations
                self.left_ee_frame.position = left_pos
                self.left_ee_frame.wxyz = left_quat
                self.right_ee_frame.position = right_pos
                self.right_ee_frame.wxyz = right_quat
                
            except Exception as e:
                print(f"Error parsing XMI state: {e}")
                print(f"State shape: {state.shape}, State: {state}")
        
        # Update camera displays
        for camera_key, display in self.camera_displays.items():
            if camera_key in self.episode_data and frame_idx < len(self.episode_data[camera_key]):
                img = self.episode_data[camera_key][frame_idx]
                display.image = img
                if camera_key in self.camera_frustums:
                    self.camera_frustums[camera_key].image = img

    def run(self):
        """Run the trajectory viewer."""
        self._update_visualization()
        
        while True:
            if self.pause_button.visible:
                if self.frame_slider.value < self.frame_slider.max:
                    self.frame_slider.value += 1
                else:
                    # Auto-advance to next available episode or loop
                    available_episodes = sorted(self.episode_indices.keys())
                    current_episode = int(self.episode_selector.value)
                    try:
                        current_idx = available_episodes.index(current_episode)
                        if current_idx < len(available_episodes) - 1:
                            # Go to next available episode
                            self.episode_selector.value = available_episodes[current_idx + 1]
                        else:
                            # Loop back to first available episode
                            self.episode_selector.value = available_episodes[0]
                    except ValueError:
                        # Current episode not in available list, go to first
                        self.episode_selector.value = available_episodes[0]
            time.sleep(1.0 / 30.0) # Aim for 30fps playback


def main(
    dataset_path: str = "/home/justinyu/.cache/huggingface/lerobot/uynitsuj/xmi_rby_coffee_cup_on_dish_combined",
):
    """
    Main function for XMI LeRobot trajectory viewer.
    """
    if not Path(dataset_path).exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    viewer = XMILeRobotViewer(dataset_path=dataset_path)
    viewer.run()

if __name__ == "__main__":
    tyro.cli(main)