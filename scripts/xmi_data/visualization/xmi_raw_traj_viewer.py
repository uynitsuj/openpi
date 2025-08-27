import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional
import tyro
import viser
import viser.extras
import viser.transforms as vtf
# from robot_descriptions.loaders.yourdfpy import load_robot_description
import time
import cv2
import json
from traj_view_utils import *
from typing import Literal

class XMITrajectoryViewer:
    def __init__(self, data_dir: str, 
                 enable_stereo: bool = False,
                 resize_method: Literal["center_crop", "pad"] = "center_crop",
                 data_source: Literal["sz", "us"] = "sz",
                 stereo_ckpt_dir: str = "FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth",
                 stereo_baseline: float = 0.12,
                 intrinsics_path: str = "/nfs_us/justinyu/us_xmi_calib/SN39962371.conf",
                 quest_to_zed_calib: str = "/nfs_us/justinyu/us_xmi_calib/Head_Franka_20250604_12/calib_results/head2cam.npy",
                 left_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Left_Controller_20250603_15/calib_results/controller2franka.npy",
                 right_controller_calib: str = "/nfs_us/justinyu/us_xmi_calib/Right_Controller_20250603_15/calib_results/controller2franka.npy"
                 ):
        self.data_dir = Path(data_dir)
        self.data_source = data_source
        self.current_idx = 0
        self.resize_method = resize_method
        if resize_method == "center_crop":
            self.resize_func = resize_with_center_crop
        elif resize_method == "pad":
            self.resize_func = resize_with_pad
        else:
            raise ValueError(f"Invalid resize method: {resize_method}")
        
        self.head_height = 0.95
        self._extracting_video = False
        
        # Stereo processing setup
        self.enable_stereo = enable_stereo
        self.stereo_baseline = stereo_baseline
        self.stereo_model = None
        self.intrinsics = None
        self.distortion = None
        
        if enable_stereo:
            try:
                from traj_view_utils import load_foundation_stereo_model, load_ZED_intrinsics
                self.stereo_model = load_foundation_stereo_model(stereo_ckpt_dir)
                assert self.stereo_model is not None, "Failed to load stereo model"
                self.intrinsics, self.distortion = load_ZED_intrinsics(intrinsics_path, "LEFT_CAM_HD")
                print(f"Foundation Stereo model loaded successfully")
            except Exception as e:
                print(f"Failed to load stereo model: {e}")
                self.enable_stereo = False
        
        # Load calibration matrices
        self.quest_to_zed_calib_tf = None
        self.left_controller_calib_tf = None  
        self.right_controller_calib_tf = None

        self.sz_to_us_tf = vtf.SE3.from_matrix(np.array([[1.0, 0.0,      0.0,      0.0],
                                                    [0.0, 0.66913,  0.74314,  0.02955328],
                                                    [0.0, -0.74314, 0.66913,  0.00117534],
                                                    [0.0, 0.0,      0.0,      1.0]]))
        self.sz_to_us_tf = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.from_rpy_radians(-25 * np.pi/180, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
        )
        
        if quest_to_zed_calib and os.path.exists(quest_to_zed_calib):
            quest_to_zed_calib_matrix = np.load(quest_to_zed_calib)
            self.quest_to_zed_calib_tf = vtf.SE3.from_matrix(quest_to_zed_calib_matrix)
            print(f"Loaded Quest to ZED calibration from: {quest_to_zed_calib}")
        
        if left_controller_calib and os.path.exists(left_controller_calib):
            left_controller_calib_matrix = np.load(left_controller_calib)
            self.left_controller_calib_tf = vtf.SE3.from_rotation_and_translation(
                vtf.SO3.from_matrix(left_controller_calib_matrix[:3, :3]), 
                left_controller_calib_matrix[:3, 3]
            ).inverse()
            print(f"Loaded left controller calibration from: {left_controller_calib}")
            
        if right_controller_calib and os.path.exists(right_controller_calib):
            right_controller_calib_matrix = np.load(right_controller_calib)
            self.right_controller_calib_tf = vtf.SE3.from_rotation_and_translation(
                vtf.SO3.from_matrix(right_controller_calib_matrix[:3, :3]), 
                right_controller_calib_matrix[:3, 3]
            ).inverse()
            print(f"Loaded right controller calibration from: {right_controller_calib}")
        
        # Find all episode directories
        self.episodes = self._find_episode_directories(self.data_dir)
        if not self.episodes:
            raise ValueError(f"No episode directories found in {self.data_dir}")
        
        self.current_episode = self.episodes[0]
        
        # Flag to prevent circular updates between episode controls
        self._updating_episode_controls = False
        
        # Load the first episode data
        self._load_episode_data(self.current_episode)
        
        # Set up viser with RBY1 URDF 
        self.viser_server = viser.ViserServer()
        # try:
        #     self.urdf = load_robot_description("rby1_description")
        #     self.has_urdf = True
        # except:
        #     print("Warning: Could not load RBY1 URDF, proceeding without robot visualization")
        #     self.has_urdf = False
        
        self._setup_viser_scene()
        self._setup_viser_gui()
    
    def _find_episode_directories(self, parent_dir: Path) -> List[Path]:
        """Find all episode directories under the parent directory."""
        episode_dirs = []
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith("episode_"):
                episode_dirs.append(item)
        return sorted(episode_dirs)
    
    def _load_episode_data(self, episode_path: Path):
        """Load data for a specific episode."""
        self.loading_episode = True
        self.current_episode = episode_path
        print(f"Loading episode: {episode_path.name}")
        
        try:
            # Load action data (4x4 transformation matrices)
            self.action_data = {}
            action_files = [
                "action-left-hand_in_quest_world_frame.npy",
                "action-left-head.npy", 
                "action-left-pos.npy",
                "action-left-quest_world_frame.npy",
                "action-right-hand_in_quest_world_frame.npy",
                "action-right-head.npy",
                "action-right-pos.npy", 
                "action-right-quest_world_frame.npy"
            ]
            
            for action_file in action_files:
                file_path = episode_path / action_file
                if file_path.exists():
                    self.action_data[action_file.replace('.npy', '')] = np.load(file_path)
            
            # Load joint data
            self.joint_data = {}
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
                    self.joint_data[joint_file.replace('.npy', '')] = np.load(file_path)
            
            # Load timestamp data
            self.timestamp_data = {}
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
                    self.timestamp_data[timestamp_file.replace('.npy', '')] = np.load(file_path)
            
            # Load annotation data if available
            annotation_file = episode_path / "top_camera-images-rgb_annotation.json"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    self.annotations = json.load(f)
            else:
                self.annotations = None
            
            # Find and extract video files
            self.video_files = {}
            self.images = []
            camera_names = []

            self._extracting_video = True
            
            for video_file in episode_path.glob("*.mp4"):
                camera_name = video_file.stem  # e.g., "left_camera-images-rgb"
                self.video_files[camera_name] = video_file
                
                # Extract frames from video
                frames = extract_video_frames(video_file)
                if len(frames) > 0:
                    self.images.append(frames)
                    camera_names.append(camera_name)

            self._extracting_video = False
            
            # Keep images as list since different cameras may have different resolutions
            if self.images:
                # Find the minimum number of frames across all cameras
                min_frames = min(len(frames) for frames in self.images)
                
                # Truncate all camera feeds to the same length
                self.images = [frames[:min_frames] for frames in self.images]
                print(f"Loaded {len(self.images)} camera feeds with {min_frames} frames each")
                
                # Print resolution info for each camera
                for idx, camera_name in enumerate(camera_names):
                    shape = self.images[idx].shape
                    print(f"  {camera_name}: {shape[1]}x{shape[2]} resolution")
            
            self.camera_names = camera_names
            
            # Determine the number of frames (use the longest sequence)
            frame_counts = []
            for data in self.action_data.values():
                frame_counts.append(len(data))
            for data in self.joint_data.values():
                frame_counts.append(len(data))
            for data in self.timestamp_data.values():
                frame_counts.append(len(data))
            
            # Also consider video frames
            if len(self.images) > 0:
                frame_counts.append(len(self.images[0]))  # Use first camera's frame count
            
            self.total_frames = max(frame_counts) if frame_counts else 0
            print(f"Total frames: {self.total_frames}")
            
        except Exception as e:
            print(f"Error loading episode data: {e}")
            raise
        self.loading_episode = False
    
    def _setup_viser_scene(self):
        """Set up the viser 3D scene."""
        self.viser_server.scene.add_grid("/ground", width=2, height=2)
        
        # Add robot URDF visualization if available
        # if self.has_urdf:
        #     self.urdf_vis = viser.extras.ViserUrdf(
        #         self.viser_server,
        #         self.urdf, 
        #         root_node_name="/base"
        #     )

        # Add frames for head tracking
        self.tf_head_frame = self.viser_server.scene.add_frame(
            "head_tf",
            axes_length=0.1,
            axes_radius=0.005,
            origin_radius=0.02,
        )
        
        # Add ZED frame relative to head (if calibration available)
        if self.quest_to_zed_calib_tf is not None:
            if self.data_source == "sz":
                self.quest_to_zed_calib_tf = self.quest_to_zed_calib_tf @ self.sz_to_us_tf.inverse()
            elif self.data_source == "us":
                pass
            else:
                raise ValueError(f"Invalid data source: {self.data_source}")

            self.zed_tf_handle = self.viser_server.scene.add_frame(
                "head_tf/zed_frame",
                wxyz=self.quest_to_zed_calib_tf.wxyz_xyz[:4],
                position=self.quest_to_zed_calib_tf.wxyz_xyz[-3:],
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.015,
            )

        # Add frames for Quest controller tracking
        self.tf_left_hand_frame = self.viser_server.scene.add_frame(
            "left_hand_tf",
            axes_length=0.1,
            axes_radius=0.005,
            origin_radius=0.02,
        )
        
        self.tf_right_hand_frame = self.viser_server.scene.add_frame(
            "right_hand_tf", 
            axes_length=0.1,
            axes_radius=0.005,
            origin_radius=0.02,
        )
        
        # Add end effector frames (if controller calibrations available)
        if self.left_controller_calib_tf is not None:
            self.left_franka_ee_tf_handle = self.viser_server.scene.add_frame(
                "left_hand_tf/left_franka_ee_tf", 
                wxyz=self.left_controller_calib_tf.wxyz_xyz[:4],
                position=self.left_controller_calib_tf.wxyz_xyz[-3:],
                axes_length=0.08, 
                axes_radius=0.004, 
                origin_radius=0.015,
                show_axes=False
            )
            
            # Add end effector TCP frame with offset (same as combined viewer)
            yaw_45 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4), np.array([0.0, 0.0, 0.0]))
            offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.identity(), np.array([-0.08275, 0.0, 0.005 + 0.12]))
            ee_tf = yaw_45 @ offset
            
            self.left_ee_tf_handle = self.viser_server.scene.add_frame(
                "left_hand_tf/left_franka_ee_tf/left_ee_tf", 
                wxyz=ee_tf.wxyz_xyz[:4],
                position=ee_tf.wxyz_xyz[-3:],
                axes_length=0.1, 
                axes_radius=0.005, 
                origin_radius=0.02
            )
            
        if self.right_controller_calib_tf is not None:
            self.right_franka_ee_tf_handle = self.viser_server.scene.add_frame(
                "right_hand_tf/right_franka_ee_tf", 
                wxyz=self.right_controller_calib_tf.wxyz_xyz[:4],
                position=self.right_controller_calib_tf.wxyz_xyz[-3:],
                axes_length=0.08, 
                axes_radius=0.004, 
                origin_radius=0.015,
                show_axes=False
            )
            
            # Add end effector TCP frame with offset (same as combined viewer)
            yaw_45 = vtf.SE3.from_rotation_and_translation(vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4), np.array([0.0, 0.0, 0.0]))
            offset = vtf.SE3.from_rotation_and_translation(vtf.SO3.identity(), np.array([-0.08275, 0.0, 0.005 + 0.12]))
            ee_tf = yaw_45 @ offset
            
            self.right_ee_tf_handle = self.viser_server.scene.add_frame(
                "right_hand_tf/right_franka_ee_tf/right_ee_tf", 
                wxyz=ee_tf.wxyz_xyz[:4],
                position=ee_tf.wxyz_xyz[-3:],
                axes_length=0.1, 
                axes_radius=0.005, 
                origin_radius=0.02
            )

        self.left_frustum = self.viser_server.scene.add_camera_frustum(
            "left_hand_tf/left_frustum",
            fov=np.pi/3,  # 60 degrees vertical FOV
            aspect=1,   # Standard 4:3 aspect ratio
            scale=0.15,    # Smaller scale for visualization
            line_width=1.5,
            position=(0.0, 0.0, 0.15),
            wxyz=(vtf.SO3.from_rpy_radians(-np.pi/2, 0.0, 0.0)).wxyz,
        )
        self.right_frustum = self.viser_server.scene.add_camera_frustum(
            "right_hand_tf/right_frustum",
            fov=np.pi/3,  # 60 degrees vertical FOV
            aspect=1,   # Standard 4:3 aspect ratio
            scale=0.15,    # Smaller scale for visualization
            line_width=1.5,
            position=(0.0, 0.0, 0.15),
            wxyz=(vtf.SO3.from_rpy_radians(-np.pi/2, 0.0, 0.0)).wxyz,
        )
        self.head_frustum = self.viser_server.scene.add_camera_frustum(
            "head_tf/zed_frame/head_frustum",
            fov=np.pi/3,  # 60 degrees vertical FOV
            aspect=1,   
            scale=0.15,    # Smaller scale for visualization
            line_width=1.5,
            # wxyz=(vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi)).wxyz,
        )
        
        # Add point cloud for stereo depth visualization
        self.point_cloud_handle = None
        if self.enable_stereo and self.quest_to_zed_calib_tf is not None:
            # Initialize with empty point cloud attached to ZED frame
            self.point_cloud_handle = self.viser_server.scene.add_point_cloud(
                "head_tf/zed_frame/stereo_pointcloud", 
                np.zeros((0, 3)), 
                np.zeros((0, 3)), 
                point_size=0.002, 
                point_shape='circle'
            )
    
    def _setup_viser_gui(self):
        """Set up the viser GUI elements."""
        
        # Episode selection
        with self.viser_server.gui.add_folder("Episode Selection"):
            episode_names = [ep.name for ep in self.episodes]
            self.episode_selector = self.viser_server.gui.add_dropdown(
                "Select Episode",
                options=episode_names,
                initial_value=episode_names[0]
            )
            
            # Add episode slider for easy navigation
            self.episode_slider = self.viser_server.gui.add_slider(
                "Episode Index",
                min=0,
                max=max(1, len(self.episodes) - 1),
                step=1,
                initial_value=0
            )
            
            # Only add episode navigation if we have multiple episodes
            if len(self.episodes) > 1:
                # Add navigation buttons for episodes
                with self.viser_server.gui.add_folder("Navigation"):
                    self.prev_episode_button = self.viser_server.gui.add_button(
                        label="Previous Episode", 
                        icon=viser.Icon.CHEVRON_LEFT
                    )
                    self.next_episode_button = self.viser_server.gui.add_button(
                        label="Next Episode", 
                        icon=viser.Icon.CHEVRON_RIGHT
                    )
            
            @self.episode_selector.on_update
            def _(_) -> None:
                if self._updating_episode_controls:
                    return
                self._updating_episode_controls = True
                try:
                    selected_idx = episode_names.index(self.episode_selector.value)
                    self.episode_slider.value = selected_idx  # Sync slider
                    self._load_episode_data(self.episodes[selected_idx])
                    self._update_gui_after_episode_change()
                finally:
                    self._updating_episode_controls = False
            
            @self.episode_slider.on_update
            def _(_) -> None:
                if self._updating_episode_controls:
                    return
                self._updating_episode_controls = True
                try:
                    episode_idx = int(self.episode_slider.value)
                    self.episode_selector.value = episode_names[episode_idx]  # Sync dropdown
                    self._load_episode_data(self.episodes[episode_idx])
                    self._update_gui_after_episode_change()
                finally:
                    self._updating_episode_controls = False
            
            # Add episode navigation button callbacks if they exist
            if len(self.episodes) > 1:
                @self.prev_episode_button.on_click
                def _(_) -> None:
                    self.slider_handle.value = 0
                    current_idx = int(self.episode_slider.value)
                    # Wrap around to last episode if at the beginning
                    new_idx = (current_idx - 1) % len(episode_names)
                    self._updating_episode_controls = True
                    try:
                        self.episode_slider.value = new_idx
                    finally:
                        self._updating_episode_controls = False
                    
                @self.next_episode_button.on_click
                def _(_) -> None:
                    self.slider_handle.value = 0
                    current_idx = int(self.episode_slider.value)
                    # Wrap around to first episode if at the end
                    new_idx = (current_idx + 1) % len(episode_names)
                    self._updating_episode_controls = True
                    try:
                        self.episode_slider.value = new_idx
                    finally:
                        self._updating_episode_controls = False
        
        # Playback controls
        with self.viser_server.gui.add_folder("Playback Controls"):
            self.play_button = self.viser_server.gui.add_button(
                label="Play", 
                icon=viser.Icon.PLAYER_PLAY_FILLED
            )
            self.pause_button = self.viser_server.gui.add_button(
                label="Pause", 
                icon=viser.Icon.PLAYER_PAUSE_FILLED, 
                visible=False
            )
            self.next_button = self.viser_server.gui.add_button(
                label="Step Forward", 
                icon=viser.Icon.ARROW_BIG_RIGHT_FILLED
            )
            self.prev_button = self.viser_server.gui.add_button(
                label="Step Back", 
                icon=viser.Icon.ARROW_BIG_LEFT_FILLED
            )
        
        # Frame slider
        self.slider_handle = self.viser_server.gui.add_slider(
            "Frame Index", 
            min=0, 
            max=max(1, self.total_frames - 1), 
            step=1, 
            initial_value=0
        )
        
        # Image viewers for camera feeds
        self.viser_img_handles = []
        if len(self.images) > 0:
            with self.viser_server.gui.add_folder("Camera Observations"):
                for idx, camera_name in enumerate(self.camera_names):
                    # Use first frame as initial image
                    initial_image = self.images[idx][0] if len(self.images[idx]) > 0 else np.zeros((480, 640, 3), dtype=np.uint8)
                    self.viser_img_handles.append(self.viser_server.gui.add_image(
                        image=initial_image,
                        label=camera_name.replace('-images-rgb', '').replace('_', ' ').title()
                    ))
        
        # Data display
        with self.viser_server.gui.add_folder("Robot State"):
            self.frame_info = self.viser_server.gui.add_text("Frame Info", "Frame: 0")
            self.left_gripper_pos = self.viser_server.gui.add_number("Left Gripper Position", 0.0, disabled=True)
            self.right_gripper_pos = self.viser_server.gui.add_number("Right Gripper Position", 0.0, disabled=True)

        # Annotation display
        with self.viser_server.gui.add_folder("Annotations"):
            self.current_annotation = self.viser_server.gui.add_text("Current Task", "No annotation")
        
        # Coordinate frame toggles
        with self.viser_server.gui.add_folder("Visualization Options"):
            self.show_left_hand = self.viser_server.gui.add_checkbox("Show Left Hand Frame", True)
            self.show_right_hand = self.viser_server.gui.add_checkbox("Show Right Hand Frame", True)
            self.show_head_frames = self.viser_server.gui.add_checkbox("Show Head Frames", True)
            self.show_frustums = self.viser_server.gui.add_checkbox("Show Camera Frustums", True)
            
            # Stereo visualization controls
            if self.enable_stereo:
                with self.viser_server.gui.add_folder("Stereo Depth"):
                    self.enable_stereo_viz = self.viser_server.gui.add_checkbox("Enable Stereo Processing", True)
                    self.stereo_point_size = self.viser_server.gui.add_slider(
                        "Point Size", min=0.001, max=0.02, step=0.001, initial_value=0.002
                    )
                    self.stereo_subsample = self.viser_server.gui.add_slider(
                        "Subsample Factor", min=1, max=8, step=1, initial_value=1
                    )
                    self.stereo_far_clip = self.viser_server.gui.add_slider(
                        "Far Clip (m)", min=1.0, max=50.0, step=1.0, initial_value=10.0
                    )
                    self.stereo_near_clip = self.viser_server.gui.add_slider(
                        "Near Clip (m)", min=0.01, max=2.0, step=0.01, initial_value=0.1
                    )
        
        # Set up button callbacks
        @self.play_button.on_click
        def _(_) -> None:
            self.play_button.visible = False
            self.pause_button.visible = True
            
        @self.pause_button.on_click
        def _(_) -> None:
            self.play_button.visible = True
            self.pause_button.visible = False
        
        @self.next_button.on_click
        def _(_) -> None:
            if self.slider_handle.value < self.slider_handle.max:
                self.slider_handle.value += 1
            
        @self.prev_button.on_click
        def _(_) -> None:
            if self.slider_handle.value > 0:
                self.slider_handle.value -= 1
        
        @self.slider_handle.on_update
        def _(_) -> None:
            self._update_visualization()
        
        @self.show_left_hand.on_update
        def _(_) -> None:
            self.tf_left_hand_frame.visible = self.show_left_hand.value
        
        @self.show_right_hand.on_update
        def _(_) -> None:
            self.tf_right_hand_frame.visible = self.show_right_hand.value
        
        @self.show_head_frames.on_update
        def _(_) -> None:
            self.tf_head_frame.visible = self.show_head_frames.value
            if hasattr(self, 'zed_tf_handle'):
                self.zed_tf_handle.visible = self.show_head_frames.value

        @self.show_frustums.on_update
        def _(_) -> None:
            self.left_frustum.visible = self.show_frustums.value
            self.right_frustum.visible = self.show_frustums.value
            self.head_frustum.visible = self.show_frustums.value
            
        # Stereo visualization callbacks
        if self.enable_stereo:
            @self.stereo_point_size.on_update
            def _(_) -> None:
                if self.point_cloud_handle is not None:
                    self.point_cloud_handle.point_size = self.stereo_point_size.value
    
    def _update_gui_after_episode_change(self):
        """Update GUI elements after changing episodes."""
        self.slider_handle.max = max(1, self.total_frames - 1)
        self.slider_handle.value = 0
        self._update_visualization()
    
    def _get_current_annotation(self, frame_idx):
        """Get the current annotation for the given frame."""
        if not self.annotations or 'annotations' not in self.annotations:
            return "No annotation"
        
        for annotation in self.annotations['annotations']:
            if annotation['from_frame'] <= frame_idx <= annotation['to_frame']:
                return annotation['label']
        
        return "No annotation"
    
    def _update_visualization(self):
        # start_time = time.time()
        while self.loading_episode:
            time.sleep(0.1)
        """Update the visualization based on current frame."""
        frame_idx = self.slider_handle.value
        self.frame_info.value = f"Frame: {frame_idx}/{self.total_frames-1}"

        # Apply coordinate transformation from Quest to world frame (same as combined viewer)
        q2w = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
        )

        # Update head frame
        if "action-left-head" in self.action_data and frame_idx < len(self.action_data["action-left-head"]):
            head_matrix = self.action_data["action-left-head"][frame_idx]
            # right_head_matrix = self.action_data["action-right-head"][frame_idx]
            # print(f"head_matrix: {head_matrix}")
            # print(f"right_head_matrix: {right_head_matrix}")
            head_tf = vtf.SE3.from_matrix(head_matrix)
            head_tf = q2w @ head_tf
            
            # Apply coordinate transformation like in combined viewer
            self.tf_head_frame.position = np.array([head_tf.wxyz_xyz[-3], -head_tf.wxyz_xyz[-2], head_tf.wxyz_xyz[-1]])
            self.tf_head_frame.wxyz = vtf.SO3.from_rpy_radians(
                -head_tf.rotation().as_rpy_radians().roll,
                head_tf.rotation().as_rpy_radians().pitch,
                -head_tf.rotation().as_rpy_radians().yaw,
            ).wxyz

            # self.tf_head_frame.position = head_tf.wxyz_xyz[-3:]
            # self.tf_head_frame.wxyz = head_tf.wxyz_xyz[:4]
        # print(f"Time taken to update head frame: {time.time() - start_time} seconds")
        # Update left hand frame
        if "action-left-hand_in_quest_world_frame" in self.action_data and frame_idx < len(self.action_data["action-left-hand_in_quest_world_frame"]):
            left_hand_matrix = self.action_data["action-left-hand_in_quest_world_frame"][frame_idx]
            world_frame = self.action_data["action-left-quest_world_frame"][frame_idx]
            left_hand_tf = vtf.SE3.from_matrix(left_hand_matrix)
            left_hand_tf = q2w @ vtf.SE3.from_matrix(world_frame) @ left_hand_tf
            
            self.tf_left_hand_frame.position = np.array([left_hand_tf.wxyz_xyz[-3], -left_hand_tf.wxyz_xyz[-2], left_hand_tf.wxyz_xyz[-1]])
            self.tf_left_hand_frame.wxyz = vtf.SO3.from_rpy_radians(
                -left_hand_tf.rotation().as_rpy_radians().roll,
                left_hand_tf.rotation().as_rpy_radians().pitch,
                -left_hand_tf.rotation().as_rpy_radians().yaw,
            ).wxyz
        # print(f"Time taken to update left hand frame: {time.time() - start_time} seconds")
        # Update right hand frame 
        if ("action-right-hand_in_quest_world_frame" in self.action_data and 
            "action-left-quest_world_frame" in self.action_data and 
            "action-right-quest_world_frame" in self.action_data and
            frame_idx < len(self.action_data["action-right-hand_in_quest_world_frame"])):
            
            left_quest_world_frame = self.action_data["action-left-quest_world_frame"][frame_idx]
            right_world_frame = self.action_data["action-right-quest_world_frame"][frame_idx]
            right_hand_matrix = self.action_data["action-right-hand_in_quest_world_frame"][frame_idx]
            
            # Transform right hand through the quest world frames like in combined viewer
            right_hand_in_world = np.linalg.inv(left_quest_world_frame) @ right_world_frame @ right_hand_matrix
            right_hand_tf = vtf.SE3.from_matrix(right_hand_in_world)
            right_hand_tf = q2w @ vtf.SE3.from_matrix(right_world_frame) @ right_hand_tf
            
            self.tf_right_hand_frame.position = np.array([right_hand_tf.wxyz_xyz[-3], -right_hand_tf.wxyz_xyz[-2], right_hand_tf.wxyz_xyz[-1]])
            self.tf_right_hand_frame.wxyz = vtf.SO3.from_rpy_radians(
                -right_hand_tf.rotation().as_rpy_radians().roll,
                right_hand_tf.rotation().as_rpy_radians().pitch,
                -right_hand_tf.rotation().as_rpy_radians().yaw,
            ).wxyz
        # print(f"Time taken to update right hand frame: {time.time() - start_time} seconds")
        # Update gripper positions
        if "left-gripper_pos" in self.joint_data:
            if frame_idx < len(self.joint_data["left-gripper_pos"]):
                self.left_gripper_pos.value = float(self.joint_data["left-gripper_pos"][frame_idx][0])
        
        if "right-gripper_pos" in self.joint_data:
            if frame_idx < len(self.joint_data["right-gripper_pos"]):
                self.right_gripper_pos.value = float(self.joint_data["right-gripper_pos"][frame_idx][0])
        # print(f"Time taken to update gripper positions: {time.time() - start_time} seconds")
        # Update camera images
        if len(self.images) > 0 and len(self.viser_img_handles) > 0 and not self._extracting_video:
            # Make sure frame_idx is within bounds for video frames
            video_frame_idx = min(frame_idx, len(self.images[0]) - 1)
            for idx, img_handle in enumerate(self.viser_img_handles):
                if idx < len(self.images):
                    if "top" in self.camera_names[idx]:
                        if self.images[idx][video_frame_idx].shape[1] / self.images[idx][video_frame_idx].shape[0] > 3:
                            top_img = self.images[idx][video_frame_idx][:, :self.images[idx][video_frame_idx].shape[1]//2, :]
                        else:
                            top_img = self.images[idx][video_frame_idx]
                        img_handle.image = self.resize_func(top_img, 224, 224)
                    else:
                        img_handle.image = self.resize_func(self.images[idx][video_frame_idx], 224, 224)
                    # update camera names
                    img_handle.label = self.camera_names[idx].replace('-images-rgb', '').replace('_', ' ').title()

                    if "left" in self.camera_names[idx]:
                        self.left_frustum.image = self.resize_func(self.images[idx][video_frame_idx], 224, 224)
                    elif "right" in self.camera_names[idx]:
                        self.right_frustum.image = self.resize_func(self.images[idx][video_frame_idx], 224, 224)
                    elif "top" in self.camera_names[idx]:
                        if self.images[idx][video_frame_idx].shape[1] / self.images[idx][video_frame_idx].shape[0] > 3:
                            top_img = self.images[idx][video_frame_idx][:, :self.images[idx][video_frame_idx].shape[1]//2, :]
                        else:
                            top_img = self.images[idx][video_frame_idx]
                        self.head_frustum.image = self.resize_func(top_img, 224, 224)
        # print(f"Time taken to update camera images: {time.time() - start_time} seconds")
        # Update stereo depth visualization
        if (self.enable_stereo and hasattr(self, 'enable_stereo_viz') and 
            self.enable_stereo_viz.value and self.stereo_model is not None and
            len(self.images) > 0 and not self._extracting_video):
            
            # Find top camera (stereo pair)
            top_camera_idx = None
            for idx, camera_name in enumerate(self.camera_names):
                if "top" in camera_name:
                    top_camera_idx = idx
                    break
            
            if top_camera_idx is not None:
                try:
                    from traj_view_utils import process_stereo_pair, depth_color_to_pointcloud
                    
                    video_frame_idx = min(frame_idx, len(self.images[0]) - 1)
                    top_image = self.images[top_camera_idx][video_frame_idx]
                    
                    # Split stereo pair (left/right images)
                    H, W = top_image.shape[:2]
                    img_left = top_image[:, :W//2, :]
                    img_right = top_image[:, W//2:, :]
                    
                    # Process stereo pair to get depth
                    depth = process_stereo_pair(
                        img_left, img_right, self.stereo_model, 
                        self.intrinsics, self.distortion, 
                        self.stereo_baseline
                    )
                    
                    # Convert depth to point cloud
                    points, colors = depth_color_to_pointcloud(
                        depth, img_left, self.intrinsics,
                        subsample_factor=int(self.stereo_subsample.value),
                        far_clip=self.stereo_far_clip.value,
                        near_clip=self.stereo_near_clip.value
                    )

                    self.point_cloud_handle.points = points
                    self.point_cloud_handle.colors = colors
                    
                except Exception as e:
                    print(f"Stereo processing error: {e}")
        # print(f"Time taken to update stereo depth visualization: {time.time() - start_time} seconds")
        # Update annotation
        current_annotation = self._get_current_annotation(frame_idx)
        self.current_annotation.value = current_annotation
        
        # Apply visibility settings
        self.tf_left_hand_frame.visible = self.show_left_hand.value
        self.tf_right_hand_frame.visible = self.show_right_hand.value
        self.tf_head_frame.visible = self.show_head_frames.value
        if hasattr(self, 'zed_tf_handle'):
            self.zed_tf_handle.visible = self.show_head_frames.value

        # print(f"Time taken to update visualization: {time.time() - start_time} seconds")
    
    def run(self):
        """Run the trajectory viewer."""
        print("Starting XMI Trajectory Viewer...")
        print("Use the GUI to navigate through episodes and frames.")
        print(f"Found {len(self.video_files)} video files:")
        for name, path in self.video_files.items():
            print(f"  - {name}: {path}")
        
        if len(self.images) > 0:
            print(f"Loaded {len(self.images)} camera feeds with {len(self.images[0])} frames each")
        
        # Initialize visualization
        self._update_visualization()
        
        while True:
            # Auto-play functionality
            if self.pause_button.visible:
                if self.slider_handle.value < self.slider_handle.max:
                    self.slider_handle.value += 1
                else:
                    # Roll back to start when reaching the end
                    self.slider_handle.value = 0
                    # Move to next episode
                    current_idx = int(self.episode_slider.value)
                    new_idx = (current_idx + 1) % len(self.episodes)
                    
                    # Update controls and load new episode
                    self._updating_episode_controls = True
                    try:
                        episode_names = [ep.name for ep in self.episodes]
                        self.episode_slider.value = new_idx
                        self.episode_selector.value = episode_names[new_idx]
                        self._load_episode_data(self.episodes[new_idx])
                        self._update_gui_after_episode_change()
                    finally:
                        self._updating_episode_controls = False

            time.sleep(0.05)

def main(
    data_dir = "/home/justinyu/openpi/problematic_episodes_uynitsuj_shelf_soup_in_domain_xmi_data_20250818/Jerky_motion_detected_(rotation)",
    enable_stereo: bool = False,
    stereo_baseline: float = 0.12,
    intrinsics_path: str = "/nfs_us/justinyu/us_xmi_calib/SN39962371.conf",
    quest_to_zed_calib: str = "/nfs_us/justinyu/us_xmi_calib/Head_Franka_20250604_12/calib_results/head2cam.npy",
    left_controller_calib: str = "/nfs_us/justinyu/us_xmi_calibLeft_Controller_20250603_15/calib_results/controller2franka.npy",
    right_controller_calib: str = "/nfs_us/justinyu/us_xmi_calibRight_Controller_20250603_15/calib_results/controller2franka.npy",
    ):
    """Main function for XMI trajectory viewer."""
    # current_file_path = os.path.abspath(__file__)
    stereo_ckpt_dir = "/home/justinyu/dev/viser_xmi_data_vis/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth"

    viewer = XMITrajectoryViewer(
        data_dir=data_dir,
        enable_stereo=enable_stereo,
        stereo_ckpt_dir=stereo_ckpt_dir,
        stereo_baseline=stereo_baseline,
        intrinsics_path=intrinsics_path,
        quest_to_zed_calib=quest_to_zed_calib,
        left_controller_calib=left_controller_calib,
        right_controller_calib=right_controller_calib,
    )
    viewer.run()


if __name__ == "__main__":
    tyro.cli(main) 
