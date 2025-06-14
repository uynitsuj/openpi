from typing import Literal, Optional, Tuple
import time
from dataclasses import dataclass
from loguru import logger

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import viser
import viser.extras
import viser.transforms as vtf

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print(
        "pip install git+https://github.com/uynitsuj/robot_descriptions.py.git@e2502f3d4d8aa38d18b72527c5baae2b19a1182e"
    )
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()

import pyroki as pk
import os
from pathlib import Path

# Import the solve_ik function from pyroki_snippets  
from xdof.pyroki.pyroki_snippets import solve_ik_with_multiple_targets

@dataclass
class TransformHandle:
    """Data class to store transform handles."""
    frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None

class YAMSBaseInterface:
    """
    Base interface for dual YAM (YAMS) robot visualization.
    - This class handles two YAM robots positioned at configurable xy locations
    - Running this file allows IK control of dual virtual YAM robots in viser with transform handle gizmos.
    """
    
    def __init__(
        self,
        server: viser.ViserServer = None,
        minimal: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
        left_base_xy: Tuple[float, float] = (0.0, 0.0),
        right_base_xy: Tuple[float, float] = (0.0, -0.61),
    ):
        self.minimal = minimal
        self.left_base_xy = left_base_xy
        self.right_base_xy = right_base_xy
        
        # Set device
        jax.config.update("jax_platform_name", device)

        # Initialize viser server
        self.server = server if server is not None else viser.ViserServer()
        
        # Load robot description using PyRoki for both arms
        self.urdf = load_robot_description("yam_description")
        self.robot_left = pk.Robot.from_urdf(self.urdf)
        self.robot_right = pk.Robot.from_urdf(self.urdf)
        
        # Get rest pose from URDF default configuration
        self.rest_pose = self.urdf.cfg
        self.joints_left = self.rest_pose.copy()
        self.joints_right = self.rest_pose.copy()
        
        # Target link names for each YAM end effector 
        self.target_names = ["link_6"]  # YAM end effector is link_6
        print(f"[YAMSBaseInterface] Target link name for each arm: {self.target_names}")
        
        # Setup visualization
        self._setup_visualization()
        
        if not minimal:
            self._setup_gui()
            self._setup_transform_handles()
        
        # Initialize base poses for each arm
        self.base_pose_left = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), 
            jnp.array([left_base_xy[0], left_base_xy[1], 0.0])
        )
        self.base_pose_right = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), 
            jnp.array([right_base_xy[0], right_base_xy[1], 0.0])
        )
        
        # Update base frame positions
        self.base_frame_left.position = onp.array(self.base_pose_left.translation())
        self.base_frame_left.wxyz = onp.array(self.base_pose_left.rotation().wxyz)
        self.base_frame_right.position = onp.array(self.base_pose_right.translation())
        self.base_frame_right.wxyz = onp.array(self.base_pose_right.rotation().wxyz)
        
        if not minimal:
            # Initialize solver parameters
            self.has_jitted_left = False
            self.has_jitted_right = False
        
    def _setup_visualization(self):
        """Setup basic visualization elements for dual YAMs."""
        # Add base frames for both arms
        self.base_frame_left = self.server.scene.add_frame("/base_left", show_axes=False, axes_length=0.2)
        self.base_frame_right = self.server.scene.add_frame("/base_right", show_axes=False, axes_length=0.2)
        
        # Add robot URDFs for both arms
        self.urdf_vis_left = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base_left"
        )
        self.urdf_vis_right = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base_right"
        )
        
        # Update initial configurations
        self.urdf_vis_left.update_cfg(self.rest_pose)
        self.urdf_vis_right.update_cfg(self.rest_pose)
        
        # Add ground grid
        self.server.scene.add_grid("ground", width=4, height=4, cell_size=0.1)
        
    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing displays
        self.timing_handle_left = self.server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        self.timing_handle_right = self.server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)
        
        # Add gizmo size control
        self.tf_size_handle = self.server.gui.add_slider(
            "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.15
        )
        
        # Add TCP offset control
        self.use_tcp_offset_handle = self.server.gui.add_checkbox("Use TCP Offset", initial_value=True)
        
        # Add base position controls
        with self.server.gui.add_folder("Base Positions"):
            self.left_x_handle = self.server.gui.add_slider(
                "Left Base X", min=-2.0, max=2.0, step=0.1, initial_value=self.left_base_xy[0]
            )
            self.left_y_handle = self.server.gui.add_slider(
                "Left Base Y", min=-2.0, max=2.0, step=0.1, initial_value=self.left_base_xy[1]
            )
            self.right_x_handle = self.server.gui.add_slider(
                "Right Base X", min=-2.0, max=2.0, step=0.1, initial_value=self.right_base_xy[0]
            )
            self.right_y_handle = self.server.gui.add_slider(
                "Right Base Y", min=-2.0, max=2.0, step=0.1, initial_value=self.right_base_xy[1]
            )
        
        # Setup base position update callbacks
        @self.left_x_handle.on_update
        def update_left_x(_):
            self._update_base_positions()
            
        @self.left_y_handle.on_update
        def update_left_y(_):
            self._update_base_positions()
            
        @self.right_x_handle.on_update
        def update_right_x(_):
            self._update_base_positions()
            
        @self.right_y_handle.on_update
        def update_right_y(_):
            self._update_base_positions()

    def _update_base_positions(self):
        """Update base positions based on GUI sliders."""
        # Update base poses
        self.base_pose_left = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), 
            jnp.array([self.left_x_handle.value, self.left_y_handle.value, 0.0])
        )
        self.base_pose_right = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), 
            jnp.array([self.right_x_handle.value, self.right_y_handle.value, 0.0])
        )
        
        # Update base frame positions
        self.base_frame_left.position = onp.array(self.base_pose_left.translation())
        self.base_frame_right.position = onp.array(self.base_pose_right.translation())
      
    def _setup_transform_handles(self):
        """Setup transform handles for both YAM end effectors."""
        # Main transform controls for both arms
        self.transform_handles = {
            'left': TransformHandle(
                frame=self.server.scene.add_frame(
                    "/tf_left",
                    axes_length=0.3 * self.tf_size_handle.value,
                    axes_radius=0.01 * self.tf_size_handle.value,
                    origin_radius=0.05 * self.tf_size_handle.value,
                ),
                control=self.server.scene.add_transform_controls(
                    "/base_left/target_left",
                    scale=self.tf_size_handle.value,
                    position=(0.3, 0.0, 0.4),
                    wxyz=(1, 0, 0, 0)
                )
            ),
            'right': TransformHandle(
                frame=self.server.scene.add_frame(
                    "/tf_right",
                    axes_length=0.3 * self.tf_size_handle.value,
                    axes_radius=0.01 * self.tf_size_handle.value,
                    origin_radius=0.05 * self.tf_size_handle.value,
                ),
                control=self.server.scene.add_transform_controls(
                    "/base_right/target_right",
                    scale=self.tf_size_handle.value,
                    position=(0.3, 0.0, 0.4),
                    wxyz=(1, 0, 0, 0)
                )
            )
        }
        
        # TCP offset frames (180 degrees around x-axis, -0.1m in z)
        self.tcp_frames = {
            'left': self.server.scene.add_frame(
                "/base_left/target_left/tcp_left", 
                show_axes=False, 
                position=(0.0, 0.0, -0.1), 
                # wxyz=(0, 1, 0, 0)  # 180 degrees around x-axis
            ),
            'right': self.server.scene.add_frame(
                "/base_right/target_right/tcp_right", 
                show_axes=False, 
                position=(0.0, 0.0, -0.1), 
                # wxyz=(0, 1, 0, 0)  # 180 degrees around x-axis
            )
        }
        
        # Update transform handles when size changes
        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
                handle.frame.axes_length = 0.3 * self.tf_size_handle.value
                handle.frame.axes_radius = 0.01 * self.tf_size_handle.value
                handle.frame.origin_radius = 0.05 * self.tf_size_handle.value
                
    def update_visualization(self):
        """Update visualization with current state for both arms."""
        # Update base frames
        if hasattr(self, 'base_frame_left'):
            self.base_frame_left.position = onp.array(self.base_pose_left.translation())
            self.base_frame_left.wxyz = onp.array(self.base_pose_left.rotation().wxyz)
        if hasattr(self, 'base_frame_right'):
            self.base_frame_right.position = onp.array(self.base_pose_right.translation())
            self.base_frame_right.wxyz = onp.array(self.base_pose_right.rotation().wxyz)
        
        # Update robot configurations
        self.urdf_vis_left.update_cfg(onp.array(self.joints_left))
        self.urdf_vis_right.update_cfg(onp.array(self.joints_right))
        
        # Update end-effector frames
        if hasattr(self, 'transform_handles'):
            # Left arm
            link_poses_left = self.robot_left.forward_kinematics(jnp.array(self.joints_left))
            target_link_idx = self.robot_left.links.names.index(self.target_names[0])
            T_target_world_left = self.base_pose_left @ jaxlie.SE3(link_poses_left[target_link_idx])
            self.transform_handles['left'].frame.position = onp.array(T_target_world_left.translation())
            self.transform_handles['left'].frame.wxyz = onp.array(T_target_world_left.rotation().wxyz)
            
            # Right arm
            link_poses_right = self.robot_right.forward_kinematics(jnp.array(self.joints_right))
            T_target_world_right = self.base_pose_right @ jaxlie.SE3(link_poses_right[target_link_idx])
            self.transform_handles['right'].frame.position = onp.array(T_target_world_right.translation())
            self.transform_handles['right'].frame.wxyz = onp.array(T_target_world_right.rotation().wxyz)
            
    def solve_ik(self, target_positions=None, target_wxyzs=None):
        """Solve inverse kinematics for both YAM arms using PyRoki."""
        if not self.minimal and hasattr(self, 'transform_handles'):
            # Get target poses from transform controls with TCP offset
            use_tcp_offset = getattr(self.use_tcp_offset_handle, 'value', True) if hasattr(self, 'use_tcp_offset_handle') else True
            
            # Process left arm - targets are already in left base frame since controls are parented to /base_left
            left_handle = self.transform_handles['left']
            if hasattr(self, 'tcp_frames') and use_tcp_offset:
                # Compute TCP transform
                handle_transform = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(left_handle.control.wxyz, -1)), 
                    left_handle.control.position
                )
                tcp_frame = self.tcp_frames['left']
                tcp_offset = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(tcp_frame.wxyz, -1)), 
                    tcp_frame.position
                )
                tcp_transform = handle_transform @ tcp_offset
                left_target_position = tcp_transform.wxyz_xyz[-3:]
                left_target_wxyz = tcp_transform.wxyz_xyz[:4]
            else:
                left_target_position = left_handle.control.position
                left_target_wxyz = left_handle.control.wxyz
            
            # Process right arm - targets are already in right base frame since controls are parented to /base_right
            right_handle = self.transform_handles['right']
            if hasattr(self, 'tcp_frames') and use_tcp_offset:
                # Compute TCP transform
                handle_transform = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(right_handle.control.wxyz, -1)), 
                    right_handle.control.position
                )
                tcp_frame = self.tcp_frames['right']
                tcp_offset = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(tcp_frame.wxyz, -1)), 
                    tcp_frame.position
                )
                tcp_transform = handle_transform @ tcp_offset
                right_target_position = tcp_transform.wxyz_xyz[-3:]
                right_target_wxyz = tcp_transform.wxyz_xyz[:4]
            else:
                right_target_position = right_handle.control.position
                right_target_wxyz = right_handle.control.wxyz
                
        elif target_positions is None or target_wxyzs is None:
            # For minimal mode, use default targets if not provided (in base frame coordinates)
            left_target_position = onp.array([0.3, 0.0, 0.4])
            left_target_wxyz = onp.array([1.0, 0.0, 0.0, 0.0])
            right_target_position = onp.array([0.3, 0.0, 0.4])
            right_target_wxyz = onp.array([1.0, 0.0, 0.0, 0.0])
        
        # Solve IK for left arm
        timing_start_left = False
        if hasattr(self, 'has_jitted_left') and not self.has_jitted_left:
            start_time = time.time()
            timing_start_left = True
            
        prev_cfg_left = self.joints_left if hasattr(self, 'joints_left') else self.rest_pose
        
        # Targets are already in left robot base frame - no transformation needed
        self.joints_left = solve_ik_with_multiple_targets(
            robot=self.robot_left,
            target_link_names=self.target_names,
            target_positions=onp.array([left_target_position]),
            target_wxyzs=onp.array([left_target_wxyz]),
            prev_cfg=prev_cfg_left,
        )
        
        if timing_start_left:
            elapsed_time = (time.time() - start_time) * 1000
            if hasattr(self, 'timing_handle_left'):
                self.timing_handle_left.value = elapsed_time
            logger.info("Left arm JIT compile + running took {} ms.", elapsed_time)
            self.has_jitted_left = True
        
        # Solve IK for right arm
        timing_start_right = False
        if hasattr(self, 'has_jitted_right') and not self.has_jitted_right:
            start_time = time.time()
            timing_start_right = True
            
        prev_cfg_right = self.joints_right if hasattr(self, 'joints_right') else self.rest_pose
        
        # Targets are already in right robot base frame - no transformation needed
        self.joints_right = solve_ik_with_multiple_targets(
            robot=self.robot_right,
            target_link_names=self.target_names,
            target_positions=onp.array([right_target_position]),
            target_wxyzs=onp.array([right_target_wxyz]),
            prev_cfg=prev_cfg_right,
        )
        
        if timing_start_right:
            elapsed_time = (time.time() - start_time) * 1000
            if hasattr(self, 'timing_handle_right'):
                self.timing_handle_right.value = elapsed_time
            logger.info("Right arm JIT compile + running took {} ms.", elapsed_time)
            self.has_jitted_right = True
    
    def home(self):
        """Reset both arms to rest pose."""
        self.joints_left = self.rest_pose.copy()
        self.joints_right = self.rest_pose.copy()
        
    def run(self):
        """Main run loop."""
        while True:
            if not self.minimal:
                self.solve_ik()
            self.update_visualization()
            time.sleep(0.01) 

    def update_cfg(self, cfg_left, cfg_right):
        """Update the robot configurations."""
        self.joints_left = cfg_left
        self.joints_right = cfg_right
        self.urdf_vis_left.update_cfg(onp.array(self.joints_left))
        self.urdf_vis_right.update_cfg(onp.array(self.joints_right))


if __name__ == "__main__":
    yams_interface = YAMSBaseInterface(
        left_base_xy=(-0.6, 0.0),
        right_base_xy=(0.6, 0.0)
    )
    yams_interface.run()