"""
Minimal example script for converting a dataset to LeRobot format.

We use the XMI dataset for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run openpi/scripts/xmi/convert_xmi_data.py

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run openpi/scripts/xmi/convert_xmi_data.py --push_to_hub
"""

import os 
os.environ["HF_LEROBOT_HOME"] = "/home/justinyu/nfs_us/justinyu/xmi_lerobot_datasets"
import shutil
import h5py 
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm, trange
import zarr
from PIL import Image
from openpi_client.image_tools import resize_with_pad
from pathlib import Path
import viser.transforms as vtf
import tyro
from openpi.utils.xmi_dataloader_utils import load_episode_data
from openpi.utils.matrix_utils import *

RAW_DATASET_FOLDERS = [
    "/home/justinyu/nfs_us/justinyu/20250611/20250611",
]
LANGUAGE_INSTRUCTIONS = [
    "testing"
]

REPO_NAME = "uynitsuj/xmi_bimanual_testing"  # Name of the output dataset, also used for the Hugging Face Hub

# Define camera keys and mappings
CAMERA_KEYS = [
    "left_camera-images-rgb",
    "right_camera-images-rgb", 
    "top_camera-images-rgb"
]

CAMERA_KEY_MAPPING = {
    "left_camera-images-rgb": "exterior_image_1_left",
    "right_camera-images-rgb": "exterior_image_2_right", 
    "top_camera-images-rgb": "exterior_image_3_top"
}

RESIZE_SIZE = 224
UP_TO_N_TRAJ = None
# UP_TO_N_TRAJ = 1100

def main(
    left_controller_calib: str = "/home/justinyu/Left_Controller_20250603_15/calib_results/controller2franka.npy",
    right_controller_calib: str = "/home/justinyu/Right_Controller_20250603_15/calib_results/controller2franka.npy",
    push_to_hub: bool = False,
    debug: bool = False,
    ):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    print("Dataset saved to ", output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset_features = {
        "state": {
            "dtype": "float32",
            "shape": (20,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32", 
            "shape": (20,),
            "names": ["actions"],
        },
    }
    
    # Add camera features dynamically based on available cameras
    for camera_key in CAMERA_KEYS:
        if camera_key in CAMERA_KEY_MAPPING:
            dataset_features[CAMERA_KEY_MAPPING[camera_key]] = {
                "dtype": "video",
                "shape": (RESIZE_SIZE, RESIZE_SIZE, 3),
                "names": ["height", "width", "channel"],
            }
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xmi",
        fps=30,
        features=dataset_features,
        image_writer_threads=20,
        image_writer_processes=10,
    )

    q2w = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, 0.0), np.array([0.0, 0.0, 0.0])
        )

    # Loop over raw XMI datasets and write episodes to the LeRobot dataset
    for raw_dataset_name, language_instruction in zip(RAW_DATASET_FOLDERS, LANGUAGE_INSTRUCTIONS):
        print("Processing folder: ", raw_dataset_name)
        trajs = [d for d in os.listdir(raw_dataset_name) if os.path.isdir(os.path.join(raw_dataset_name, d))]
        
        if UP_TO_N_TRAJ is not None:
            trajs = trajs[:UP_TO_N_TRAJ]
            
        for idx, task in tqdm(enumerate(trajs), desc="Processing trajectories"):
            print(f"Trajectory {idx}/{len(trajs)}: {task} is being processed")
            task_folder = Path(raw_dataset_name) / task
            
            # Load episode data using our data loader
            episode_data = load_episode_data(task_folder)
            
            # Extract relevant data
            joint_data = episode_data['joint_data']
            action_data = episode_data['action_data']
            images = episode_data['images']

            # head
            # Determine direction that head z axis is pointing in the first frame to reorient the RBY1 base frame (for proprio world frame normalization)
            head_z_tf = vtf.SE3.from_matrix(action_data["action-left-head"][0])
            head_data_all = vtf.SE3.from_matrix(action_data["action-left-head"])

            head_data_all = q2w @ head_data_all
            head_z_tf = q2w @ head_z_tf

            # average head height
            head_height = np.mean(head_data_all.wxyz_xyz[:, -1])
            print(f"Average head height: {head_height}m")

            head_translation = np.array([head_z_tf.translation()[0], -head_z_tf.translation()[1], 0.0])

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
            rby1_base_frame_position = head_translation

            # left hand
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

            left_controller_calib_tf = vtf.SE3.from_matrix(np.load(left_controller_calib)).inverse()

            tf_left_ee_ik_target = left_hand_tf_reflected @ left_controller_calib_tf @ ee_tf

            left_ee_ik_target_handle_position = tf_left_ee_ik_target.wxyz_xyz[:, -3:]
            left_ee_ik_target_handle_wxyz = tf_left_ee_ik_target.wxyz_xyz[:, :4]


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

            right_controller_calib_tf = vtf.SE3.from_matrix(np.load(right_controller_calib)).inverse()

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

            if debug:
                # Uncomment below for visualization debugging
                import viser
                viser_server = viser.ViserServer()
                for i in range(len(right_ee_ik_target_handle_position)):
                    viser_server.scene.add_frame(
                        f"right_hand_tf/tf_{i}",
                        position = right_ee_ik_target_handle_position[i],
                        wxyz = right_ee_ik_target_handle_wxyz[i],
                        axes_length = 0.02,
                        axes_radius = 0.0003,
                    )
                    viser_server.scene.add_frame(
                        f"left_hand_tf/tf_{i}",
                        position = left_ee_ik_target_handle_position[i],
                        wxyz = left_ee_ik_target_handle_wxyz[i],
                        axes_length = 0.02,
                        axes_radius = 0.0003,
                    )
                viser_server.scene.add_frame(
                    "rby1_base_frame",
                    position = rby1_base_frame_position,
                    wxyz = rby1_base_frame_wxyz,
                    axes_length = 0.15,
                    axes_radius = 0.004,
                )

            left_gripper_pos = action_data['action-left-pos']
            right_gripper_pos = action_data['action-right-pos']

            # Check array lengths are consistent  
            # Note: removing the joint_pos checks since we're using end-effector data now
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
            
            # Combine into full end-effector state
            # FORMAT : [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
            proprio_data = np.concatenate([
                left_6d_rot, left_ee_ik_target_handle_position, left_gripper_pos,
                right_6d_rot, right_ee_ik_target_handle_position, right_gripper_pos
            ], axis=1)

            # Convert the proprio_data back to SE3 transforms for verification
            # FORMAT: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
            # Extract 6D rotations and positions
            left_6d_rot_recovered = proprio_data[:, 0:6]    # indices 0:6
            left_ee_xyz_recovered = proprio_data[:, 6:9]    # indices 6:9  
            left_gripper_recovered = proprio_data[:, 9:10]  # index 9
            right_6d_rot_recovered = proprio_data[:, 10:16] # indices 10:16
            right_ee_xyz_recovered = proprio_data[:, 16:19] # indices 16:19
            right_gripper_recovered = proprio_data[:, 19:20] # index 19
            
            # Convert 6D rotations back to quaternions using matrix_utils
            left_ee_quat_recovered = rot_6d_to_quat(left_6d_rot_recovered)  # Returns [w, x, y, z]
            right_ee_quat_recovered = rot_6d_to_quat(right_6d_rot_recovered)  # Returns [w, x, y, z]
            
            # Debug print shapes and formats
            # print(f"Original left quaternion shape: {left_ee_ik_target_handle_wxyz.shape}")
            # print(f"Recovered left quaternion shape: {left_ee_quat_recovered.shape}")
            # print(f"Original left quaternion sample: {left_ee_ik_target_handle_wxyz[0]}")
            # print(f"Recovered left quaternion sample: {left_ee_quat_recovered[0]}")
            
            # Convert to wxyz format for viser (already in [w, x, y, z] format)
            left_ee_wxyz_recovered = left_ee_quat_recovered  # Already in [w, x, y, z] format
            right_ee_wxyz_recovered = right_ee_quat_recovered  # Already in [w, x, y, z] format
            
            # Create SE3 transforms for visualization
            left_ee_tf_recovered = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(wxyz=left_ee_wxyz_recovered), 
                left_ee_xyz_recovered
            )
            right_ee_tf_recovered = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(wxyz=right_ee_wxyz_recovered), 
                right_ee_xyz_recovered
            )

            if debug:
                # Add recovered transforms to viser for comparison
                for i in range(len(right_ee_xyz_recovered)):
                    viser_server.scene.add_frame(
                        f"right_hand_tf_recovered/tf_{i}",
                        position = right_ee_xyz_recovered[i],
                        wxyz = right_ee_wxyz_recovered[i], 
                        axes_length = 0.015,
                        axes_radius = 0.0002,
                    )
                    viser_server.scene.add_frame(
                        f"left_hand_tf_recovered/tf_{i}",
                        position = left_ee_xyz_recovered[i],
                        wxyz = left_ee_wxyz_recovered[i],
                        axes_length = 0.015,
                        axes_radius = 0.0002,
                    )

            # Adjust sequence length to match available data
            if images:
                # Use minimum of joint data length and image sequence length
                available_cameras = [cam for cam in CAMERA_KEYS if cam in images]
                if available_cameras:
                    image_seq_length = len(images[available_cameras[0]])
                    seq_length = min(seq_length, image_seq_length)
                    proprio_data = proprio_data[:seq_length]
            
            # We need seq_length - 1 steps since we calculate actions as deltas
            seq_length = seq_length - 1

            frames = {}
            for camera_key in CAMERA_KEYS:
                frames[camera_key] = []

            for step in range(seq_length):
                # Current joint state
                proprio_t = proprio_data[step]

                # Calculate delta action (next state - current state)
                action_t = proprio_data[step + 1] - proprio_t
                
                # For grippers, use absolute position from t+1 instead of delta
                action_t[7] = proprio_data[step + 1][7]    # left gripper
                action_t[15] = proprio_data[step + 1][15]  # right gripper
                
                # Prepare frame data
                frame_data = {
                    "state": proprio_t.astype(np.float32),
                    "actions": action_t.astype(np.float32),
                }
                
                # Add available camera images
                for camera_key in CAMERA_KEYS:
                    if camera_key in images and camera_key in CAMERA_KEY_MAPPING:
                        lerobot_key = CAMERA_KEY_MAPPING[camera_key]
                        if step < len(images[camera_key]):
                            image = images[camera_key][step]
                            if "top" in camera_key:
                                # Take half width left of the image since it's stereo appended
                                image = image[:, :image.shape[1]//2, :]

                            # Resize image
                            resized_image = resize_with_pad(image, RESIZE_SIZE, RESIZE_SIZE)
                            frame_data[lerobot_key] = resized_image
                            if debug:
                                frames[camera_key].append(resized_image)
                
                dataset.add_frame(frame_data, task=language_instruction)

            if debug:
                import imageio.v2 as iio
                for key in frames.keys():
                    os.makedirs(f"debug_videos", exist_ok=True)
                    writer = iio.get_writer(f"debug_videos/{task}_{key}.mp4", fps=30, format="FFMPEG", codec='h264')
                    for frame in frames[key]:
                        writer.append_data(frame)
                    writer.close()
                
            dataset.save_episode()
            
    dataset.push_to_hub(
        tags=["xmi", "rby", "xdof"],
        private=True,
        push_videos=True,
        license="apache-2.0",
    )


    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)
    print("Dataset saved to ", output_path)

if __name__ == "__main__":
    tyro.cli(main)
