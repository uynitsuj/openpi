import numpy as np
from typing import List
from openpi.utils.matrix_utils import rot_6d_to_quat
import viser.transforms as vtf
from collections import deque

def zed_tf_intrinsics(factory_intrinsics: np.ndarray, capture_resolution: tuple[int, int], new_resolution: tuple[int, int], crop_method: str = "center_crop") -> np.ndarray:
    """
    Transform ZED factory intrinsics to capture resolution.
    Args:
        factory_intrinsics: The factory intrinsics. Shape: (3, 3).
        capture_resolution: The capture resolution. Width x Height.
        new_resolution: The new resolution (after cropping or padding).
        crop_method: The crop method. Currently supported: "center_crop".
    """
    intrinsics = factory_intrinsics.copy()
    if crop_method == "center_crop":
        new_resolution = (round(capture_resolution[0] / (min(capture_resolution) / new_resolution[0])), round(capture_resolution[1] / (min(capture_resolution) / new_resolution[1])))
        # Pre-center crop intrinsics
    intrinsics[0, 2] = intrinsics[0, 2] * new_resolution[1] / capture_resolution[1]
    intrinsics[1, 2] = intrinsics[1, 2] * new_resolution[0] / capture_resolution[0]
    intrinsics[0, 0] = intrinsics[0, 0] * new_resolution[1] / capture_resolution[1]
    intrinsics[1, 1] = intrinsics[1, 1] * new_resolution[0] / capture_resolution[0]
    if crop_method == "center_crop":
        crop_resolution = (min(new_resolution), min(new_resolution))

        crop_x = (new_resolution[0] - crop_resolution[1]) / 2
        crop_y = (new_resolution[1] - crop_resolution[0]) / 2
        intrinsics[0, 2] -= crop_x
        intrinsics[1, 2] -= crop_y

    # TODO: implement padded crop

    return intrinsics

def select_keyframes_helper(head_trajectory: np.ndarray, head_camera_fov: float, past_idxs: deque, criterion: str = "cost_select") -> List[int]:
    """
    Helper function to select keyframes based on a given criterion.

    Args:
        head_trajectory: The head trajectory (causal and index -1 is current head pose). Shape: (N, 9) where N is the number of head poses. Each row is [6D rotation, 3D position]. Should not include future head poses to respect causality when selecting keyframes.
        head_camera_fov: The head camera fov (in radians).
        criterion: The criterion to use for selecting keyframes. Currently supported: "frustum_overlap".

    Returns:
        The indices of the selected keyframes. (backwards indexed; -1 for the last keyframe, -2 for the second last keyframe, etc.)
    """
    if criterion == "cost_select":
        selected_idx = cost_select(head_trajectory)
        
        if len(past_idxs) > 0:
            selected_pose = head_trajectory[len(head_trajectory)-selected_idx]
            selected_pose_z_vec = vtf.SO3(wxyz=rot_6d_to_quat(selected_pose[:6])).as_matrix()[-1, :, 2]
            selected_pose_pos = selected_pose[6:9]

            past_idxs_z_vecs = [vtf.SO3(wxyz=rot_6d_to_quat(head_trajectory[idx][:6])).as_matrix()[-1, :, 2] for idx in past_idxs]
            past_idxs_pos = [head_trajectory[idx][6:9] for idx in past_idxs]

            angles_all = [np.arccos(np.clip(np.dot(selected_pose_z_vec, z_vec) / (np.linalg.norm(selected_pose_z_vec) * np.linalg.norm(z_vec)), -1, 1)) for z_vec in past_idxs_z_vecs]
            dists_all = [np.linalg.norm(selected_pose_pos - pos) for pos in past_idxs_pos]

            substantial_angle_delta = angles_all[-1] > head_camera_fov * 0.45
            substantial_dist_delta = dists_all[-1] > 0.4

        if len(past_idxs) == 0:
            past_idxs.append(len(head_trajectory)-selected_idx)
        else:
            if len(head_trajectory)-selected_idx > past_idxs[-1]:
                if substantial_angle_delta or substantial_dist_delta:
                    past_idxs.append(len(head_trajectory)-selected_idx) # Treat select_keyframes_helper as a candidate selector, and keep past idx monotonically increasing

        return past_idxs
    else:
        raise ValueError(f"Invalid criterion: {criterion}")

def cost_select(head_trajectory: np.ndarray, lookback_time_s: float = 2.0, timestep_weight: float = 0.01, angle_weight: float = 30.0, velocity_weight: float = 2600.0) -> List[int]:
    """
    Select keyframes based on cost heuristic.
    """
    if len(head_trajectory) > int(30 * lookback_time_s):
        head_trajectory = head_trajectory[-int(30 * lookback_time_s):] # past lookback_time_s seconds under consideration

    orig_len = len(head_trajectory)
    if len(head_trajectory) == 0:
        raise ValueError("Head trajectory is empty")

    if len(head_trajectory) == 1:
        return 1

    head_quats = vtf.SO3(wxyz=rot_6d_to_quat(head_trajectory[:, :6]))

    current_head_z_vec = head_quats.as_matrix()[-1, :, 2]

    previous_head_z_vecs = head_quats.as_matrix()[:-1, :, 2]

    # Larger angle difference is better
    thetas = [np.arccos(np.clip(np.dot(z_vec, current_head_z_vec) / (np.linalg.norm(z_vec) * np.linalg.norm(current_head_z_vec)), -1, 1)) for z_vec in previous_head_z_vecs]
    theta_costs = [(1/(theta+1e-6))*angle_weight for theta in thetas]


    # Timesteps closer to current time are better
    timestep_costs = [np.exp(np.abs(i - len(head_trajectory) + 1) * timestep_weight) for i in range(len(head_trajectory) - 1)]

    # Lower angular velocity is better
    delta_thetas = [np.arccos(np.clip(np.dot(head_quats.as_matrix()[i-1, 2], head_quats.as_matrix()[i, 2]) / (np.linalg.norm(head_quats.as_matrix()[i-1, 2]) * np.linalg.norm(head_quats.as_matrix()[i, 2])), -1, 1)) for i in range(1, len(head_quats.wxyz))]
    delta_theta_costs = [np.exp(delta_theta*velocity_weight) for delta_theta in delta_thetas]

    costs = [theta_costs[i] + timestep_costs[i] + delta_theta_costs[i] for i in range(len(head_trajectory) - 1)]


    select = np.argsort(costs)[0]

    return orig_len - select

