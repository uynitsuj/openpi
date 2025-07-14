#!/usr/bin/env python3
"""
Debug script to properly access video frames from LeRobot dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def debug_lerobot_video():
    """Debug video frame access from LeRobot dataset."""
    
    repo_id = "uynitsuj/hummus_xmi_full_subsample_2_cleaned2"
    episode_idx = 2648
    
    print(f"Debugging video frame access for episode {episode_idx}")
    
    # Try different approaches to access video frames
    
    # Approach 1: Use multiple timestamps in a single dataset call
    print("\n--- Approach 1: Multiple timestamps in single dataset ---")
    try:
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={"state": [0.0, 0.0667, 0.1333, 0.2]},
        )
        
        episode_data = dataset[episode_idx]
        print(f"Episode data type: {type(episode_data)}")
        
        if hasattr(episode_data, '__len__'):
            print(f"Episode data length: {len(episode_data)}")
            
            # Check if we can access different frames
            for i in range(min(4, len(episode_data))):
                frame_data = episode_data[i]
                print(f"  Frame {i} type: {type(frame_data)}")
                
                if hasattr(frame_data, 'keys'):
                    print(f"  Frame {i} keys: {list(frame_data.keys())}")
                    
                    if "top_camera-images-rgb" in frame_data:
                        image = frame_data["top_camera-images-rgb"]
                        if isinstance(image, torch.Tensor):
                            image = image.numpy()
                        
                        if len(image.shape) == 3 and image.shape[0] == 3:
                            image = np.transpose(image, (1, 2, 0))
                        
                        mean_val = image.mean()
                        print(f"  Frame {i} mean: {mean_val:.6f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Approach 2: Use frame-based access
    print("\n--- Approach 2: Frame-based access ---")
    try:
        # Try accessing with frame indices directly
        frame_indices = [0, 1, 2, 3]
        
        for frame_idx in frame_indices:
            try:
                # Create dataset for this specific frame
                dataset = lerobot_dataset.LeRobotDataset(
                    repo_id,
                    delta_timestamps={"state": [frame_idx * 0.0667]},  # 15 FPS
                )
                
                frame_data = dataset[episode_idx]
                print(f"  Frame {frame_idx} data type: {type(frame_data)}")
                
                if hasattr(frame_data, 'keys') and "top_camera-images-rgb" in frame_data:
                    image = frame_data["top_camera-images-rgb"]
                    if isinstance(image, torch.Tensor):
                        image = image.numpy()
                    
                    if len(image.shape) == 3 and image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                    
                    mean_val = image.mean()
                    print(f"  Frame {frame_idx} mean: {mean_val:.6f}")
                    
            except Exception as e:
                print(f"  Error loading frame {frame_idx}: {e}")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    # Approach 3: Check if dataset supports video access
    print("\n--- Approach 3: Check dataset capabilities ---")
    try:
        # Load dataset metadata
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        episode_info = dataset_meta.episodes[episode_idx]
        
        print(f"Episode length: {episode_info['length']}")
        print(f"Episode keys: {list(episode_info.keys())}")
        
        # Try to understand the data format
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={"state": [0.0]},
        )
        
        sample_data = dataset[episode_idx]
        print(f"Sample data keys: {list(sample_data.keys())}")
        
        # Check if there are any video-related fields
        video_keys = [key for key in sample_data.keys() if 'video' in key.lower() or 'mp4' in key.lower()]
        print(f"Video-related keys: {video_keys}")
        
        # Check if there are any temporal fields
        temporal_keys = [key for key in sample_data.keys() if 'time' in key.lower() or 'frame' in key.lower()]
        print(f"Temporal-related keys: {temporal_keys}")
        
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    debug_lerobot_video() 