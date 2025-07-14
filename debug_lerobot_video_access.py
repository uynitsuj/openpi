#!/usr/bin/env python3
"""
Debug script to explore video access patterns in LeRobot dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def debug_video_access():
    """Debug different video access patterns."""
    
    repo_id = "uynitsuj/hummus_xmi_full_subsample_2_cleaned2"
    episode_idx = 2648
    
    print(f"Debugging video access for episode {episode_idx}")
    
    # Check if we can access the raw data differently
    print("\n--- Checking raw data access ---")
    
    try:
        # Load dataset metadata
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        episode_info = dataset_meta.episodes[episode_idx]
        
        print(f"Episode info: {episode_info}")
        
        # Try to access the dataset with different configurations
        print("\n--- Trying different dataset configurations ---")
        
        # Configuration 1: No timestamps (raw access)
        try:
            dataset1 = lerobot_dataset.LeRobotDataset(repo_id)
            data1 = dataset1[episode_idx]
            print(f"Config 1 (no timestamps): {type(data1)}")
            if hasattr(data1, 'keys'):
                print(f"  Keys: {list(data1.keys())}")
        except Exception as e:
            print(f"  Config 1 error: {e}")
        
        # Configuration 2: Single timestamp
        try:
            dataset2 = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={"state": [0.0]},
            )
            data2 = dataset2[episode_idx]
            print(f"Config 2 (single timestamp): {type(data2)}")
            if hasattr(data2, 'keys'):
                print(f"  Keys: {list(data2.keys())}")
        except Exception as e:
            print(f"  Config 2 error: {e}")
        
        # Configuration 3: Multiple timestamps
        try:
            dataset3 = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={"state": [0.0, 0.0667]},
            )
            data3 = dataset3[episode_idx]
            print(f"Config 3 (multiple timestamps): {type(data3)}")
            if hasattr(data3, '__len__'):
                print(f"  Length: {len(data3)}")
                for i in range(min(2, len(data3))):
                    frame = data3[i]
                    print(f"  Frame {i} keys: {list(frame.keys()) if hasattr(frame, 'keys') else 'No keys'}")
        except Exception as e:
            print(f"  Config 3 error: {e}")
        
        # Configuration 4: Try to access with frame indices
        print("\n--- Trying frame-based access ---")
        try:
            # Try to access the dataset with frame indices
            for frame_idx in range(4):
                timestamp = frame_idx * 0.0667  # 15 FPS
                
                dataset = lerobot_dataset.LeRobotDataset(
                    repo_id,
                    delta_timestamps={"state": [timestamp]},
                )
                
                frame_data = dataset[episode_idx]
                
                if "top_camera-images-rgb" in frame_data:
                    image = frame_data["top_camera-images-rgb"]
                    if isinstance(image, torch.Tensor):
                        image = image.numpy()
                    
                    if len(image.shape) == 3 and image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                    
                    mean_val = image.mean()
                    print(f"  Frame {frame_idx} (timestamp {timestamp:.3f}): mean = {mean_val:.6f}")
                    
                    # Check if this frame is different from the first frame
                    if frame_idx == 0:
                        first_frame = image.copy()
                    else:
                        diff = np.abs(image - first_frame).mean()
                        print(f"    Difference from frame 0: {diff:.6f}")
                        
        except Exception as e:
            print(f"  Frame access error: {e}")
        
        # Configuration 5: Check if there are video files in the dataset
        print("\n--- Checking for video files ---")
        try:
            # Try to find video-related data
            dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={"state": [0.0]},
            )
            
            sample_data = dataset[episode_idx]
            
            # Look for any video-related fields
            for key, value in sample_data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"    Length: {len(value)}")
                    
        except Exception as e:
            print(f"  Video file check error: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_video_access() 