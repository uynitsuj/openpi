#!/usr/bin/env python3
"""
Debug script to understand LeRobot dataset structure and frame access.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def debug_lerobot_dataset():
    """Debug the LeRobot dataset structure."""
    
    repo_id = "uynitsuj/hummus_xmi_full_subsample_2_cleaned2"
    
    print(f"Loading LeRobot dataset: {repo_id}")
    
    # Load dataset metadata
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    print(f"Dataset metadata loaded. Total episodes: {len(dataset_meta.episodes)}")
    
    # Get a sample episode
    episode_idx = 2648  # Use the same episode from the test
    episode_info = dataset_meta.episodes[episode_idx]
    print(f"\nEpisode {episode_idx} info:")
    print(f"  Length: {episode_info['length']}")
    print(f"  Keys: {list(episode_info.keys())}")
    
    # Load the actual dataset
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={"state": [0.0, 0.0667, 0.1333, 0.2]},  # 4 timestamps, multiples of 1/15
    )
    
    print(f"\nDataset loaded. Testing episode {episode_idx}...")
    
    try:
        # Get the episode data
        episode_data = dataset[episode_idx]
        print(f"Episode data type: {type(episode_data)}")
        
        if hasattr(episode_data, '__len__'):
            print(f"Episode data length: {len(episode_data)}")
        
        if hasattr(episode_data, 'keys'):
            print(f"Episode data keys: {list(episode_data.keys())}")
        
        # Try to access different frames
        for frame_idx in [0, 1, 2, 3]:
            print(f"\n--- Frame {frame_idx} ---")
            
            try:
                if hasattr(episode_data, '__getitem__'):
                    frame_data = episode_data[frame_idx]
                    print(f"  Frame data type: {type(frame_data)}")
                    
                    if hasattr(frame_data, 'keys'):
                        print(f"  Frame data keys: {list(frame_data.keys())}")
                        
                        # Check for camera data
                        camera_keys = ["top_camera-images-rgb", "left_camera-images-rgb", "right_camera-images-rgb"]
                        for cam_key in camera_keys:
                            if cam_key in frame_data:
                                cam_data = frame_data[cam_key]
                                print(f"  {cam_key} type: {type(cam_data)}")
                                if hasattr(cam_data, 'shape'):
                                    print(f"  {cam_key} shape: {cam_data.shape}")
                                elif hasattr(cam_data, '__len__'):
                                    print(f"  {cam_key} length: {len(cam_data)}")
                else:
                    print(f"  Episode data is not indexable")
                    
            except Exception as e:
                print(f"  Error accessing frame {frame_idx}: {e}")
        
        # Try different access patterns
        print(f"\n--- Alternative access patterns ---")
        
        # Try accessing with different timestamps
        timestamps = [0.0, 0.0667, 0.1333, 0.2]
        for i, timestamp in enumerate(timestamps):
            try:
                # Create a dataset with single timestamp
                single_dataset = lerobot_dataset.LeRobotDataset(
                    repo_id,
                    delta_timestamps={"state": [timestamp]},
                )
                single_data = single_dataset[episode_idx]
                print(f"  Timestamp {timestamp}: data type {type(single_data)}")
                
                if hasattr(single_data, 'keys'):
                    print(f"    Keys: {list(single_data.keys())}")
                    
            except Exception as e:
                print(f"  Error with timestamp {timestamp}: {e}")
        
    except Exception as e:
        print(f"Error loading episode {episode_idx}: {e}")

if __name__ == "__main__":
    debug_lerobot_dataset() 