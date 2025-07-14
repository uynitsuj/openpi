#!/usr/bin/env python3
"""
Debug script to understand the dataset structure and check if episodes have multiple frames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def debug_dataset_structure():
    """Debug the dataset structure to understand frame availability."""
    
    repo_id = "uynitsuj/hummus_xmi_full_subsample_2_cleaned2"
    
    print(f"Debugging dataset structure: {repo_id}")
    
    # Load dataset metadata
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    print(f"Total episodes: {len(dataset_meta.episodes)}")
    
    # Check a few episodes
    test_episodes = [2648, 1000, 5000, 10000]
    
    for episode_idx in test_episodes:
        if episode_idx >= len(dataset_meta.episodes):
            continue
            
        episode_info = dataset_meta.episodes[episode_idx]
        print(f"\n--- Episode {episode_idx} ---")
        print(f"  Length: {episode_info['length']}")
        print(f"  Keys: {list(episode_info.keys())}")
        
        # Try to load this episode with different timestamps
        try:
            # Test with a single timestamp first
            dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={"state": [0.0]},
            )
            
            episode_data = dataset[episode_idx]
            print(f"  Episode data type: {type(episode_data)}")
            
            if hasattr(episode_data, 'keys'):
                print(f"  Episode data keys: {list(episode_data.keys())}")
                
                # Check if there are camera images
                camera_keys = ["top_camera-images-rgb", "left_camera-images-rgb", "right_camera-images-rgb"]
                for cam_key in camera_keys:
                    if cam_key in episode_data:
                        cam_data = episode_data[cam_key]
                        print(f"  {cam_key} type: {type(cam_data)}")
                        if hasattr(cam_data, 'shape'):
                            print(f"  {cam_key} shape: {cam_data.shape}")
            
            # Test if we can access different timestamps
            print(f"  Testing multiple timestamps...")
            timestamps = [0.0, 0.0667, 0.1333, 0.2]
            frame_means = []
            
            for timestamp in timestamps:
                try:
                    single_dataset = lerobot_dataset.LeRobotDataset(
                        repo_id,
                        delta_timestamps={"state": [timestamp]},
                    )
                    single_data = single_dataset[episode_idx]
                    
                    if "top_camera-images-rgb" in single_data:
                        image = single_data["top_camera-images-rgb"]
                        if isinstance(image, torch.Tensor):
                            image = image.numpy()
                        
                        if len(image.shape) == 3 and image.shape[0] == 3:
                            # (C, H, W) -> (H, W, C)
                            image = np.transpose(image, (1, 2, 0))
                        
                        mean_val = image.mean()
                        frame_means.append(mean_val)
                        print(f"    Timestamp {timestamp:.3f}: mean = {mean_val:.6f}")
                    else:
                        print(f"    Timestamp {timestamp:.3f}: no camera data")
                        
                except Exception as e:
                    print(f"    Timestamp {timestamp:.3f}: error - {e}")
            
            # Check if all frames are the same
            if len(frame_means) > 1:
                frame_diffs = [abs(frame_means[i] - frame_means[i-1]) for i in range(1, len(frame_means))]
                max_diff = max(frame_diffs) if frame_diffs else 0
                print(f"  Max frame difference: {max_diff:.6f}")
                
                if max_diff < 1e-6:
                    print(f"  ⚠️  All frames are identical!")
                else:
                    print(f"  ✅ Frames are different!")
            
        except Exception as e:
            print(f"  Error loading episode {episode_idx}: {e}")

if __name__ == "__main__":
    debug_dataset_structure() 