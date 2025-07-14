#!/usr/bin/env python3
"""
Test script to verify sequential frame loading logic with fake data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.openpi.training.world_model_training.data_loader import (
    WorldModelDataConfig, 
    FakeWorldModelDataset,
    WorldModelDataset
)

def test_fake_sequential_frames():
    """Test that the fake dataset creates different frames correctly."""
    
    # Create config
    config = WorldModelDataConfig(
        repo_id=None,  # Not needed for fake dataset
        num_frames=4,
        frame_skip=1,
        image_size=(112, 112),
        mask_ratio=0.5,
        min_episode_length=10,
        chunk_size=100,
    )
    
    # Create fake dataset
    dataset = FakeWorldModelDataset(
        config=config,
        size=10,
    )
    
    print(f"Fake dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        print(f"\n--- Testing fake sample {i} ---")
        
        try:
            model_input, model_output = dataset[i]
            
            video_frames = model_input.video_frames
            print(f"Video frames shape: {video_frames.shape}")
            print(f"Expected shape: ({config.num_frames}, {config.image_size[0]}, {config.image_size[1]}, 3)")
            
            # Check if frames are different (not identical)
            frames_different = False
            for j in range(1, video_frames.shape[0]):
                if not np.allclose(video_frames[j], video_frames[0], atol=1e-6):
                    frames_different = True
                    break
            
            if frames_different:
                print("✅ Frames are different - sequential loading is working!")
                
                # Calculate frame differences
                frame_diffs = []
                for j in range(1, video_frames.shape[0]):
                    diff = np.mean(np.abs(video_frames[j] - video_frames[j-1]))
                    frame_diffs.append(diff)
                
                print(f"Frame differences (mean abs diff): {frame_diffs}")
                
            else:
                print("❌ All frames are identical - this indicates a problem!")
                
                # Save debug images
                save_debug_images(video_frames, f"fake_sample_{i}_frames")
                
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    print("\n" + "="*50)
    print("Fake dataset test completed!")

def test_real_dataset_with_different_episode():
    """Test with a different episode that might work better."""
    
    # Create config
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=1,
        image_size=(112, 112),
        mask_ratio=0.5,
        min_episode_length=10,
        chunk_size=100,
    )
    
    # Create dataset
    dataset = WorldModelDataset(
        config=config,
        split="train",
        shuffle=False,  # Don't shuffle for testing
        current_step=0,
        chunk_size=100,
    )
    
    print(f"Real dataset length: {len(dataset)}")
    
    # Try different indices to find a working episode
    test_indices = [100, 500, 1000, 5000, 10000]
    
    for idx in test_indices:
        if idx >= len(dataset):
            continue
            
        print(f"\n--- Testing real sample {idx} ---")
        
        try:
            sample = dataset[idx]
            if isinstance(sample, tuple):
                model_input, model_output = sample
            else:
                # Multi-view mode returns a list
                model_input, model_output = sample[0]
            
            video_frames = model_input.video_frames
            print(f"Video frames shape: {video_frames.shape}")
            
            # Check if frames are different (not identical)
            frames_different = False
            for j in range(1, video_frames.shape[0]):
                if not np.allclose(video_frames[j], video_frames[0], atol=1e-6):
                    frames_different = True
                    break
            
            if frames_different:
                print("✅ Frames are different - sequential loading is working!")
                
                # Calculate frame differences
                frame_diffs = []
                for j in range(1, video_frames.shape[0]):
                    diff = np.mean(np.abs(video_frames[j] - video_frames[j-1]))
                    frame_diffs.append(diff)
                
                print(f"Frame differences (mean abs diff): {frame_diffs}")
                break
                
            else:
                print("❌ All frames are identical - trying next index...")
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            continue
    
    print("\n" + "="*50)
    print("Real dataset test completed!")

def save_debug_images(video_frames, prefix):
    """Save debug images to visualize the frames."""
    try:
        # Convert to uint8 for saving
        frames_uint8 = np.clip(video_frames * 255, 0, 255).astype(np.uint8)
        
        # Create a grid of images
        fig, axes = plt.subplots(1, video_frames.shape[0], figsize=(3*video_frames.shape[0], 3))
        if video_frames.shape[0] == 1:
            axes = [axes]
        
        for i, frame in enumerate(frames_uint8):
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_debug.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug images saved to {prefix}_debug.png")
        
    except Exception as e:
        print(f"Error saving debug images: {e}")

if __name__ == "__main__":
    print("Testing fake dataset...")
    test_fake_sequential_frames()
    
    print("\n" + "="*50)
    print("Testing real dataset...")
    test_real_dataset_with_different_episode() 