#!/usr/bin/env python3
"""
Test script to verify that the dataloader loads sequential frames correctly.
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
    WorldModelDataset
)

def test_sequential_frames():
    """Test that the dataloader loads sequential frames correctly."""
    
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
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        print(f"\n--- Testing sample {i} ---")
        
        try:
            sample = dataset[i]
            if isinstance(sample, tuple):
                model_input, model_output = sample
            else:
                # Multi-view mode returns a list
                model_input, model_output = sample[0]
            
            video_frames = model_input.video_frames
            print(f"Video frames shape: {video_frames.shape}")
            print(f"Expected shape: ({config.num_frames}, {config.image_size[0]}, {config.image_size[1]}, 3)")

            # Print frame indices for debug
            frame_indices = None
            if hasattr(dataset, 'episodes') and i < len(dataset.episodes):
                frame_indices = dataset.episodes[i]["frame_indices"]
                print(f"Frame indices for sample {i}: {frame_indices}")
            else:
                print("Could not retrieve frame indices for this sample.")

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
            # Always save debug images for visual inspection
            save_debug_images(video_frames, f"sample_{i}_frames_{frame_indices if frame_indices else 'unknown'}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    print("\n" + "="*50)
    print("Test completed!")

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
    test_sequential_frames() 