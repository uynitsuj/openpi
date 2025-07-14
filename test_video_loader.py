#!/usr/bin/env python3
"""
Test script for the video loader to verify sequential frame extraction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.openpi.training.world_model_training.video_loader import VideoFrameLoader

def test_video_loader():
    """Test the video loader with actual MP4 files."""
    
    # Initialize video loader
    dataset_cache_path = os.path.expanduser("~/.cache/huggingface/lerobot/uynitsuj/hummus_xmi_full_subsample_2_cleaned2")
    video_loader = VideoFrameLoader(dataset_cache_path)
    
    print(f"Testing video loader with cache path: {dataset_cache_path}")
    
    # Test with a few episodes
    test_episodes = [0, 100, 500, 1000]
    
    for episode_idx in test_episodes:
        print(f"\n--- Testing episode {episode_idx} ---")
        
        # Get video info
        video_info = video_loader.get_video_info(episode_idx)
        if video_info:
            print(f"Video info: {video_info}")
        else:
            print(f"No video info found for episode {episode_idx}")
            continue
        
        # Test loading sequential frames
        frame_indices = [0, 1, 2, 3]
        frames = video_loader.load_frames(
            episode_idx=episode_idx,
            frame_indices=frame_indices,
            camera="top_camera-images-rgb",
            target_size=(112, 112)
        )
        
        print(f"Loaded {len(frames)} frames")
        
        # Check if frames are different
        if len(frames) >= 2:
            frames_different = False
            frame_diffs = []
            
            for i in range(1, len(frames)):
                diff = np.abs(frames[i] - frames[i-1]).mean()
                frame_diffs.append(diff)
                
                if diff > 1e-6:
                    frames_different = True
            
            if frames_different:
                print("✅ Frames are different - video loading is working!")
                print(f"Frame differences: {[f'{d:.6f}' for d in frame_diffs]}")
            else:
                print("❌ All frames are identical")
            
            # Save debug images
            save_debug_images(frames, f"episode_{episode_idx}_frames")
        else:
            print("❌ Could not load frames")

def save_debug_images(frames, prefix):
    """Save debug images to visualize the frames."""
    try:
        # Convert to uint8 for saving
        frames_uint8 = []
        for frame in frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                else:
                    frame_uint8 = frame.astype(np.uint8)
            else:
                frame_uint8 = frame.astype(np.uint8)
            frames_uint8.append(frame_uint8)
        
        # Create a grid of images
        fig, axes = plt.subplots(1, len(frames_uint8), figsize=(3*len(frames_uint8), 3))
        if len(frames_uint8) == 1:
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
    test_video_loader() 