#!/usr/bin/env python3
"""
Test script to verify dataset caching works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openpi.training.world_model_training.data_loader import WorldModelDataset, WorldModelDataConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_caching():
    """Test that dataset is only loaded once."""
    
    print("Creating dataset...")
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=2,
        min_episode_length=10,
        max_episodes=100,  # Small number for quick test
        chunk_size=50,  # Small chunk size for quick test
    )
    
    dataset = WorldModelDataset(config, split="train", shuffle=False)
    
    print(f"Dataset created. Total sequences: {len(dataset)}")
    print("Testing sample access...")
    
    # Test accessing a few samples to see if file resolution repeats
    for i in range(5):
        print(f"Accessing sample {i}...")
        sample = dataset[i]
        print(f"Sample {i} shape: {sample[0].video_frames.shape}")
    
    print("Testing chunk loading...")
    # Force chunk loading by accessing a sample from a different chunk
    if len(dataset) > 100:
        sample = dataset[100]
        print(f"Sample 100 shape: {sample[0].video_frames.shape}")
    
    print("Dataset caching test completed!")

if __name__ == "__main__":
    test_dataset_caching() 