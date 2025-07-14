#!/usr/bin/env python3
"""
Simple test script to verify LeRobot dataset loading works correctly.
"""

import sys
import os
sys.path.append('/home/justinyu/openpi_jy/src')

from openpi.training.world_model_training.data_loader import WorldModelDataset, WorldModelDataConfig
from openpi.models.video_masking import MaskingStrategy

def test_data_loading():
    """Test that the data loading works correctly."""
    print("Testing LeRobot dataset loading...")
    
    # Create a simple config
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        image_size=(224, 224),
        masking_strategy=MaskingStrategy.BLOCK,
        mask_ratio=0.5,
        max_episodes=5,  # Very small for testing
    )
    
    try:
        # Create dataset
        dataset = WorldModelDataset(config, split="train", shuffle=False)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Try to get one item
        if len(dataset) > 0:
            item = dataset[0]
            print("Successfully loaded first item!")
            print(f"Input video shape: {item[0].video_frames.shape}")
            print(f"Mask shape: {item[0].mask.shape}")
            print(f"Camera names: {item[0].camera_names}")
            print(f"Output features shape: {item[1].predicted_features.shape}")
            return True
        else:
            print("Dataset is empty!")
            return False
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("✅ Data loading test passed!")
    else:
        print("❌ Data loading test failed!")
        sys.exit(1) 