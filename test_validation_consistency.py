#!/usr/bin/env python3
"""
Test script to check validation consistency issues.

This script tests whether the target encoder updates are causing
validation loss to be lower than training loss.
"""

import torch
import numpy as np
from openpi.models.vjepa2_world_model import create_vjepa2_model, VJEPA2WorldModelConfig

def test_validation_consistency():
    """Test if target encoder updates cause validation loss discrepancy."""
    print("Testing validation consistency...")
    
    # Create model with smaller config for testing
    config = VJEPA2WorldModelConfig(
        num_frames=4,
        image_size=224,
        encoder_hidden_size=384,
        predictor_hidden_size=192,
        encoder_num_layers=2,
        predictor_num_layers=1,
        use_pretrained_encoder=False,
        loss_exp=2.0,
    )
    
    model = create_vjepa2_model(config)
    
    # Test with dummy video data
    batch_size = 2
    num_frames = 4
    height, width = 224, 224
    channels = 3
    
    # Create dummy video frames
    video_frames = torch.randn(batch_size, num_frames, height, width, channels)
    
    print(f"Video frames shape: {video_frames.shape}")
    
    # Test 1: Initial loss without any updates
    print("\n1. Testing initial loss...")
    model.eval()
    initial_loss = model.compute_loss(video_frames)
    print(f"Initial loss: {initial_loss.item():.6f}")
    
    # Test 2: Loss after target encoder update
    print("\n2. Testing loss after target encoder update...")
    model.train()
    
    # Simulate a training step with target encoder update
    loss = model.compute_loss(video_frames)
    loss.backward()
    
    # Update target encoder
    model.update_target_encoder(momentum=0.99)
    
    # Test loss again
    model.eval()
    updated_loss = model.compute_loss(video_frames)
    print(f"Loss after target encoder update: {updated_loss.item():.6f}")
    
    # Test 3: Multiple updates to see the trend
    print("\n3. Testing multiple target encoder updates...")
    losses = []
    
    for i in range(5):
        model.train()
        loss = model.compute_loss(video_frames)
        loss.backward()
        
        # Update target encoder
        model.update_target_encoder(momentum=0.99)
        
        # Test loss
        model.eval()
        test_loss = model.compute_loss(video_frames)
        losses.append(test_loss.item())
        print(f"  Update {i+1}: Loss = {test_loss.item():.6f}")
    
    # Analyze the trend
    losses = np.array(losses)
    print(f"\nLoss trend analysis:")
    print(f"  Initial: {initial_loss.item():.6f}")
    print(f"  After updates - Mean: {losses.mean():.6f}")
    print(f"  After updates - Std: {losses.std():.6f}")
    print(f"  Change: {losses.mean() - initial_loss.item():.6f}")
    
    if losses.mean() < initial_loss.item():
        print("⚠️  WARNING: Target encoder updates are reducing loss!")
        print("This could explain why validation loss is lower than training loss.")
    else:
        print("✅ Target encoder updates are not reducing loss.")
    
    # Test 4: Check if target encoder is actually being updated
    print("\n4. Checking target encoder parameter changes...")
    
    # Get initial target encoder parameters
    initial_target_params = {}
    for name, param in model.target_encoder.named_parameters():
        initial_target_params[name] = param.data.clone()
    
    # Update target encoder
    model.update_target_encoder(momentum=0.99)
    
    # Check parameter changes
    changed_params = 0
    total_params = 0
    for name, param in model.target_encoder.named_parameters():
        total_params += 1
        if not torch.allclose(param.data, initial_target_params[name]):
            changed_params += 1
    
    print(f"Changed {changed_params}/{total_params} target encoder parameters.")
    
    # Test 5: Compare training vs validation mode
    print("\n5. Testing training vs validation mode...")
    
    # Reset model
    model = create_vjepa2_model(config)
    
    # Training mode
    model.train()
    train_loss = model.compute_loss(video_frames)
    print(f"Training mode loss: {train_loss.item():.6f}")
    
    # Validation mode
    model.eval()
    val_loss = model.compute_loss(video_frames)
    print(f"Validation mode loss: {val_loss.item():.6f}")
    
    if abs(train_loss.item() - val_loss.item()) > 1e-6:
        print("⚠️  WARNING: Training and validation modes produce different losses!")
    else:
        print("✅ Training and validation modes produce consistent losses.")
    
    print("\n✅ Validation consistency test completed!")

if __name__ == "__main__":
    test_validation_consistency() 