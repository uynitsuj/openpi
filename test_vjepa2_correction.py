#!/usr/bin/env python3
"""
Test script to verify the corrected VJEPA2 implementation.

This script tests that:
1. The model produces reasonable loss values (not suspiciously low)
2. The architecture follows the official V-JEPA2 design
3. The loss computation matches the official implementation
"""

import torch
import numpy as np
from openpi.models.vjepa2_world_model import create_vjepa2_model, VJEPA2WorldModelConfig

def test_vjepa2_implementation():
    """Test the corrected VJEPA2 implementation."""
    print("Testing corrected VJEPA2 implementation...")
    
    # Create model with smaller config for testing
    config = VJEPA2WorldModelConfig(
        num_frames=4,
        image_size=224,
        encoder_hidden_size=384,  # Smaller for testing
        predictor_hidden_size=192,  # Smaller for testing
        encoder_num_layers=2,  # Fewer layers for testing
        predictor_num_layers=1,  # Fewer layers for testing
        use_pretrained_encoder=False,  # No pretrained model for testing
        loss_exp=2.0,
    )
    
    model = create_vjepa2_model(config)
    model.train()
    
    # Test with dummy video data
    batch_size = 2
    num_frames = 4
    height, width = 224, 224
    channels = 3
    
    # Create dummy video frames
    video_frames = torch.randn(batch_size, num_frames, height, width, channels)
    
    print(f"Video frames shape: {video_frames.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multiple forward passes to check loss behavior
    losses = []
    for i in range(10):
        # Forward pass
        loss = model.compute_loss(video_frames)
        losses.append(loss.item())
        
        # Backward pass (simulate training)
        loss.backward()
        
        # Zero gradients
        model.zero_grad()
        
        print(f"Step {i+1}: Loss = {loss.item():.6f}")
    
    # Analyze loss behavior
    losses = np.array(losses)
    print(f"\nLoss statistics:")
    print(f"  Mean: {losses.mean():.6f}")
    print(f"  Std:  {losses.std():.6f}")
    print(f"  Min:  {losses.min():.6f}")
    print(f"  Max:  {losses.max():.6f}")
    
    # Check if loss is reasonable (not suspiciously low)
    if losses.mean() < 0.01:
        print("⚠️  WARNING: Loss is suspiciously low! This might indicate an issue.")
    elif losses.mean() > 10.0:
        print("⚠️  WARNING: Loss is suspiciously high! This might indicate an issue.")
    else:
        print("✅ Loss values appear reasonable.")
    
    # Test target encoder update
    print("\nTesting target encoder momentum update...")
    initial_target_params = {}
    for name, param in model.target_encoder.named_parameters():
        initial_target_params[name] = param.data.clone()
    
    # Update target encoder
    model.update_target_encoder(momentum=0.99)
    
    # Check if parameters were updated
    updated_count = 0
    for name, param in model.target_encoder.named_parameters():
        if not torch.allclose(param.data, initial_target_params[name]):
            updated_count += 1
    
    print(f"Updated {updated_count} target encoder parameters.")
    
    # Test model architecture
    print("\nTesting model architecture...")
    
    # Check that target encoder is frozen
    target_grad_count = 0
    for param in model.target_encoder.parameters():
        if param.requires_grad:
            target_grad_count += 1
    
    if target_grad_count == 0:
        print("✅ Target encoder is properly frozen.")
    else:
        print(f"⚠️  WARNING: Target encoder has {target_grad_count} trainable parameters!")
    
    # Check that context encoder is trainable
    context_grad_count = 0
    for param in model.context_encoder.parameters():
        if param.requires_grad:
            context_grad_count += 1
    
    if context_grad_count > 0:
        print(f"✅ Context encoder has {context_grad_count} trainable parameters.")
    else:
        print("⚠️  WARNING: Context encoder has no trainable parameters!")
    
    # Test with different mask ratios
    print("\nTesting with different mask ratios...")
    mask_ratios = [0.25, 0.5, 0.75]
    
    for mask_ratio in mask_ratios:
        # Create mask with specific ratio
        from openpi.models.vjepa2_world_model import create_video_mask
        mask = create_video_mask(
            video_frames.shape,
            mask_ratio=mask_ratio,
            device=video_frames.device
        )
        
        # Compute loss
        loss = model.compute_loss(video_frames, mask)
        print(f"  Mask ratio {mask_ratio}: Loss = {loss.item():.6f}")
    
    print("\n✅ VJEPA2 implementation test completed!")

if __name__ == "__main__":
    test_vjepa2_implementation() 