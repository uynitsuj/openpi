#!/usr/bin/env python3
"""
Debug script to understand validation loss discrepancy.
"""

import torch
import numpy as np
from openpi.models.vjepa2_world_model import create_vjepa2_model, VJEPA2WorldModelConfig

def debug_validation_discrepancy():
    """Debug why validation loss is lower than training loss."""
    print("Debugging validation loss discrepancy...")
    
    # Create model
    config = VJEPA2WorldModelConfig(
        num_frames=4,
        image_size=224,
        encoder_hidden_size=384,
        predictor_hidden_size=192,
        encoder_num_layers=2,
        predictor_num_layers=1,
        use_pretrained_encoder=False,
    )
    
    model = create_vjepa2_model(config)
    
    # Test data
    batch_size = 2
    num_frames = 4
    height, width = 224, 224
    channels = 3
    video_frames = torch.randn(batch_size, num_frames, height, width, channels)
    
    print(f"Test data shape: {video_frames.shape}")
    
    # Test 1: Training step simulation
    print("\n1. Simulating training step...")
    model.train()
    
    # Simulate training with gradient computation
    loss_train = model.compute_loss(video_frames)
    loss_train.backward()
    
    # Update target encoder
    model.update_target_encoder(momentum=0.99)
    
    print(f"Training loss: {loss_train.item():.6f}")
    
    # Test 2: Validation step simulation
    print("\n2. Simulating validation step...")
    model.eval()
    
    with torch.no_grad():
        loss_val = model.compute_loss(video_frames)
    
    print(f"Validation loss: {loss_val.item():.6f}")
    print(f"Difference (val - train): {loss_val.item() - loss_train.item():.6f}")
    
    # Test 3: Same mode comparison
    print("\n3. Comparing same mode...")
    
    # Both in training mode
    model.train()
    loss_train1 = model.compute_loss(video_frames)
    model.train()
    loss_train2 = model.compute_loss(video_frames)
    
    print(f"Training mode 1: {loss_train1.item():.6f}")
    print(f"Training mode 2: {loss_train2.item():.6f}")
    print(f"Difference: {abs(loss_train1.item() - loss_train2.item()):.6f}")
    
    # Both in eval mode
    model.eval()
    loss_val1 = model.compute_loss(video_frames)
    model.eval()
    loss_val2 = model.compute_loss(video_frames)
    
    print(f"Eval mode 1: {loss_val1.item():.6f}")
    print(f"Eval mode 2: {loss_val2.item():.6f}")
    print(f"Difference: {abs(loss_val1.item() - loss_val2.item()):.6f}")
    
    # Test 4: Check if target encoder updates affect loss
    print("\n4. Testing target encoder effect...")
    
    # Reset model
    model = create_vjepa2_model(config)
    
    # Before target encoder update
    model.eval()
    loss_before = model.compute_loss(video_frames)
    print(f"Loss before target encoder update: {loss_before.item():.6f}")
    
    # Update target encoder
    model.update_target_encoder(momentum=0.99)
    
    # After target encoder update
    model.eval()
    loss_after = model.compute_loss(video_frames)
    print(f"Loss after target encoder update: {loss_after.item():.6f}")
    print(f"Change: {loss_after.item() - loss_before.item():.6f}")
    
    # Test 5: Check if the issue is with mixed precision
    print("\n5. Testing mixed precision effect...")
    
    model.eval()
    
    # Without mixed precision
    loss_no_amp = model.compute_loss(video_frames)
    print(f"Loss without mixed precision: {loss_no_amp.item():.6f}")
    
    # With mixed precision
    with torch.cuda.amp.autocast():
        loss_with_amp = model.compute_loss(video_frames)
    print(f"Loss with mixed precision: {loss_with_amp.item():.6f}")
    print(f"Difference: {abs(loss_with_amp.item() - loss_no_amp.item()):.6f}")
    
    # Test 6: Check if the issue is with gradient computation
    print("\n6. Testing gradient computation effect...")
    
    model.train()
    
    # With gradient computation
    loss_with_grad = model.compute_loss(video_frames)
    loss_with_grad.backward()
    print(f"Loss with gradient computation: {loss_with_grad.item():.6f}")
    
    # Without gradient computation
    with torch.no_grad():
        loss_no_grad = model.compute_loss(video_frames)
    print(f"Loss without gradient computation: {loss_no_grad.item():.6f}")
    print(f"Difference: {abs(loss_no_grad.item() - loss_with_grad.item()):.6f}")
    
    # Test 7: Check if the issue is with model state
    print("\n7. Testing model state consistency...")
    
    # Reset model and check if parameters are identical
    model1 = create_vjepa2_model(config)
    model2 = create_vjepa2_model(config)
    
    # Check if initial parameters are identical
    params_match = True
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1.data, p2.data):
            params_match = False
            break
    
    print(f"Initial parameters identical: {params_match}")
    
    # Test with same input
    video_frames = torch.randn(batch_size, num_frames, height, width, channels)
    
    model1.eval()
    model2.eval()
    
    loss1 = model1.compute_loss(video_frames)
    loss2 = model2.compute_loss(video_frames)
    
    print(f"Model 1 loss: {loss1.item():.6f}")
    print(f"Model 2 loss: {loss2.item():.6f}")
    print(f"Difference: {abs(loss1.item() - loss2.item()):.6f}")
    
    print("\n✅ Validation discrepancy debug completed!")

if __name__ == "__main__":
    debug_validation_discrepancy() 