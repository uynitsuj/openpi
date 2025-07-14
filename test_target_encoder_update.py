#!/usr/bin/env python3
"""
Test script to debug target encoder parameter updates.
"""

import torch
import numpy as np
from openpi.models.vjepa2_world_model import create_vjepa2_model, VJEPA2WorldModelConfig

def test_target_encoder_update():
    """Test target encoder parameter updates."""
    print("Testing target encoder parameter updates...")
    
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
    
    # Check initial state
    print("\n1. Checking initial state...")
    target_params = list(model.target_encoder.parameters())
    context_params = list(model.context_encoder.parameters())
    
    print(f"Target encoder parameters: {len(target_params)}")
    print(f"Context encoder parameters: {len(context_params)}")
    
    # Check if parameters are frozen
    frozen_target = sum(1 for p in target_params if not p.requires_grad)
    print(f"Frozen target parameters: {frozen_target}/{len(target_params)}")
    
    # Get initial parameter values
    initial_target_values = {}
    for i, param in enumerate(target_params):
        initial_target_values[i] = param.data.clone()
    
    # Test parameter update
    print("\n2. Testing parameter update...")
    
    # First, let's modify context parameters to make them different
    with torch.no_grad():
        for param in context_params:
            param.data += torch.randn_like(param.data) * 0.1
    
    # Now update target encoder
    model.update_target_encoder(momentum=0.99)
    
    # Check if parameters changed
    changed_count = 0
    for i, param in enumerate(target_params):
        if not torch.allclose(param.data, initial_target_values[i]):
            changed_count += 1
            print(f"  Parameter {i} changed")
        else:
            print(f"  Parameter {i} unchanged")
    
    print(f"\nChanged parameters: {changed_count}/{len(target_params)}")
    
    # Test with different momentum values
    print("\n3. Testing different momentum values...")
    
    # Reset model
    model = create_vjepa2_model(config)
    
    # Modify context parameters
    with torch.no_grad():
        for param in context_params:
            param.data += torch.randn_like(param.data) * 0.1
    
    # Test with momentum = 0.0 (should copy context exactly)
    model.update_target_encoder(momentum=0.0)
    
    # Check if target matches context
    matches = 0
    for target_param, context_param in zip(target_params, context_params):
        if torch.allclose(target_param.data, context_param.data):
            matches += 1
    
    print(f"Parameters matching context (momentum=0.0): {matches}/{len(target_params)}")
    
    # Test with momentum = 1.0 (should not change)
    model = create_vjepa2_model(config)
    initial_values = {i: param.data.clone() for i, param in enumerate(target_params)}
    
    model.update_target_encoder(momentum=1.0)
    
    unchanged = 0
    for i, param in enumerate(target_params):
        if torch.allclose(param.data, initial_values[i]):
            unchanged += 1
    
    print(f"Parameters unchanged (momentum=1.0): {unchanged}/{len(target_params)}")
    
    print("\n✅ Target encoder update test completed!")

if __name__ == "__main__":
    test_target_encoder_update() 