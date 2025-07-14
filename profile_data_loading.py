#!/usr/bin/env python3
"""
Profile data loading performance to identify bottlenecks.
"""

import time
import torch
from openpi.training.world_model_training.config import get_world_model_config
from openpi.training.world_model_training.data_loader import create_world_model_data_loader

def profile_data_loading():
    """Profile data loading performance."""
    
    config = get_world_model_config('hummus_vjepa2_world_model_debug')
    
    print(f"Profiling data loading with config:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_workers: {config.num_workers}")
    print(f"  num_frames: {config.data_config.num_frames}")
    print(f"  multi_view_batch_mode: {config.data_config.multi_view_batch_mode}")
    
    # Create data loader
    data_loader = create_world_model_data_loader(
        config.data_config,
        batch_size=config.batch_size,
        split="train",
        shuffle=False,
        num_workers=config.num_workers,
        fake_data=False,
        current_step=0,
    )
    
    # Profile data loading
    print(f"\nProfiling data loading...")
    start_time = time.time()
    
    batch_times = []
    for i, batch in enumerate(data_loader):
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        if i >= 10:  # Profile first 10 batches
            break
            
        start_time = time.time()
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"\nData loading performance:")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Batches per second: {1/avg_batch_time:.2f}")
    print(f"  GPU utilization likely bottlenecked by data loading if < 50%")

if __name__ == "__main__":
    profile_data_loading() 