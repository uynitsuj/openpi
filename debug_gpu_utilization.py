#!/usr/bin/env python3
"""
Debug script to identify bottlenecks causing low GPU utilization.
"""

import sys
import os
import time
import torch
import numpy as np
import jax.numpy as jnp
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openpi.training.world_model_training.data_loader import WorldModelDataset, WorldModelDataConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_gpu_utilization():
    """Debug GPU utilization issues."""
    
    print("=== GPU Utilization Debug ===")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    else:
        print("No GPU available!")
        return
    
    # Create config
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=2,
        min_episode_length=10,
        max_episodes=1000,
        chunk_size=500,
    )
    
    # Create dataset
    print("\n=== Creating Dataset ===")
    start_time = time.time()
    dataset = WorldModelDataset(config, split="train", shuffle=False)
    dataset_time = time.time() - start_time
    print(f"Dataset creation time: {dataset_time:.2f}s")
    
    # Test data loading speed
    print("\n=== Testing Data Loading Speed ===")
    num_samples = 10
    
    # Test CPU loading speed
    cpu_times = []
    for i in range(num_samples):
        start_time = time.time()
        sample = dataset[i]
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        print(f"Sample {i}: {cpu_time:.3f}s")
    
    avg_cpu_time = np.mean(cpu_times)
    print(f"Average CPU loading time: {avg_cpu_time:.3f}s")
    
    # Test GPU transfer speed (convert JAX to numpy first)
    print("\n=== Testing GPU Transfer Speed ===")
    gpu_times = []
    for i in range(num_samples):
        start_time = time.time()
        sample = dataset[i]
        # Convert JAX arrays to numpy, then to GPU
        video_frames_np = np.array(sample[0].video_frames)
        mask_np = np.array(sample[0].mask)
        video_frames = torch.from_numpy(video_frames_np).cuda()
        mask = torch.from_numpy(mask_np).cuda()
        torch.cuda.synchronize()  # Wait for GPU
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)
        print(f"Sample {i} (with GPU transfer): {gpu_time:.3f}s")
    
    avg_gpu_time = np.mean(gpu_times)
    print(f"Average GPU transfer time: {avg_gpu_time:.3f}s")
    
    # Check memory usage
    print(f"\n=== Memory Usage ===")
    print(f"GPU Memory after transfers: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
    
    # Calculate throughput
    samples_per_second = 1.0 / avg_cpu_time
    print(f"\n=== Throughput Analysis ===")
    print(f"Data loading throughput: {samples_per_second:.1f} samples/second")
    print(f"GPU transfer throughput: {1.0/avg_gpu_time:.1f} samples/second")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if avg_cpu_time > 0.1:
        print("⚠️  Data loading is slow (>0.1s per sample)")
        print("   - Consider reducing chunk_size")
        print("   - Consider using num_workers > 0")
        print("   - Consider using prefetch_factor > 2")
        print("   - Consider using pin_memory=True")
    
    if avg_gpu_time > avg_cpu_time * 1.5:
        print("⚠️  GPU transfer is bottleneck")
        print("   - Consider batch processing")
        print("   - Consider using pin_memory=True")
    
    if samples_per_second < 10:
        print("⚠️  Low throughput (<10 samples/second)")
        print("   - This will limit GPU utilization")
        print("   - Consider optimizing data loading pipeline")
    
    # Additional analysis
    print(f"\n=== Bottleneck Analysis ===")
    if avg_cpu_time > 0.05:
        print("🔴 Data loading is the primary bottleneck")
        print("   - GPU is waiting for data")
        print("   - Consider: smaller chunks, more workers, prefetching")
    else:
        print("🟢 Data loading speed is acceptable")
    
    print(f"Expected GPU utilization: {min(100, (0.05/avg_cpu_time)*100):.1f}%")

if __name__ == "__main__":
    debug_gpu_utilization() 