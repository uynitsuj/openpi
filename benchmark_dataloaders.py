#!/usr/bin/env python3
"""
Benchmark script for WorldModelDataLoader and PyTorch DataLoader.
"""
import sys
import os
import time
import itertools
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openpi.training.world_model_training.data_loader import WorldModelDataset, WorldModelDataConfig, create_world_model_data_loader
from torch.utils.data import DataLoader

# Configurations to try
NUM_WORKERS_LIST = [0]  # Only test num_workers=0 to avoid JAX multiprocessing conflicts
PREFETCH_FACTORS = [2, 4]
PIN_MEMORY_LIST = [False, True]
BATCH_SIZES = [16, 32, 64]  # Try larger batch sizes since we can't use multiprocessing

# Use a small dataset for quick benchmarking
config = WorldModelDataConfig(
    repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
    num_frames=4,
    min_episode_length=10,
    max_episodes=200,
    chunk_size=50,
)

def custom_collate_fn(batch):
    # batch is a list of (WorldModelInput, WorldModelOutput)
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.world_model import WorldModelInput, WorldModelOutput
    inputs, outputs = zip(*batch)
    # Stack video_frames and mask as numpy arrays
    video_frames = np.stack([np.array(inp.video_frames) for inp in inputs])
    mask = np.stack([np.array(inp.mask) for inp in inputs])
    camera_names = [inp.camera_names for inp in inputs]
    # Stack outputs
    predicted_features = np.stack([np.array(out.predicted_features) for out in outputs])
    reconstruction_loss = np.stack([np.array(out.reconstruction_loss) for out in outputs])
    mask_ratio = np.stack([np.array(out.mask_ratio) for out in outputs])
    # Return as tuple of numpy arrays (for fair GPU transfer test)
    return (video_frames, mask, camera_names, predicted_features, reconstruction_loss, mask_ratio)

def benchmark_loader(loader, n_batches=20, loader_name="Loader"):
    start = time.time()
    n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Move to GPU if possible
        if isinstance(batch, (list, tuple)):
            # Try to move all tensors to GPU
            for x in batch:
                if isinstance(x, np.ndarray):
                    torch.from_numpy(x).cuda(non_blocking=True)
        n += 1
    elapsed = time.time() - start
    throughput = n / elapsed if elapsed > 0 else 0
    print(f"{loader_name}: {n} batches in {elapsed:.2f}s, {throughput:.2f} batches/sec")
    return throughput

def run_benchmarks():
    results = []
    for batch_size, num_workers, prefetch_factor, pin_memory in itertools.product(
        BATCH_SIZES, NUM_WORKERS_LIST, PREFETCH_FACTORS, PIN_MEMORY_LIST
    ):
        print(f"\n=== Custom Loader: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}, pin_memory={pin_memory} ===")
        loader = create_world_model_data_loader(
            config,
            batch_size=batch_size,
            split="train",
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            chunk_size=config.chunk_size,
        )
        throughput = benchmark_loader(loader, n_batches=10, loader_name="CustomLoader")
        results.append(("CustomLoader", batch_size, num_workers, prefetch_factor, pin_memory, throughput))

        print(f"\n=== PyTorch DataLoader: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}, pin_memory={pin_memory} ===")
        dataset = WorldModelDataset(config, split="train", shuffle=True, chunk_size=config.chunk_size)
        torch_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Always use 0 to avoid JAX multiprocessing conflicts
            pin_memory=pin_memory,
            drop_last=True,
        )
        # Don't set prefetch_factor when num_workers=0
        torch_loader = DataLoader(
            dataset,
            collate_fn=custom_collate_fn,
            **torch_loader_kwargs
        )
        throughput = benchmark_loader(torch_loader, n_batches=10, loader_name="TorchLoader")
        results.append(("TorchLoader", batch_size, num_workers, prefetch_factor, pin_memory, throughput))

    # Print summary table
    print("\n=== Benchmark Summary ===")
    print(f"{'Loader':<12} {'Batch':<6} {'Workers':<8} {'Prefetch':<9} {'PinMem':<7} {'Throughput (b/s)':<18}")
    for row in results:
        print(f"{row[0]:<12} {row[1]:<6} {row[2]:<8} {row[3]:<9} {str(row[4]):<7} {row[5]:<18.2f}")

if __name__ == "__main__":
    run_benchmarks() 