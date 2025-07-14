#!/usr/bin/env python3
"""
Debug script to understand chunk mapping issues in the data loader.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openpi.training.world_model_training.data_loader import WorldModelDataset, WorldModelDataConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chunk_mapping():
    """Debug the chunk mapping logic."""
    
    # Create config
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=2,
        min_episode_length=10,
        max_episodes=1000,
        chunk_size=10,  # Small chunk size for debugging
    )
    
    # Create dataset
    dataset = WorldModelDataset(config, split="train", shuffle=False)
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Total episodes in split: {len(dataset.episode_info)}")
    
    # Get episode indices
    episode_indices = list(dataset.episode_info.keys())
    
    # Test specific problematic sequences
    test_sequences = [914, 1000, 9790, 23967, 24999, 26176, 28467, 31800, 41431, 51863, 53569, 57032, 63598, 63599]
    
    for seq_idx in test_sequences:
        print(f"\n--- Testing sequence {seq_idx} ---")
        
        # Find which episode this sequence belongs to
        sequence_count = 0
        target_episode_idx = None
        target_episode_global_idx = None
        
        for episode_idx in episode_indices:
            episode_data = dataset.episode_info[episode_idx]
            episode_length = episode_data["length"]
            if episode_length < config.min_episode_length:
                continue
            num_sequences = max(0, episode_length - config.num_frames + 1)
            if config.frame_skip > 1:
                num_sequences = num_sequences // config.frame_skip
            
            if sequence_count + num_sequences > seq_idx:
                target_episode_idx = episode_idx
                target_episode_global_idx = episode_indices.index(episode_idx)
                break
            sequence_count += num_sequences
        else:
            print(f"Sequence {seq_idx} not found in any episode!")
            continue
        
        print(f"Sequence {seq_idx} belongs to episode {target_episode_global_idx} (episode_idx={target_episode_idx})")
        
        # Calculate chunk boundaries
        chunk_start = (target_episode_global_idx // config.chunk_size) * config.chunk_size
        chunk_end = min(chunk_start + config.chunk_size, len(episode_indices))
        
        print(f"Chunk boundaries: episodes {chunk_start} to {chunk_end}")
        
        # Calculate global sequence index of the first sequence in the chunk
        chunk_global_start = 0
        for episode_idx in episode_indices[:chunk_start]:
            episode_data = dataset.episode_info[episode_idx]
            episode_length = episode_data["length"]
            if episode_length < config.min_episode_length:
                continue
            num_sequences = max(0, episode_length - config.num_frames + 1)
            if config.frame_skip > 1:
                num_sequences = num_sequences // config.frame_skip
            chunk_global_start += num_sequences
        
        print(f"Global sequence index of chunk start: {chunk_global_start}")
        
        # Load the chunk manually to see what's actually loaded
        dataset._load_chunk(chunk_start)
        print(f"Loaded {len(dataset.episodes)} sequences from chunk")
        
        # Calculate local index
        local_idx = seq_idx - chunk_global_start
        print(f"Calculated local index: {local_idx}")
        
        if local_idx < 0 or local_idx >= len(dataset.episodes):
            print(f"Local index {local_idx} out of bounds for chunk (len: {len(dataset.episodes)})")
            
            # Let's see what sequences are actually in the chunk
            print("First 10 sequences in chunk:")
            for i in range(min(10, len(dataset.episodes))):
                episode = dataset.episodes[i]
                print(f"  {i}: episode={episode['episode_index']}, frames={episode['frame_indices']}")
            
            if len(dataset.episodes) > 10:
                print("Last 10 sequences in chunk:")
                for i in range(max(0, len(dataset.episodes) - 10), len(dataset.episodes)):
                    episode = dataset.episodes[i]
                    print(f"  {i}: episode={episode['episode_index']}, frames={episode['frame_indices']}")
        else:
            episode = dataset.episodes[local_idx]
            print(f"Successfully accessed sequence {seq_idx}: episode={episode['episode_index']}, frames={episode['frame_indices']}")

if __name__ == "__main__":
    debug_chunk_mapping() 