#!/usr/bin/env python3
"""
Debug script to understand the chunk structure and fix the data loader.
"""

import sys
import os
sys.path.append('/home/justinyu/openpi_jy/src')

import logging
from openpi.training.world_model_training.data_loader import (
    WorldModelDataConfig,
    create_world_model_data_loader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chunk_structure():
    """Debug the chunk structure to understand what's happening."""
    print("Starting debug script...")
    
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=2,
        image_size=(224, 224),
        min_episode_length=1,
        max_episodes=1000,  # User's real dataset
        chunk_size=10,    # Keep small for debug clarity
        multi_view_batch_mode=False,
        train_split_ratio=0.6,
        val_split_ratio=0.3,
        test_split_ratio=0.1,
        split_seed=42,
    )
    
    print("=== DEBUGGING CHUNK STRUCTURE ===")
    
    # Create data loader
    print("Creating data loader...")
    loader = create_world_model_data_loader(
        config=config,
        batch_size=2,
        split="train",
        shuffle=False,  # No shuffle for debugging
        num_workers=0,
        fake_data=False,
        current_step=0,
        chunk_size=config.chunk_size,
    )
    
    dataset = loader.dataset
    print(f"Total dataset size: {len(dataset)}")
    print(f"Total episodes in split: {len(dataset.episode_info)}")
    
    # Print episode info
    print("=== EPISODE INFO ===")
    episode_indices = list(dataset.episode_info.keys())
    for i, episode_idx in enumerate(episode_indices[:10]):  # First 10 episodes
        episode_data = dataset.episode_info[episode_idx]
        episode_length = episode_data["length"]
        
        # Calculate number of sequences in this episode
        num_sequences = max(0, episode_length - config.num_frames + 1)
        if config.frame_skip > 1:
            num_sequences = num_sequences // config.frame_skip
        
        print(f"Episode {i}: idx={episode_idx}, length={episode_length}, sequences={num_sequences}")
    
    # Test chunk loading
    print("=== TESTING CHUNK LOADING ===")
    
    # Test first few sequences
    for idx in range(0, min(40, len(dataset))):
        print(f"\n--- Testing sequence {idx} ---")
        
        # Find which episode this sequence belongs to
        sequence_count = 0
        target_episode_idx = 0
        
        for episode_idx in episode_indices:
            episode_data = dataset.episode_info[episode_idx]
            episode_length = episode_data["length"]
            
            if episode_length < config.min_episode_length:
                continue
            
            # Calculate number of sequences in this episode
            num_sequences = max(0, episode_length - config.num_frames + 1)
            if config.frame_skip > 1:
                num_sequences = num_sequences // config.frame_skip
            
            if sequence_count + num_sequences > idx:
                # This episode contains our target sequence
                target_episode_idx = episode_indices.index(episode_idx)
                print(f"Sequence {idx} belongs to episode {target_episode_idx} (episode_idx={episode_idx})")
                print(f"Sequence offset within episode: {idx - sequence_count}")
                break
            
            sequence_count += num_sequences
        else:
            print(f"Sequence {idx} not found!")
            continue
        
        # Calculate chunk boundaries
        chunk_start = (target_episode_idx // config.chunk_size) * config.chunk_size
        chunk_end = min(chunk_start + config.chunk_size, len(episode_indices))
        print(f"Chunk boundaries: episodes {chunk_start} to {chunk_end}")
        
        # Test loading the chunk
        dataset._load_chunk(chunk_start)
        print(f"Loaded {len(dataset.episodes)} sequences from chunk")
        
        # Find the sequence within the loaded episodes
        chunk_sequence_start = 0
        for episode_idx in episode_indices[chunk_start:chunk_end]:
            episode_data = dataset.episode_info[episode_idx]
            episode_length = episode_data["length"]
            
            if episode_length < config.min_episode_length:
                continue
            
            num_sequences = max(0, episode_length - config.num_frames + 1)
            if config.frame_skip > 1:
                num_sequences = num_sequences // config.frame_skip
            
            if chunk_sequence_start + num_sequences > idx:
                # Found the episode containing our sequence
                local_idx = idx - chunk_sequence_start
                print(f"Local index: {local_idx}")
                break
            chunk_sequence_start += num_sequences
        else:
            print(f"Sequence {idx} not found in chunk!")
            continue
        
        # Test accessing the sequence
        if local_idx < len(dataset.episodes):
            episode = dataset.episodes[local_idx]
            print(f"Successfully accessed sequence {idx}: episode={episode['episode_index']}, frames={episode['frame_indices']}")
        else:
            print(f"Local index {local_idx} out of bounds for episodes (len: {len(dataset.episodes)})")

if __name__ == "__main__":
    debug_chunk_structure() 