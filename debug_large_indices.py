#!/usr/bin/env python3
"""
Debug script to test mapping for large/random indices across the dataset.
"""

import sys
import os
import random
sys.path.append('/home/justinyu/openpi_jy/src')

from openpi.training.world_model_training.data_loader import (
    WorldModelDataConfig,
    create_world_model_data_loader,
)

def debug_large_indices():
    print("Starting large index debug script...")
    config = WorldModelDataConfig(
        repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
        num_frames=4,
        frame_skip=2,
        image_size=(224, 224),
        min_episode_length=1,
        max_episodes=1000,
        chunk_size=10,
        multi_view_batch_mode=False,
        train_split_ratio=0.6,
        val_split_ratio=0.3,
        test_split_ratio=0.1,
        split_seed=42,
    )
    loader = create_world_model_data_loader(
        config=config,
        batch_size=2,
        split="train",
        shuffle=False,
        num_workers=0,
        fake_data=False,
        current_step=0,
        chunk_size=config.chunk_size,
    )
    dataset = loader.dataset
    print(f"Total dataset size: {len(dataset)}")
    episode_indices = list(dataset.episode_info.keys())
    print(f"Total episodes in split: {len(episode_indices)}")

    # Pick 10 random indices across the dataset
    indices = [0, 1, 2, 10, 100, 500, 1000, len(dataset)//2, len(dataset)-2, len(dataset)-1]
    if len(dataset) > 10000:
        indices += random.sample(range(len(dataset)), 10)
    indices = sorted(set([i for i in indices if i < len(dataset)]))

    for idx in indices:
        print(f"\n--- Testing sequence {idx} ---")
        sequence_count = 0
        target_episode_idx = 0
        for episode_idx in episode_indices:
            episode_data = dataset.episode_info[episode_idx]
            episode_length = episode_data["length"]
            if episode_length < config.min_episode_length:
                continue
            num_sequences = max(0, episode_length - config.num_frames + 1)
            if config.frame_skip > 1:
                num_sequences = num_sequences // config.frame_skip
            if sequence_count + num_sequences > idx:
                target_episode_idx = episode_indices.index(episode_idx)
                print(f"Sequence {idx} belongs to episode {target_episode_idx} (episode_idx={episode_idx})")
                print(f"Sequence offset within episode: {idx - sequence_count}")
                break
            sequence_count += num_sequences
        else:
            print(f"Sequence {idx} not found!")
            continue
        chunk_start = (target_episode_idx // config.chunk_size) * config.chunk_size
        chunk_end = min(chunk_start + config.chunk_size, len(episode_indices))
        print(f"Chunk boundaries: episodes {chunk_start} to {chunk_end}")
        dataset._load_chunk(chunk_start)
        print(f"Loaded {len(dataset.episodes)} sequences from chunk")
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
                local_idx = idx - chunk_sequence_start
                print(f"Local index: {local_idx}")
                break
            chunk_sequence_start += num_sequences
        else:
            print(f"Sequence {idx} not found in chunk!")
            continue
        if local_idx < len(dataset.episodes):
            episode = dataset.episodes[local_idx]
            print(f"Successfully accessed sequence {idx}: episode={episode['episode_index']}, frames={episode['frame_indices']}")
        else:
            print(f"Local index {local_idx} out of bounds for episodes (len: {len(dataset.episodes)})")

if __name__ == "__main__":
    debug_large_indices() 