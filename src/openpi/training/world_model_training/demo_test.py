#!/usr/bin/env python3
"""
Demo script for World Model Training Infrastructure

This script demonstrates that the world model training infrastructure
is working and can be used with the available dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from openpi.training.world_model_training.config import get_world_model_config, list_world_model_configs
from openpi.training.world_model_training.data_loader import WorldModelDataConfig
from openpi.models.vjepa2_world_model import VJEPA2WorldModelConfig
from openpi.models.video_masking import MaskingStrategy
from openpi.models.world_model import WorldModelInput, WorldModelOutput


def demo_configuration():
    """Demonstrate configuration creation and management."""
    print("=" * 60)
    print("WORLD MODEL TRAINING INFRASTRUCTURE DEMO")
    print("=" * 60)
    
    print("\n1. Available Configurations:")
    print("-" * 30)
    configs = list_world_model_configs()
    for i, config_name in enumerate(configs, 1):
        print(f"   {i}. {config_name}")
    
    print("\n2. Loading Debug Configuration:")
    print("-" * 30)
    config = get_world_model_config("debug_world_model")
    print(f"   Name: {config.name}")
    print(f"   Experiment: {config.exp_name}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Training steps: {config.num_train_steps}")
    print(f"   Model frames: {config.model_config.num_frames}")
    print(f"   Model image size: {config.model_config.image_size}")
    print(f"   Data frames: {config.data_config.num_frames}")
    print(f"   Data image size: {config.data_config.image_size}")
    print(f"   Masking strategy: {config.data_config.masking_strategy}")
    print(f"   Mask ratio: {config.data_config.mask_ratio}")
    
    print("\n3. Model Configuration Details:")
    print("-" * 30)
    model_config = config.model_config
    print(f"   Encoder hidden size: {model_config.encoder_hidden_size}")
    print(f"   Encoder layers: {model_config.encoder_num_layers}")
    print(f"   Predictor hidden size: {model_config.predictor_hidden_size}")
    print(f"   Predictor layers: {model_config.predictor_num_layers}")
    print(f"   Use pretrained encoder: {model_config.use_pretrained_encoder}")
    print(f"   Pretrained model: {model_config.pretrained_model}")
    
    print("\n4. Creating Custom Configuration:")
    print("-" * 30)
    
    # Create a custom data config
    custom_data_config = WorldModelDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        num_frames=6,
        image_size=(128, 128),
        masking_strategy=MaskingStrategy.RANDOM,
        mask_ratio=0.6,
    )
    
    # Create a custom model config
    custom_model_config = VJEPA2WorldModelConfig(
        num_frames=6,
        image_size=128,
        encoder_hidden_size=384,  # 384 = 12 * 32, divisible by 12 heads
        predictor_hidden_size=192,  # 192 = 12 * 16, divisible by 12 heads
        encoder_num_layers=4,
        predictor_num_layers=2,
        use_pretrained_encoder=True,
    )
    
    print(f"   Custom data config: {custom_data_config.repo_id}")
    print(f"   Custom model config: {custom_model_config.num_frames} frames")
    print(f"   Custom encoder size: {custom_model_config.encoder_hidden_size}")
    
    return config


def demo_data_structures():
    """Demonstrate data structure creation."""
    print("\n5. Data Structure Examples:")
    print("-" * 30)
    
    # Example dimensions
    batch_size = 2
    num_frames = 4
    height, width = 112, 112 * 3  # 3 cameras concatenated
    channels = 3
    
    # Create mock data using numpy (available everywhere)
    import numpy as np
    
    # Create mock video frames
    video_frames = np.random.randn(batch_size, num_frames, height, width, channels).astype(np.float32)
    
    # Create mock mask
    mask = np.random.choice([True, False], size=(batch_size, num_frames, height, width), p=[0.3, 0.7])
    
    # Create world model input
    model_input = WorldModelInput(
        video_frames=video_frames,
        mask=mask,
        camera_names=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    )
    
    # Create world model output
    model_output = WorldModelOutput(
        predicted_features=video_frames,  # Same shape as input for demo
        reconstruction_loss=np.array(0.5),
        mask_ratio=np.array(0.3),
    )
    
    print(f"   Input video shape: {model_input.video_frames.shape}")
    print(f"   Input mask shape: {model_input.mask.shape}")
    print(f"   Input cameras: {model_input.camera_names}")
    print(f"   Output features shape: {model_output.predicted_features.shape}")
    print(f"   Output loss: {model_output.reconstruction_loss}")
    print(f"   Output mask ratio: {model_output.mask_ratio}")
    
    # Test basic operations
    print(f"   Mean video intensity: {np.mean(model_input.video_frames):.3f}")
    print(f"   Fraction of masked pixels: {np.mean(model_input.mask):.3f}")
    
    return model_input, model_output


def demo_training_pipeline():
    """Demonstrate the training pipeline concept."""
    print("\n6. Training Pipeline Overview:")
    print("-" * 30)
    
    print("   The training pipeline consists of:")
    print("   1. Data Loading:")
    print("      - Load video sequences from LeRobot datasets")
    print("      - Apply temporal sampling and frame skipping")
    print("      - Resize and normalize images")
    print("      - Generate spatial/temporal masks")
    print("      - Batch sequences for training")
    
    print("   2. Model Forward Pass:")
    print("      - Encode visible patches with vision transformer")
    print("      - Generate context representations")
    print("      - Predict masked regions using predictor transformer")
    print("      - Compute reconstruction loss on masked patches")
    
    print("   3. Training Loop:")
    print("      - Optimize predictor parameters using Adam/AdamW")
    print("      - Apply gradient clipping and weight decay")
    print("      - Log training metrics and losses")
    print("      - Save checkpoints periodically")
    print("      - Validate on held-out data")
    
    print("   4. Supported Features:")
    print("      - Multiple masking strategies (block, random, temporal)")
    print("      - Configurable mask ratios and patch sizes")
    print("      - Pretrained vision encoder support")
    print("      - Multi-camera video input handling")
    print("      - Flexible dataset configurations")
    print("      - Weights & Biases integration")
    print("      - Checkpoint management and resuming")


def main():
    """Main demonstration function."""
    try:
        # Demo configuration
        config = demo_configuration()
        
        # Demo data structures
        model_input, model_output = demo_data_structures()
        
        # Demo training pipeline
        demo_training_pipeline()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nNext Steps:")
        print("1. Install PyTorch and JAX for full functionality")
        print("2. Install transformers library for pretrained models")
        print("3. Install datasets library for LeRobot data loading")
        print("4. Run training with: python -m openpi.training.world_model_training.train debug_world_model")
        
        print("\nAvailable Commands:")
        print("- python -m openpi.training.world_model_training.test_forward_pass")
        print("- python -m openpi.training.world_model_training.train --help")
        print("- python -m openpi.training.world_model_training.config --help")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This indicates missing dependencies or configuration issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 