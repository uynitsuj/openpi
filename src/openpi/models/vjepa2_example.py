"""
Example Usage of VJEPA-2 World Model

This script demonstrates how to use the VJEPA-2 world model for video understanding,
including training, inference, and various configuration options.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import os
from typing import Optional, Dict, Any

# Import our VJEPA-2 components
from .vjepa2_world_model import (
    VJEPA2WorldModel, 
    VJEPA2WorldModelConfig, 
    create_vjepa2_model
)
from .video_masking import (
    VideoMaskGenerator, 
    MaskingStrategy, 
    AdaptiveMaskGenerator,
    create_video_mask
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyVideoDataset(Dataset):
    """
    Dummy video dataset for demonstration purposes.
    In practice, this would load real video data.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 16,
        image_size: int = 224,
        num_channels: int = 3,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_channels = num_channels
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random video frames
        video = torch.randn(
            self.num_frames, 
            self.image_size, 
            self.image_size, 
            self.num_channels
        )
        
        # Normalize to [-1, 1] range
        video = torch.tanh(video)
        
        return {
            'video': video,
            'index': idx,
        }


class VJEPA2Trainer:
    """
    Trainer class for VJEPA-2 world model.
    """
    
    def __init__(
        self,
        model: VJEPA2WorldModel,
        config: VJEPA2WorldModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup adaptive mask generator
        self.mask_generator = AdaptiveMaskGenerator(
            input_size=(config.num_frames, config.image_size, config.image_size),
            patch_size=(config.patch_size, config.patch_size),
            temporal_patch_size=config.temporal_patch_size,
            device=device,
        )
        
        # Training metrics
        self.train_losses = []
        self.step = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of video data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Move batch to device
        video_frames = batch['video'].to(self.device)  # (B, T, H, W, C)
        batch_size = video_frames.size(0)
        
        # Generate masks
        mask = self.mask_generator.generate_mask(
            batch_size=batch_size,
            strategy=MaskingStrategy.BLOCK,
            mask_ratio=self.config.mask_ratio,
        )
        
        # Forward pass
        self.optimizer.zero_grad()
        loss = self.model.compute_loss(video_frames, mask)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.train_losses.append(loss.item())
        self.step += 1
        
        return {
            'loss': loss.item(),
            'step': self.step,
            'mask_ratio': mask.float().mean().item(),
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary of epoch metrics
        """
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            epoch_losses.append(metrics['loss'])
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Mask Ratio: {metrics['mask_ratio']:.3f}"
                )
        
        return {
            'avg_loss': np.mean(epoch_losses),
            'epoch_steps': len(dataloader),
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                video_frames = batch['video'].to(self.device)
                batch_size = video_frames.size(0)
                
                # Generate masks
                mask = self.mask_generator.generate_mask(
                    batch_size=batch_size,
                    strategy=MaskingStrategy.BLOCK,
                    mask_ratio=self.config.mask_ratio,
                )
                
                # Forward pass
                loss = self.model.compute_loss(video_frames, mask)
                eval_losses.append(loss.item())
        
        return {
            'avg_loss': np.mean(eval_losses),
            'num_samples': len(dataloader.dataset),
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
            'train_losses': self.train_losses,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        logger.info(f"Checkpoint loaded from {filepath}")


def demo_basic_usage():
    """
    Demonstrate basic usage of VJEPA-2 world model.
    """
    print("=== VJEPA-2 World Model Demo ===")
    
    # Create model with default configuration
    config = VJEPA2WorldModelConfig(
        num_frames=16,
        image_size=224,
        patch_size=16,
        temporal_patch_size=2,
        encoder_hidden_size=768,
        predictor_hidden_size=384,
        mask_ratio=0.75,
        use_pretrained_encoder=False,  # Set to False for faster demo
    )
    
    model = create_vjepa2_model(config)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Test with dummy data
    batch_size = 2
    video_frames = torch.randn(
        batch_size, 
        config.num_frames, 
        config.image_size, 
        config.image_size, 
        config.num_channels
    )
    
    print(f"Input video shape: {video_frames.shape}")
    
    # Test encoding
    features = model.encode_video(video_frames)
    print(f"Encoded features shape: {features.shape}")
    
    # Test masked prediction
    mask = create_video_mask(
        video_frames.shape,
        mask_ratio=0.75,
        strategy=MaskingStrategy.BLOCK,
        device="cpu"
    )
    
    predicted_features = model.predict_masked_regions(video_frames, mask)
    print(f"Predicted features shape: {predicted_features.shape}")
    
    # Test loss computation
    loss = model.compute_loss(video_frames)
    print(f"Loss: {loss.item():.4f}")
    
    print("Basic demo completed successfully!")


def demo_training():
    """
    Demonstrate training the VJEPA-2 world model.
    """
    print("\n=== VJEPA-2 Training Demo ===")
    
    # Configuration
    config = VJEPA2WorldModelConfig(
        num_frames=16,
        image_size=224,
        encoder_hidden_size=512,  # Smaller for faster training
        predictor_hidden_size=256,
        encoder_num_layers=6,
        predictor_num_layers=4,
        mask_ratio=0.75,
        use_pretrained_encoder=False,
    )
    
    # Create model and trainer
    model = create_vjepa2_model(config)
    trainer = VJEPA2Trainer(
        model=model,
        config=config,
        device="cpu",  # Use CPU for demo
        learning_rate=1e-4,
    )
    
    # Create dummy dataset
    train_dataset = DummyVideoDataset(
        num_samples=100,
        num_frames=config.num_frames,
        image_size=config.image_size,
    )
    
    val_dataset = DummyVideoDataset(
        num_samples=20,
        num_frames=config.num_frames,
        image_size=config.image_size,
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Train for a few epochs
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Training - Avg Loss: {train_metrics['avg_loss']:.4f}")
        
        # Evaluation
        val_metrics = trainer.evaluate(val_loader)
        print(f"Validation - Avg Loss: {val_metrics['avg_loss']:.4f}")
    
    print("Training demo completed!")


def demo_masking_strategies():
    """
    Demonstrate different masking strategies.
    """
    print("\n=== Masking Strategies Demo ===")
    
    # Video dimensions
    input_size = (16, 224, 224)  # T, H, W
    batch_size = 2
    
    # Test different strategies
    strategies = [
        MaskingStrategy.RANDOM,
        MaskingStrategy.BLOCK,
        MaskingStrategy.TUBE,
        MaskingStrategy.TEMPORAL,
        MaskingStrategy.RUNNING_CELL,
    ]
    
    for strategy in strategies:
        generator = VideoMaskGenerator(
            input_size=input_size,
            strategy=strategy,
            mask_ratio=0.75,
            device="cpu"
        )
        
        mask = generator.generate_mask(batch_size)
        mask_ratio = mask.float().mean().item()
        
        print(f"{strategy.value}: mask shape={mask.shape}, ratio={mask_ratio:.3f}")
    
    print("Masking strategies demo completed!")


def demo_adaptive_masking():
    """
    Demonstrate adaptive masking with curriculum learning.
    """
    print("\n=== Adaptive Masking Demo ===")
    
    input_size = (16, 224, 224)
    batch_size = 2
    
    # Create adaptive mask generator
    adaptive_generator = AdaptiveMaskGenerator(
        input_size=input_size,
        device="cpu"
    )
    
    # Define curriculum schedule
    curriculum_schedule = [
        (0, MaskingStrategy.RANDOM, 0.5),
        (1000, MaskingStrategy.BLOCK, 0.6),
        (5000, MaskingStrategy.TUBE, 0.75),
    ]
    
    # Test curriculum at different steps
    test_steps = [0, 500, 1500, 6000]
    
    for step in test_steps:
        mask = adaptive_generator.generate_curriculum_mask(
            batch_size=batch_size,
            training_step=step,
            curriculum_schedule=curriculum_schedule
        )
        
        mask_ratio = mask.float().mean().item()
        print(f"Step {step}: mask ratio={mask_ratio:.3f}")
    
    print("Adaptive masking demo completed!")


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_training()
    demo_masking_strategies()
    demo_adaptive_masking()
    
    print("\n=== All Demos Completed Successfully! ===")
    
    # Print summary
    print("\nVJEPA-2 World Model Summary:")
    print("- Vision encoder + predictor architecture")
    print("- Supports multiple masking strategies")
    print("- Self-supervised learning from video")
    print("- Predicts masked regions in representation space")
    print("- Can be used for video understanding and prediction")
    print("- Built with PyTorch and HuggingFace transformers")
    print("- Inspired by Meta AI's V-JEPA research") 