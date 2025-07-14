"""
Configuration for World Model Training

This module provides configuration classes for training world models,
including data configuration, model configuration, and training parameters.
"""

import dataclasses
import pathlib
from typing import Optional, Any, Dict, List, Literal
import logging

from openpi.models.vjepa2_world_model import VJEPA2WorldModelConfig
from openpi.models.video_masking import MaskingStrategy
from openpi.training.world_model_training.data_loader import WorldModelDataConfig
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class WorldModelTrainConfig:
    """Configuration for world model training."""
    
    # Experiment configuration
    name: str  # Name of the config
    project_name: str = "openpi-worldmodel"
    exp_name: str = "debug"
    
    # Model configuration
    model_config: VJEPA2WorldModelConfig = dataclasses.field(
        default_factory=lambda: VJEPA2WorldModelConfig(
            num_frames=8,
            image_size=224,
            encoder_hidden_size=768,
            predictor_hidden_size=384,
            encoder_num_layers=6,
            predictor_num_layers=3,
            use_pretrained_encoder=True,
            pretrained_model="google/vit-base-patch16-224-in21k",
        )
    )
    
    # Data configuration
    data_config: WorldModelDataConfig = dataclasses.field(
        default_factory=lambda: WorldModelDataConfig(
            repo_id=None,  # Will be set to fake data if None
            num_frames=8,
            frame_skip=1,
            image_size=(224, 224),
            masking_strategy=MaskingStrategy.BLOCK,
            mask_ratio=0.5,
            multi_view_batch_mode=True,
        )
    )
    
    # Training configuration
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=50000,
            decay_lr=1e-6,
        )
    )
    
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(
            weight_decay=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        )
    )
    
    # Weight loading
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    
    # Training parameters
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 2
    num_train_steps: int = 50000
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: Optional[int] = 5000
    
    # Directories
    checkpoint_base_dir: str = "/home/justinyu/checkpoints"
    assets_base_dir: str = "./assets"
    
    # Control flags
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    
    # Optional S3 backup
    s3_checkpoint_path: Optional[str] = None
    
    # Device configuration
    fsdp_devices: int = 1
    
    # Validation
    validation_interval: int = 1000
    validation_steps: int = 100
    
    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
    
    @property
    def assets_dir(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()
    
    def __post_init__(self):
        """Validate configuration."""
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        
        if self.num_train_steps <= 0:
            raise ValueError("Number of training steps must be positive.")


# Predefined configurations
_WORLD_MODEL_CONFIGS = [

    WorldModelTrainConfig(
    name="hummus_vjepa2_world_model",
        exp_name="hummus_wm_training",
        model_config=VJEPA2WorldModelConfig(
            num_frames=8,
            image_size=224,
            encoder_hidden_size=768,
            predictor_hidden_size=384,
            encoder_num_layers=6,
            predictor_num_layers=3,
            use_pretrained_encoder=True,
            pretrained_model="google/vit-base-patch16-224-in21k",
        ),
        data_config=WorldModelDataConfig(
            repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
            num_frames=8,
            image_size=(224, 224),
            masking_strategy=MaskingStrategy.BLOCK,
            # mask_ratio=0.5,
            multi_view_batch_mode=True,
        ),
        batch_size=32,
        num_train_steps=100000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,
            peak_lr=1e-4,
            decay_steps=100000,
            decay_lr=1e-6,
        ),
    ),
    
    WorldModelTrainConfig(
        name="hummus_vjepa2_world_model_debug",
        exp_name="hummus_wm_training_debug",
        model_config=VJEPA2WorldModelConfig(
            num_frames=4,  # Reduced for debugging
            image_size=224,
            encoder_hidden_size=288,  # Smaller for debugging
            predictor_hidden_size=144,  # Smaller for debugging
            encoder_num_layers=2,  # Fewer layers for debugging
            predictor_num_layers=1,  # Fewer layers for debugging
            use_pretrained_encoder=False,  # No pretrained model for debugging
        ),
        data_config=WorldModelDataConfig(
            repo_id="uynitsuj/hummus_xmi_full_subsample_2_cleaned2",
            num_frames=4,  # Reduced for debugging
            image_size=(224, 224),
            masking_strategy=MaskingStrategy.BLOCK,
            # mask_ratio=0.5,
            max_episodes=10,  # Limit episodes for debugging
            multi_view_batch_mode=True,  # Disable for faster debugging
            use_progressive_masking=True,  # Enable progressive masking
        ),
        batch_size=4,  # Reduced from 32 to 4 for faster debugging
        num_workers=8,  # Increased from 2 to 8 for better data loading
        num_train_steps=30000,  # Few steps for debugging
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=1e-4,
            decay_steps=100,
            decay_lr=1e-6,
        ),
    ),
]

# Create lookup dictionary
_CONFIG_DICT = {config.name: config for config in _WORLD_MODEL_CONFIGS}


def get_world_model_config(name: str) -> WorldModelTrainConfig:
    """Get a world model configuration by name."""
    if name not in _CONFIG_DICT:
        available = list(_CONFIG_DICT.keys())
        raise ValueError(f"Config '{name}' not found. Available configs: {available}")
    
    return _CONFIG_DICT[name]


def list_world_model_configs() -> List[str]:
    """List all available world model configurations."""
    return list(_CONFIG_DICT.keys())


def create_custom_world_model_config(
    name: str,
    repo_id: str,
    exp_name: str = "custom",
    **kwargs
) -> WorldModelTrainConfig:
    """Create a custom world model configuration."""
    
    # Start with default config
    config = WorldModelTrainConfig(
        name=name,
        exp_name=exp_name,
        data_config=WorldModelDataConfig(repo_id=repo_id),
    )
    
    # Apply custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config


# CLI support functions
def create_world_model_config_cli():
    """Create a CLI interface for world model configs."""
    import tyro
    
    return tyro.extras.overridable_config_cli(
        {name: (name, config) for name, config in _CONFIG_DICT.items()}
    )


if __name__ == "__main__":
    # Example usage
    config = get_world_model_config("debug_world_model")
    print(f"Config: {config.name}")
    print(f"Model: {config.model_config}")
    print(f"Data: {config.data_config}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
