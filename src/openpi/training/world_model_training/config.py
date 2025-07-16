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
    
    name: str
    project_name: str = "openpi-worldmodel"
    exp_name: str = "debug"
    
    # Dataloader optimization settings
    use_optimized_dataloader: bool = True
    dataloader_cache_size: int = 200
    dataloader_max_workers: int = 8
    dataloader_prefetch_factor: int = 4
    
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
    
    data_config: WorldModelDataConfig = dataclasses.field(
        default_factory=lambda: WorldModelDataConfig(
            repo_id=None,
            num_frames=8,
            frame_skip=1,
            image_size=(224, 224),
            masking_strategy=MaskingStrategy.MULTI_SCALE,
            mask_ratio=0.75,  # Increased from 0.5 to make task harder
            multi_view_batch_mode=True,
        )
    )
    
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,  # Reduced from 1e-4
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
    
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    
    seed: int = 42
    batch_size: int = 8  # Increased for better gradient estimates
    num_workers: int = 2
    num_train_steps: int = 50000
    
    # VJEPA-2 specific parameters
    target_encoder_momentum: float = 0.99
    loss_exp: float = 1.0  # Changed from 2.0 to 1.0 to match official VJEPA2
    gradient_clip_norm: float = 1.0  # Gradient clipping for stability
    loss_smoothing_factor: float = 0.9  # Exponential moving average for loss
    progress_bar_verbose: bool = True  # Show detailed metrics in progress bar
    
    log_interval: int = 10
    save_interval: int = 1000
    keep_period: Optional[int] = 5000
    
    checkpoint_base_dir: str = "/home/yujustin/checkpoints"
    assets_base_dir: str = "./assets"
    
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    
    s3_checkpoint_path: Optional[str] = None
    
    fsdp_devices: int = 1
    
    validation_interval: int = 100
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


_WORLD_MODEL_CONFIGS = [

    WorldModelTrainConfig(
        name="yam_dishrack_vjepa2_world_model_debug",
        exp_name="hummus_wm_training_debug",
        model_config=VJEPA2WorldModelConfig(
            num_frames=16,  # Match official (16 frames)
            image_size=224,  # Match official (256px)
            encoder_hidden_size=768,  
            predictor_hidden_size=512,  # Half of encoder size
            encoder_num_layers=16,  # ViT-Large depth
            predictor_num_layers=8,  # Deeper predictor
            encoder_num_heads=16,  # ViT-Large heads
            predictor_num_heads=8,  # Half of encoder heads
            encoder_stochastic_depth=0.2,  # Match official
            predictor_stochastic_depth=0.1,  # Lower for predictor
            momentum=0.996,  # Official EMA momentum
            use_pretrained_encoder=False,  
        ),
        data_config=WorldModelDataConfig(
            repo_id="uynitsuj/yam_bimanual_load_dishes_full_absolute",
            num_frames=10,  # Match model config
            image_size=(224, 224),  # Match model config  
            masking_strategy=MaskingStrategy.MULTI_SCALE,
            # mask_ratio=0.85,  # Higher like official (very challenging)
            frame_skip=3,  # Skip frames for more temporal diversity (every 3rd frame)
            multi_view_batch_mode=True,  
            use_progressive_masking=True,  
        ),
        batch_size=8,   # Good balance of performance and memory
        num_workers=8,    # Optimal worker count
        num_train_steps=30000,  
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,  # Longer warmup like official
            peak_lr=1.5e-4,  # Higher peak LR with larger batch accumulation
            decay_steps=30000,  
            decay_lr=1e-6,
        ),
    ),
    
    # Optimized version of the debug config  
    WorldModelTrainConfig(
        name="yam_dishrack_vjepa2_world_model_optimized",
        exp_name="yam_wm_training_optimized",
        use_optimized_dataloader=True,
        dataloader_cache_size=200,      # Good cache size
        dataloader_max_workers=8,       # More parallel workers
        dataloader_prefetch_factor=4,   # Better prefetching
        model_config=VJEPA2WorldModelConfig(
            num_frames=8,  
            image_size=224,
            encoder_hidden_size=768,
            predictor_hidden_size=384,
            encoder_num_layers=6,
            predictor_num_layers=6,
            use_pretrained_encoder=False,  
        ),
        data_config=WorldModelDataConfig(
            repo_id="uynitsuj/yam_bimanual_load_dishes_full_absolute",
            num_frames=8,  
            image_size=(224, 224),
            masking_strategy=MaskingStrategy.MULTI_SCALE,
            multi_view_batch_mode=True,  
            use_progressive_masking=True,
            chunk_size=1500,  # Larger chunks for optimized version
        ),
        batch_size=8,  
        num_workers=16,   # Base workers, will be scaled up by optimized loader
        num_train_steps=30000,  
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=1e-6,
        ),
    ),
]


def get_world_model_config(name: str) -> WorldModelTrainConfig:
    """Get a world model configuration by name."""
    for config in _WORLD_MODEL_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown world model config: {name}")


def list_world_model_configs() -> List[str]:
    """List available world model configurations."""
    return [config.name for config in _WORLD_MODEL_CONFIGS]


def create_custom_world_model_config(
    name: str,
    repo_id: str,
    exp_name: str = "custom",
    **kwargs
) -> WorldModelTrainConfig:
    """Create a custom world model configuration."""
    base_config = WorldModelTrainConfig()
    
    # Override with custom values
    custom_config = dataclasses.replace(
        base_config,
        name=name,
        exp_name=exp_name,
        data_config=dataclasses.replace(
            base_config.data_config,
            repo_id=repo_id,
        ),
        **kwargs
    )
    
    return custom_config


def create_world_model_config_cli():
    """Create world model configuration from command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="World Model Training Configuration")
    parser.add_argument(
        "--config-name",
        type=str,
        default="hummus_vjepa2_world_model_debug",
        help="Name of the configuration to use",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Custom repository ID to override config",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Custom experiment name to override config",
    )
    
    args = parser.parse_args()
    
    config = get_world_model_config(args.config_name)
    
    if args.repo_id:
        config = dataclasses.replace(
            config,
            data_config=dataclasses.replace(
                config.data_config,
                repo_id=args.repo_id,
            )
        )
    
    if args.exp_name:
        config = dataclasses.replace(
            config,
            exp_name=args.exp_name,
        )
    
    return config
