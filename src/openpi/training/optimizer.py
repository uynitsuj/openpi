import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class CosineRestartSchedule(LRScheduleConfig):
    """Cosine restart schedule with warmup and floor."""

    warmup_steps: int = 300
    peak_lr: float = 5e-6
    min_lr: float = 5e-8
    cosine_cycle: int = 6000

    def create(self) -> optax.Schedule:
        def schedule(step):
            step = jnp.asarray(step)
            lr = jnp.where(
                step < self.warmup_steps,
                self.peak_lr * (step / self.warmup_steps),
                self._cosine_restart(step - self.warmup_steps),
            )
            return lr
        return schedule

    def _cosine_restart(self, step):
        """Periodic restart to peak_lr, but decay floor = min_lr."""
        cycle = self.cosine_cycle
        mod = step % cycle
        cos = 0.5 * (1 + jnp.cos(jnp.pi * mod / cycle))
        return self.min_lr + (self.peak_lr - self.min_lr) * cos


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def build_optimizer_with_config(params, config):
    """Build optimizer with configuration from WorldModelOptimConfig."""
    from .world_model_training.config import WorldModelOptimConfig, CosineRestartSchedule
    
    if not isinstance(config, WorldModelOptimConfig):
        # Fallback for old configs
        lr_schedule = config.create() if hasattr(config, 'create') else config
        opt = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=0.04,
            b1=0.9, b2=0.999, eps=1e-8,
        )
        return opt
    
    # Build LR schedule
    lr_schedule = CosineRestartSchedule(
        warmup_steps=config.warmup_steps,
        peak_lr=config.peak_lr,
        min_lr=config.min_lr,
        cosine_cycle=config.cosine_cycle,
    ).create()
    
    opt = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        b1=0.9, b2=0.999, eps=1e-8,
    )
    return opt


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
