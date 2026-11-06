import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    ode_inputs = []
    if data_config.ode_head:
        from openpi.policies.ode_policy import ODENormalizeStart

        ode_inputs = [ODENormalizeStart(norm_stats, use_quantiles=data_config.use_quantile_norm)]

    policy = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *ode_inputs,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

    from openpi.models import pi0_ode as _pi0_ode

    # "Execute n of H" (serving only; any model): OPENPI_EXECUTE_STEPS=n truncates every returned chunk to n rows, so a
    # caller that executes whole chunks (abc-rabc batched runner) replans every n steps. OPENPI_ODE_EXECUTED_STEPS is the
    # older ODE-only name (carried row only, no truncation).
    execute = os.environ.get("OPENPI_EXECUTE_STEPS")
    if isinstance(train_config.model, _pi0_ode.Pi0ODEConfig):
        if not data_config.ode_head:
            raise ValueError(f"{train_config.name}: a Pi0ODE model needs an ODE data config (ode_head=True)")
        _enable_ode_carry(policy, train_config.model, norm_stats, data_config.use_quantile_norm, execute)
    if execute:
        from openpi.policies import ode_policy

        n = int(execute)
        if not 0 < n <= train_config.model.action_horizon:
            raise ValueError(f"OPENPI_EXECUTE_STEPS={n} must be in (0, {train_config.model.action_horizon}]")
        policy._sample_actions = ode_policy.TruncateChunk(policy._sample_actions, n)
        logging.info(f"Serving: returning (and executing) {n} of {train_config.model.action_horizon} rows per chunk")
    return policy


def _enable_ode_carry(policy: _policy.Policy, model_config, norm_stats, use_quantiles: bool, execute=None) -> None:
    """Wrap the policy's sample function with the per-world carried reference (openpi.policies.ode_policy)."""
    from openpi.policies import ode_policy

    executed = int(
        execute
        or os.environ.get("OPENPI_ODE_EXECUTED_STEPS", model_config.executed_steps or model_config.action_horizon)
    )
    normalize, slope = ode_policy._unit_slope_and_normalize(norm_stats["actions"], use_quantiles)
    # OPENPI_ODE_SNAP_GRIPPER=1: snap integrated YAM grippers (6, 13) to the normalised open/closed command (0 / 1) at
    # chunk boundaries (paper's rule for a gripper ODE state). No-op for models whose grippers use the direct head.
    snap_dims, snap_levels = (), None
    if os.environ.get("OPENPI_ODE_SNAP_GRIPPER", "0") == "1" and model_config.ode_dim == 14:
        snap_dims = tuple(d for d in (6, 13) if d not in model_config.gripper_head_dims)
        if snap_dims:
            lo, hi = normalize(np.zeros(14, np.float32)), normalize(np.ones(14, np.float32))
            snap_levels = np.stack([[lo[d], hi[d]] for d in snap_dims]).astype(np.float32)
    carry = ode_policy.ODECarry(
        policy._sample_actions,
        executed_steps=executed,
        action_slope=transforms.pad_to_dim(slope, model_config.action_dim),
        jump_dims=ode_policy.yam_arm_dims() if model_config.ode_dim == 14 else tuple(range(model_config.ode_dim)),
        reset_jump_rad=model_config.reset_jump_rad,
        snap_dims=snap_dims,
        snap_levels=snap_levels,
    )
    policy.set_ode_carry(carry)
    logging.info(
        f"ODE head: carrying the executed reference per world (row {executed - 1} of each chunk); snap dims {snap_dims}"
    )
