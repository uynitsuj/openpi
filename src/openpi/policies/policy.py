from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # RTC action_prefix normalizer — invert the Unnormalize step in
        # output_transforms so an incoming prefix (client-space actions) is
        # projected back into the model's normalized action space, matching
        # the coordinate system of the denoiser's x_t / x1_t. Without this,
        # the RTC guidance correction pulls x1_t toward client-space values
        # which then get re-unnormalized on output → chunks violently off.
        self._rtc_action_normalizer: _transforms.Normalize | None = None
        for t in output_transforms:
            if isinstance(t, _transforms.Unnormalize):
                self._rtc_action_normalizer = _transforms.Normalize(
                    norm_stats=t.norm_stats,
                    use_quantiles=t.use_quantiles,
                    strict=False,
                )
                break

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Pluck RTC fields off before transformations — they're model-side guidance
        # hints, not observations that need _input_transform's scaling / resizing.
        # Both are forwarded straight to sample_actions via sample_kwargs below.
        rtc_action_prefix = obs.pop("action_prefix", None) if isinstance(obs, dict) else None
        rtc_inference_delay = obs.pop("inference_delay", 0) if isinstance(obs, dict) else 0
        rtc_execution_horizon = obs.pop("execution_horizon", None) if isinstance(obs, dict) else None
        rtc_max_guidance_weight = obs.pop("max_guidance_weight", None) if isinstance(obs, dict) else None
        # rtc_debug ignored at the policy layer — the RTC denoiser always emits
        # jax.debug.print stats when active (can't thread a Python bool through
        # the NNX jit wrapper without making it a static_argname).
        if isinstance(obs, dict):
            obs.pop("rtc_debug", None)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # Forward RTC guidance hints (model's sample_actions tolerates None).
        if rtc_action_prefix is not None:
            prefix_arr = np.asarray(rtc_action_prefix, dtype=np.float32)
            # Normalize client-space → model normalized-space so err =
            # (prefix − x1_t) compares apples to apples inside the denoiser.
            if self._rtc_action_normalizer is not None:
                prefix_arr = np.asarray(
                    self._rtc_action_normalizer({"actions": prefix_arr})["actions"],
                    dtype=np.float32,
                )
            prefix_arr = jnp.asarray(prefix_arr)
            if prefix_arr.ndim == 2:                  # (T_prev, A) → add batch
                prefix_arr = prefix_arr[None, ...]
            sample_kwargs["action_prefix"] = prefix_arr
            sample_kwargs["inference_delay"] = jnp.asarray(int(rtc_inference_delay), dtype=jnp.int32)
            if rtc_execution_horizon is not None:
                # Pass as a JAX scalar (dynamic) not a Python int — lets value
                # changes between inferences reuse the JIT cache. The model's
                # sample_actions_rtc coerces this into a jnp.float32 regardless.
                sample_kwargs["execution_horizon"] = jnp.asarray(
                    int(rtc_execution_horizon), dtype=jnp.int32
                )
            if rtc_max_guidance_weight is not None:
                sample_kwargs["max_guidance_weight"] = jnp.asarray(
                    float(rtc_max_guidance_weight), dtype=jnp.float32
                )

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
