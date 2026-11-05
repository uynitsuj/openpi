"""pi0 backbone with the ODE action head (no action expert).

Port of the ODE policy's VLA head (github.com/NTUMARS/ODE-POLICY, branch vla-pi05:
Pi05ODEVLMQuery) onto pi0 (pi0_base): PaliGemma (SigLIP So400m/14 + Gemma 2B) encodes the
images and the prompt; ONE learned query token appended after the prefix (it attends to the
prefix, the prefix does not attend to it) is read out after the VLM's final norm and gives the
condition c (2048-d). There is no action expert, no flow-matching time and no state token: the
measured proprioception goes into the vector field only (paper: "proprioception goes into the
vector field, not the condition").

    a_{k+1} = RK4( da/dt = vel_scale * f(a, s | c) ),  a_0 = last executed reference,  k = 0..H-1

The action state `a` is the first `ode_dim` (= 14) entries of the action vector, i.e. the YAM
absolute joint-position references (6 joints + 1 gripper per arm). With absolute position control
the reference IS the command, so no virtual-target conversion is needed (docs/PORTING.md, step 1).
All quantities are in openpi's normalised units (the checkpoint's z-score norm stats).

Inputs specific to this model (see openpi.policies.ode_policy):
  observation.ode_start          [B, action_dim]  normalised start reference (last executed / demonstrated
                                                  action before the chunk; the measured state at an episode start)
  observation.ode_unit_scale     [B, action_dim]  d(normalised)/d(raw) per dimension, to express the start
                                                  perturbation in radians (training only)
  observation.action_valid_mask  [B, H]           False on steps padded past the episode end (training only)
"""

import dataclasses
import logging

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import ode_field
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Start perturbation std per action dimension, in raw units (rad for joints, the [0, 1] gripper command
# for grippers): 3 x the median |per-step change| of each dimension, floored at 0.005 -- the paper's
# joint-space rule (ode_policy/real/build_ode_zarr_real.py, recommended_perturb_pos) measured on
# sim_put_bottles_mjwarp_rmperobj data/chunk-000/file-00000 (337 episodes, 307,644 frames, 30 Hz).
YAM_BOTTLES_PERTURB_STD = (
    0.0050, 0.0083, 0.0052, 0.0050, 0.0064, 0.0050, 0.0050,  # left arm j1..j6, left gripper
    0.0050, 0.0122, 0.0083, 0.0092, 0.0078, 0.0058, 0.0050,  # right arm j1..j6, right gripper
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Pi0ODEConfig(pi0_config.Pi0Config):
    """pi0 VLM + learned query token + ODE head. model_type stays PI0 (same tokenizer / transforms / norm)."""

    action_horizon: int = 30
    # Size of the integrated action state: the first ode_dim entries of the action vector.
    ode_dim: int = 14
    # Vector field (paper defaults: FiLM residual MLP, 4 x 512, zero-initialised output).
    field_width: int = 512
    field_layers: int = 4
    # Nominal per-step velocity in normalised units; the field's output is O(1) when the state moves by
    # vel_scale per control step. On the bottles data the RMS per-step change of the z-scored actions is
    # 0.032 (arm joints) / 0.047 (grippers), so the paper's 0.05 keeps the field output at O(1).
    vel_scale: float = 0.05
    # Contractive start perturbation (paper: half the samples, decay 0.7 per step).
    perturb_prob: float = 0.5
    perturb_std: tuple[float, ...] = YAM_BOTTLES_PERTURB_STD
    perturb_decay: float = 0.7
    # Serving: rows of the returned chunk the client executes before the next query; the per-world carried
    # reference is row (executed_steps - 1). None = the whole chunk (the abc-rabc batched runner executes all 30).
    executed_steps: int | None = None
    # Serving: a world whose measured arm joints are further than this (rad, max over the 12 arm joints) from its
    # carried reference is treated as a new episode and restarts from the measured state.
    reset_jump_rad: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.pi05:
            raise ValueError("Pi0ODEConfig is the pi0 (non-pi05) backbone; use pi05=False")
        if not 0 < self.ode_dim <= self.action_dim:
            raise ValueError(f"ode_dim={self.ode_dim} must be in (0, action_dim={self.action_dim}]")
        ode_field.as_std_vector(self.perturb_std, self.ode_dim)
        if self.executed_steps is not None and not 0 < self.executed_steps <= self.action_horizon:
            raise ValueError("executed_steps must be in (0, action_horizon]")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ODE":
        return Pi0ODE(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                ode_start=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                ode_unit_scale=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                action_valid_mask=jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.bool_),
            )
        return observation_spec, action_spec

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.Nothing


class Pi0ODE(_model.BaseModel):
    def __init__(self, config: Pi0ODEConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        vlm = _gemma.get_config(config.paligemma_variant)
        # Single-stream Gemma (VLM only): same parameter names as pi0's first expert, so pi0_base loads as is.
        llm = nnx_bridge.ToNNX(_gemma.Module(configs=[vlm], embed_dtype=config.dtype, adarms=False))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=vlm.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # The learned readout token (paper: N(0, 0.02^2)).
        self.ode_query = nnx.Param(jax.random.normal(rngs.params(), (1, 1, vlm.width)) * 0.02)
        self.ode_field = ode_field.VectorField(
            cond_dim=vlm.width,
            state_dim=config.ode_dim,
            field_dim=config.ode_dim,
            width=config.field_width,
            num_layers=config.field_layers,
            rngs=rngs,
        )
        self.ode_dim = config.ode_dim
        self.vel_scale = config.vel_scale
        self.perturb_prob = config.perturb_prob
        self.perturb_std = ode_field.as_std_vector(config.perturb_std, config.ode_dim)
        self.perturb_decay = config.perturb_decay
        self.deterministic = True

    # Same image / language embedding path (and parameter names) as pi0.
    embed_prefix = pi0.Pi0.embed_prefix

    def condition(self, obs: _model.Observation) -> at.Float[at.Array, "b emb"]:
        """VLM forward over [images, prompt, query]; returns the query's output (float32)."""
        prefix, prefix_mask, prefix_ar = self.embed_prefix(obs)
        batch = prefix.shape[0]
        query = jnp.broadcast_to(self.ode_query.value, (batch, 1, prefix.shape[-1])).astype(prefix.dtype)
        tokens = jnp.concatenate([prefix, query], axis=1)
        mask = jnp.concatenate([prefix_mask, jnp.ones((batch, 1), jnp.bool_)], axis=1)
        # ar=True on the query: it reads the whole prefix and itself; the prefix cannot read it.
        ar = jnp.concatenate([prefix_ar, jnp.ones((1,), jnp.bool_)])
        (out,), _ = self.PaliGemma.llm(
            [tokens], mask=pi0.make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1
        )
        return out[:, -1].astype(jnp.float32)

    def _integrate(self, a0, s, c):
        return ode_field.integrate(self.ode_field, a0, s, c, horizon=self.action_horizon, vel_scale=self.vel_scale)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if observation.ode_start is None:
            raise ValueError("Pi0ODE needs observation.ode_start (ODE data config: LeRobot*ODEDataConfig)")
        preprocess_rng, perturb_rng = jax.random.split(rng)
        obs = _model.preprocess_observation(preprocess_rng, observation, train=train)
        c = self.condition(obs)
        d = self.ode_dim
        a0 = obs.ode_start[..., :d].astype(jnp.float32)
        s = obs.state[..., :d].astype(jnp.float32)
        target = actions[..., :d].astype(jnp.float32)
        if obs.action_valid_mask is None:
            valid = jnp.ones(target.shape[:2], jnp.bool_)
        else:
            valid = obs.action_valid_mask
        if train and self.perturb_prob > 0:
            if obs.ode_unit_scale is None:
                raise ValueError("Start perturbation needs observation.ode_unit_scale (ODENormalizeStart)")
            std = jnp.asarray(self.perturb_std, jnp.float32) * obs.ode_unit_scale[..., :d]
            a0, target = ode_field.perturb_start(
                perturb_rng, a0, target, std, prob=self.perturb_prob, decay=self.perturb_decay
            )
        path = self._integrate(a0, s, c)
        return ode_field.path_loss(path, target, valid, vel_scale=self.vel_scale)

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        """Deterministic: integrate H steps from observation.ode_start. Extra kwargs (noise, RTC, ...) are ignored."""
        del rng
        if kwargs:
            logger.debug("Pi0ODE.sample_actions ignores %s", sorted(kwargs))
        if observation.ode_start is None:
            raise ValueError("Pi0ODE needs observation.ode_start at inference (see ode_policy.ODECarry)")
        obs = _model.preprocess_observation(None, observation, train=False)
        c = self.condition(obs)
        d = self.ode_dim
        path = self._integrate(obs.ode_start[..., :d].astype(jnp.float32), obs.state[..., :d].astype(jnp.float32), c)
        return jnp.pad(path, ((0, 0), (0, 0), (0, self.action_dim - d)))
