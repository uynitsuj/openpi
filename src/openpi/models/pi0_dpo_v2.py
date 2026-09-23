"""Pi0 with a counterfactual-command separation loss.

The negative token fields must be captured before preprocess_observation, which
drops auxiliary observation fields. Positive and negative passes share noise
and time. Identical commands therefore make the contrastive gap, and its
gradient contribution, exactly zero.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
from typing_extensions import override

import openpi.models.model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0DpoConfig(pi0_config.Pi0Config):
    dpo_beta: float = 5.0
    dpo_lambda: float = 1.0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Dpo":
        return Pi0Dpo(self, rngs=nnx.Rngs(rng))


class Pi0Dpo(Pi0):
    def __init__(self, config: Pi0DpoConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.dpo_beta = float(config.dpo_beta)
        self.dpo_lambda = float(config.dpo_lambda)

    def _flow_loss(self, observation, x_t, time, u_t):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # preprocess_observation intentionally returns only model-standard fields.
        neg_tokens = observation.tokenized_prompt_neg
        neg_mask = observation.tokenized_prompt_neg_mask
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        te = time[..., None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        l_pos = self._flow_loss(observation, x_t, time, u_t)
        if neg_tokens is None or neg_mask is None:
            return l_pos
        obs_neg = dataclasses.replace(
            observation,
            tokenized_prompt=neg_tokens,
            tokenized_prompt_mask=neg_mask,
        )
        l_neg = self._flow_loss(obs_neg, x_t, time, u_t)
        gap = l_pos.mean(axis=-1) - l_neg.mean(axis=-1)
        dpo = self.dpo_lambda * jax.nn.softplus(self.dpo_beta * gap)
        return l_pos + dpo[..., None]
