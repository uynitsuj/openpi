"""ABC-compatible clean-prefix flow matching without changing parameter shapes."""

import jax
import jax.numpy as jnp


def sample_prefix_lengths(prefix_rng, apply_rng, batch_shape, max_action_prefix, conditioning_prob):
    """ABC uses an exclusive upper bound, including unconditioned length-zero examples."""
    sampled = jax.random.randint(prefix_rng, batch_shape, 0, max_action_prefix)
    apply = jax.random.uniform(apply_rng, batch_shape) < conditioning_prob
    return jnp.where(apply, sampled, 0)


def training_inputs(actions, noise, time, prefix_lengths, *, prefix_noise=None):
    prefix_mask = jnp.arange(actions.shape[-2]) < prefix_lengths[..., None]
    token_time = jnp.where(prefix_mask, 0.0, time[..., None])
    x_t = (1.0 - token_time[..., None]) * actions + token_time[..., None] * noise
    if prefix_noise is not None:
        x_t = x_t + prefix_mask[..., None] * prefix_noise
    return x_t, token_time, prefix_mask


def flow_loss(prediction, target, prefix_mask):
    per_position = jnp.mean(jnp.square(prediction - target), axis=-1)
    if prefix_mask is None:
        return per_position
    postfix = jnp.logical_not(prefix_mask)
    # OpenPI takes mean() of this array. Normalize globally, as ABC does,
    # rather than giving each example equal weight regardless of postfix length.
    return per_position * postfix / jnp.maximum(jnp.mean(postfix), 1e-8)
