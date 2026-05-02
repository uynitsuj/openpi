import logging
import os

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# ---- Debug toggles (read once at import time, no tracing cost in prod) -----
# RTC_DEBUG=1 enables per-denoising-step jax.debug.print stats inside the RTC
# loop. Each print is a host-device callback (forces a GPU→CPU sync) — cheap
# in absolute terms but non-zero, so leave this off unless debugging.
# JAX_LOG_COMPILES=1 (set via env) makes JAX log every JIT compile event —
# useful to confirm the RTC denoiser is only compiled once, not per-call.
RTC_DEBUG: bool = os.environ.get("RTC_DEBUG", "").lower() in ("1", "true", "yes")
if RTC_DEBUG:
    logger.warning("RTC_DEBUG=1 — per-denoising-step jax.debug.print is ENABLED (adds latency)")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # --- Real-Time Chunking (RTC) --------------------------------------- #
        # When ``action_prefix`` is supplied, dispatch to the RTC-guided
        # denoiser (``sample_actions_rtc``). Otherwise this function is the
        # byte-identical original pre-RTC implementation — no behavior change
        # for existing callers.
        action_prefix: at.Float[at.Array, "b tp ad"] | None = None,
        inference_delay: int | at.Int[at.Array, ""] = 0,
        execution_horizon: int | None = None,
        # Max clamp on the RTC guidance weight. The raw schedule (c · inv_r2)
        # diverges at t=0 and t=1, so we clamp here. Default 2.0 is a safe
        # midpoint on π0-family policies assuming the action_prefix has been
        # normalized into model-space (see Policy._rtc_action_normalizer).
        # Symptoms of too-high: guided v_t sign-flips in the [rtc] trace and
        # new chunks start far from obs. Too-low: RTC barely tracks the
        # prefix and you'll see residual chunk-boundary jitter.
        max_guidance_weight: float | at.Float[at.Array, ""] = 2.0,
    ) -> _model.Actions:
        if action_prefix is not None:
            return self.sample_actions_rtc(
                rng,
                observation,
                num_steps=num_steps,
                noise=noise,
                action_prefix=action_prefix,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                max_guidance_weight=max_guidance_weight,
            )

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: at.Float[at.Array, "b tp ad"],
        inference_delay: int | at.Int[at.Array, ""] = 0,
        execution_horizon: int | None = None,
        # Max clamp on the RTC guidance weight. The raw schedule (c · inv_r2)
        # diverges at t=0 and t=1, so we clamp here. Default 2.0 is a safe
        # midpoint on π0-family policies assuming the action_prefix has been
        # normalized into model-space (see Policy._rtc_action_normalizer).
        # Symptoms of too-high: guided v_t sign-flips in the [rtc] trace and
        # new chunks start far from obs. Too-low: RTC barely tracks the
        # prefix and you'll see residual chunk-boundary jitter.
        max_guidance_weight: float | at.Float[at.Array, ""] = 2.0,
    ) -> _model.Actions:
        """Real-Time-Chunking-guided variant of ``sample_actions``.

        Identical to ``sample_actions`` except the inner denoising step adds a
        correction term that keeps the first ``execution_horizon`` actions of
        the generated chunk coherent with ``action_prefix`` (the unexecuted
        tail of the previously-emitted chunk that the robot is still committed
        to executing while this inference runs).

        Implementation mirrors Physical Intelligence's Kinetix RTC and
        LeRobot's ``RTCProcessor.denoise_step``, but uses ``jax.vjp`` in place
        of ``torch.autograd.grad`` so it integrates cleanly with the existing
        ``lax.while_loop``.

        Reference: https://arxiv.org/abs/2506.07339
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with the prefix forward pass (unchanged from sample_actions).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        T = self.action_horizon
        A = self.action_dim

        # Resolve execution_horizon. If passed as a JAX array / int, keep it
        # dynamic so value-only changes between inferences don't trigger a
        # re-JIT (only shape/dtype changes do).
        if execution_horizon is None:
            execution_horizon_dyn = jnp.asarray(T, dtype=jnp.float32)
        else:
            execution_horizon_dyn = jnp.minimum(
                jnp.asarray(execution_horizon, dtype=jnp.float32),
                jnp.asarray(T, dtype=jnp.float32),
            )

        # Right-pad the prefix to the model's chunk shape (T, A) with zeros —
        # handles both a shorter prefix near chunk drain (T_prev < T) AND a
        # narrower action dim from a task-specific client (A_prev < A, e.g.
        # YAM's bimanual 14-dim vs π0's shared 32-dim action head). Mirrors
        # LeRobot's RTCProcessor.denoise_step padding step.
        T_prev = action_prefix.shape[1]
        A_prev = action_prefix.shape[2]
        pad_width = (
            (0, 0),                         # batch
            (0, max(0, T - T_prev)),        # time / chunk index
            (0, max(0, A - A_prev)),        # action dim
        )
        action_prefix_padded = jnp.pad(action_prefix, pad_width)
        # And crop if the client sent more than we need along either axis.
        action_prefix_padded = action_prefix_padded[:, :T, :A]

        # LINEAR prefix-weight schedule:
        #   idx <  inference_delay        -> 1.0  (robot committed, enforce prefix)
        #   inference_delay <= idx < eh   -> linear ramp 1 -> 0
        #   idx >= execution_horizon      -> 0.0  (model free-forms)
        idx = jnp.arange(T, dtype=jnp.float32)
        d = jnp.asarray(inference_delay, dtype=jnp.float32)
        e = execution_horizon_dyn
        ramp = 1.0 - (idx - d) / jnp.maximum(e - d, 1.0)
        ramp = jnp.clip(ramp, 0.0, 1.0)
        prefix_weights_1d = jnp.where(idx < d, jnp.float32(1.0), ramp)
        prefix_weights_1d = prefix_weights_1d * (idx < e).astype(jnp.float32)
        prefix_weights = prefix_weights_1d[None, :, None]   # (1, T, 1)
        max_gw = jnp.asarray(max_guidance_weight, dtype=jnp.float32)

        def denoise(x_t, time):
            """Base velocity field (same forward pass used by the original step)."""
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon:])

        def step(carry):
            x_t, time = carry
            # VJP gives us the Jacobian-vector product (d x1_t / d x_t)^T @ err
            # in one backward pass — JAX equivalent of LeRobot's torch.autograd.
            # With has_aux=True, jax.vjp returns (primals_out, vjp_fn, aux) —
            # the "primal" is x1_t only, and v_t comes back via aux so we only
            # pay for one denoise() call.
            def x1_and_v(xt):
                vt = denoise(xt, time)
                x1 = xt - time * vt           # Euler projection to t=0
                return x1, vt                  # primal=x1 (vjp target), aux=vt
            x1_t, vjp_fn, v_t = jax.vjp(x1_and_v, x_t, has_aux=True)
            err = (action_prefix_padded - x1_t) * prefix_weights
            (correction,) = vjp_fn(err)

            # Time-dependent guidance weight (matches Kinetix / LeRobot):
            # small at start and end of denoising, peaks in the middle. The
            # 1e-6 offsets guard the divisions without affecting the schedule
            # in the interior of [0, 1].
            tau = 1.0 - time
            inv_r2 = ((1.0 - tau) ** 2 + tau ** 2) / ((1.0 - tau) ** 2 + 1e-6)
            c = (1.0 - tau) / (tau + 1e-6)
            guidance_weight = jnp.minimum(c * inv_r2, max_gw)

            # Per-step stats, Python-level gated (RTC_DEBUG env var, read once
            # at module import). jax.debug.print inside lax.while_loop is a
            # host-device callback and costs real latency, so keep off in prod.
            if RTC_DEBUG:
                jax.debug.print(
                    "[rtc] t={} gw={} |v_t|={} |corr|={} |err|={} |x1_t|={} |prefix|={}",
                    time,
                    guidance_weight,
                    jnp.sqrt(jnp.mean(v_t * v_t)),
                    jnp.sqrt(jnp.mean(correction * correction)),
                    jnp.sqrt(jnp.mean(err * err)),
                    jnp.sqrt(jnp.mean(x1_t * x1_t)),
                    jnp.sqrt(jnp.mean(action_prefix_padded * action_prefix_padded)),
                )

            v_t = v_t - guidance_weight * correction
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
