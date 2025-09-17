import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")
import time

def sample_tr_pairs(
    B: int,
    rng: jax.Array,
    *,
    mode: str = "logit_normal",   # "uniform_pair" | "logit_normal" | "factorized"
    mu: float = -0.4,             # for logit_normal
    sigma: float = 1.0,           # for logit_normal
    beta_a: float = 2.0,          # for factorized step size h ~ Beta(a,b)
    beta_b: float = 8.0,          # for factorized step size
    p_fm: float = 0.25,           # probability to set r=t (FM target, no JVP term effect)
    p_end: float = 0.05           # probability to force (t,r)=(1,0) (endpoint emphasis)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (t, r) with shape (B,), always t >= r, both in [0,1]."""
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    if mode == "uniform_pair":
        a = jax.random.uniform(rng1, (B,))
        b = jax.random.uniform(rng2, (B,))
        t = jnp.maximum(a, b)
        r = jnp.minimum(a, b)

    elif mode == "logit_normal":
        z1 = jax.random.normal(rng1, (B,))
        z2 = jax.random.normal(rng2, (B,))
        a  = jax.nn.sigmoid(mu + sigma * z1)
        b  = jax.nn.sigmoid(mu + sigma * z2)
        t  = jnp.maximum(a, b)
        r  = jnp.minimum(a, b)

    elif mode == "factorized":
        # sample target time t, then a step size h, and set r = max(0, t - h)
        z  = jax.random.normal(rng1, (B,))
        t  = jax.nn.sigmoid(mu + sigma * z)
        h  = jax.random.beta(rng2, beta_a, beta_b, (B,))   # typically small steps
        r  = jnp.clip(t - h, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Mix in FM-style pairs r=t (useful for stability / lower JVP variance)
    use_fm = jax.random.bernoulli(rng3, p_fm, (B,))
    r = jnp.where(use_fm, t, r)

    # Mix in a small fraction of exact endpoint pairs (1,0)
    use_end = jax.random.bernoulli(rng4, p_end, (B,))
    t = jnp.where(use_end, jnp.ones_like(t), t)
    r = jnp.where(use_end, jnp.zeros_like(r), r)

    return t, r



# ---- constants for MeanFlow ----
TIME_EMB_SCALE = 0.05     # amplitude scale (reduces du/dt ~×0.05; dudt2 ~×0.0025)
T_MIN_PERIOD   = 0.1      # was 4e-3; increase to low-pass the spectrum
T_MAX_PERIOD   = 2.0

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


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
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
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        # unbatched impl
        # for name in obs.images:
        #     image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

        #     tokens.append(image_tokens)
        #     input_mask.append(
        #         einops.repeat(
        #             obs.image_masks[name],
        #             "b -> b s",
        #             s=image_tokens.shape[1],
        #         )
        #     )
        #     # image tokens attend to each other
        #     ar_mask += [False] * image_tokens.shape[1]


        # batched:
        image_names = list(obs.images.keys())
        images = list(obs.images.values())
        if obs.past_head_images is not None:
            for t in range(obs.past_head_images.shape[1]):
                images.append(obs.past_head_images[:, t])
        else:
            assert len(images) == len(image_names)
        stacked_images = jnp.stack(images, axis=1)

        batch_size, num_cams = stacked_images.shape[:2]
        reshaped_images = stacked_images.reshape(-1, *stacked_images.shape[2:])

        all_image_tokens, _ = self.PaliGemma.img(reshaped_images, train=False)
        all_image_tokens = all_image_tokens.reshape(batch_size, num_cams, all_image_tokens.shape[1], -1)

        for i, name in enumerate(image_names):
            image_tokens = all_image_tokens[:, i] 
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
        if obs.past_head_images is not None:
            for t in range(obs.past_head_images.shape[1], 0, -1):
                image_tokens = all_image_tokens[:, -t]
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(obs.image_masks[image_names[0]], "b -> b s", s=image_tokens.shape[1]) # assumes image_names[0] is the "head" camera
                )

        # Set attention masks
        ar_mask = [False] * (image_tokens.shape[1] * num_cams)

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
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

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
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # @override
    # def compute_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     actions: _model.Actions,
    #     *,
    #     train: bool = False,
    # ) -> at.Float[at.Array, "*b ah"]:
    #     """
    #     MeanFlow:  L = || u_theta(z_t, r, t) - stopgrad( v_t - (t-r) * d/dt u_theta(z_t, r, t) ) ||^2
    #     with z_t = (1-t) * x + t * e,  v_t = e - x.  JVP uses suffix-only + detached kv-cache.
    #     """
    #     preprocess_rng, noise_rng, tr_rng, mf_rng = jax.random.split(rng, 4)
    #     observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    #     B, S, A = actions.shape
    #     width = self.action_in_proj.out_features
    #     assert (width % 2) == 0

    #     # ----- construct flow path and times -----
    #     e = jax.random.normal(noise_rng, actions.shape)      # prior sample
    #     # t_raw = jax.random.uniform(t_rng_a, (B,))
    #     # r_raw = jax.random.uniform(t_rng_b, (B,))
    #     # t = jnp.maximum(t_raw, r_raw)
    #     # r = jnp.minimum(t_raw, r_raw)

    #     t, r = sample_tr_pairs(
    #         B, tr_rng,
    #         mode="logit_normal",  # try "factorized" or "uniform_pair"
    #         mu=-0.4, sigma=1.0, # follows implementation in https://github.com/Gsunshine/meanflow/blob/main/meanflow.py
    #         beta_a=2.0, beta_b=8.0,
    #         p_fm=0.25,            # ~25% FM-style targets
    #         p_end=0.15            # a few endpoint pairs
    #     )

    #     # mix-in FM (r=t) updates to reduce JVP frequency (paper finds ~25% good)
    #     mf_ratio = 0.75 # 0.25 flow-matching instantaneous velocity targets
    #     use_mf = jax.random.bernoulli(mf_rng, mf_ratio, (B,))
    #     r = jnp.where(use_mf, r, t)

    #     t_exp = t[:, None, None]
    #     z_t = (1.0 - t_exp) * actions + t_exp * e
    #     v_t = e - actions

    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions_pref = jnp.cumsum(prefix_mask, axis=1) - 1

    #     llm_call = jax.checkpoint(self.PaliGemma.llm)

    #     (_, _), kv_cache = llm_call([prefix_tokens, None],
    #                                 mask=prefix_attn_mask,
    #                                 positions=positions_pref)

    #     def _suffix_tokens(obs, z_in, r_in, t_in):
    #         # state token
    #         state_token = self.state_proj(obs.state)[:, None, :]                 # (B,1,W)
    #         # action tokens
    #         act_tok = self.action_in_proj(z_in)                                  # (B,S,W)
    #         # embed (t, del_t)
    #         dt_in = t_in - r_in
    #         half = width // 2
    #         t_emb  = posemb_sincos(t_in,  half, T_MIN_PERIOD, T_MAX_PERIOD).astype(jnp.float32)                       # (B, W/2)
    #         dt_emb = posemb_sincos(dt_in, half, T_MIN_PERIOD, T_MAX_PERIOD).astype(jnp.float32)                       # (B, W/2)
    #         time_emb = TIME_EMB_SCALE * jnp.concatenate([t_emb, dt_emb], axis=-1)                 # (B, W)
    #         time_tok = einops.repeat(time_emb, "b w -> b s w", s=S)              # (B,S,W)
    #         # fuse
    #         atok = jnp.concatenate([act_tok, time_tok], axis=-1)                 # (B,S,2W)
    #         atok = self.action_time_mlp_in(atok)
    #         atok = nnx.swish(atok)
    #         atok = self.action_time_mlp_out(atok)                                # (B,S,W)
    #         toks = jnp.concatenate([state_token, atok], axis=1)                  # (B,1+S,W)
    #         mask = jnp.ones((B, 1 + S), dtype=jnp.bool_)                         # (B,1+S)
    #         # AR: state True; first action True; rest False
    #         ar = jnp.array([True] + [True] + [False] * (S - 1))
    #         return toks, mask, ar

    #     def _u_with_cache(z_in, r_in, t_in, kv):
    #         suf_toks, suf_mask, suf_ar = _suffix_tokens(observation, z_in, r_in, t_in)
    #         suf_suf_mask  = make_attn_mask(suf_mask, suf_ar)                     # (B, Suf, Suf)
    #         suf_pref_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suf_toks.shape[1])
    #         full_mask = jnp.concatenate([suf_pref_mask, suf_suf_mask], axis=-1)  # (B, Suf, P+Suf)
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1
    #         (p_out, s_out), _ = llm_call([None, suf_toks],
    #                                     mask=full_mask,
    #                                     positions=positions,
    #                                     kv_cache=kv)
    #         assert p_out is None
    #         return self.action_out_proj(s_out[:, -S:])                           # (B,S,A)

    #     u_val = _u_with_cache(z_t, r, t, kv_cache)

    #     # JVP path: detach kv_cache so JVP doesn’t drag the prefix graph along
    #     kv_detached = jax.lax.stop_gradient(kv_cache)
    #     u_jvp_fun = lambda zz, rr, tt: _u_with_cache(zz, rr, tt, kv_detached)

    #     u_val_j, du_dt = jax.jvp(
    #         u_jvp_fun,
    #         primals=(z_t, r, t),
    #         tangents=(v_t, jnp.zeros_like(r), jnp.ones_like(t)),
    #     )

    #     # MeanFlow target and loss
    #     scale = (t - r)[:, None, None]
    #     u_tgt = v_t - scale * du_dt
    #     err = u_val - jax.lax.stop_gradient(u_tgt)

    #     resid = u_val - jax.lax.stop_gradient(u_tgt)
    #     l_ex  = jnp.mean(resid**2, axis=(1,2))
    #     loss  = (l_ex / jax.lax.stop_gradient((l_ex + 1e-2)**1.0)).mean()

    #     # Scalars to inspect
    #     vt2     = jnp.mean(jnp.square(v_t))          # ~ 1 + Var(actions)
    #     dudt2   = jnp.mean(jnp.square(du_dt))        # magnitude of time derivative
    #     mean_dt = jnp.mean(jnp.abs(t - r))           # ~0.33 if U(0,1) ordered
    #     u2      = jnp.mean(jnp.square(u_val))
    #     loss    = jnp.mean(jnp.square(err))          # final per-example mean already reduced
        
    #     v_hat   = u_val + ((t - r)[..., None, None]) * du_dt   # implied instantaneous velocity
    #     fm_proxy = jnp.mean(jnp.square(v_hat - v_t))
        
    #     # # Optional NaN/Inf check
    #     # nan_flags = jnp.array([
    #     #     jnp.isnan(loss), jnp.isinf(loss),
    #     #     jnp.isnan(vt2),  jnp.isinf(vt2),
    #     #     jnp.isnan(dudt2),jnp.isinf(dudt2)
    #     # ])
    #     num = jnp.sum(u_val * (v_t - ((t - r)[..., None, None]) * du_dt), axis=(-2,-1))
    #     den = jnp.sqrt(jnp.sum(u_val**2, axis=(-2,-1)) *
    #                 jnp.sum((v_t - ((t - r)[..., None, None]) * du_dt)**2, axis=(-2,-1)) + 1e-8)
    #     cos_u_utgt = jnp.mean(num / (den + 1e-8))

    #     do_print = jnp.array(True)

    #     def _printer(_):
    #         return jax.debug.print(
    #             ("loss={:.4f} vt2={:.4f} dudt2={:.4f} |dt|={:.4f} u2={:.4f} fm_proxy={:.4f} cos_u_utgt={:.4f}"),
    #             loss, vt2, dudt2, mean_dt, u2, fm_proxy, cos_u_utgt
    #         )

    #     jax.lax.cond(do_print, _printer, lambda _: None, operand=None)
        
    #     return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
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

    # @override
    # def sample_actions(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     *,
    #     num_steps: int | at.Int[at.Array, ""] = 3,
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     B = observation.state.shape[0]
    #     S = self.action_horizon
    #     W = self.action_in_proj.out_features
    #     assert (W % 2) == 0

    #     # initial noise at t=1
    #     e = jax.random.normal(rng, (B, S, self.action_dim))

    #     # prefix once + KV cache
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions_pref = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm([prefix_tokens, None],
    #                                     mask=prefix_attn_mask, positions=positions_pref)

    #     def _suffix_tokens_meanflow_step(z_in, r_in, t_in):
    #         state_token = self.state_proj(observation.state)[:, None, :]
    #         action_tokens = self.action_in_proj(z_in)
    #         dt_in = t_in - r_in
    #         half = W // 2
    #         t_emb  = posemb_sincos(t_in,  half, min_period=T_MIN_PERIOD, max_period=T_MAX_PERIOD)
    #         dt_emb = posemb_sincos(dt_in, half, min_period=T_MIN_PERIOD, max_period=T_MAX_PERIOD)
    #         time_emb = TIME_EMB_SCALE * jnp.concatenate([t_emb, dt_emb], axis=-1)
    #         time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=S)
    #         atok = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    #         atok = self.action_time_mlp_in(atok)
    #         atok = nnx.swish(atok)
    #         atok = self.action_time_mlp_out(atok)
    #         toks = jnp.concatenate([state_token, atok], axis=1)                    # (B,1+S,W)
    #         mask = jnp.ones((B, 1 + S), dtype=jnp.bool_)
    #         ar = jnp.array([True] + [True] + [False] * (S - 1))                    # (1+S,)
    #         return toks, mask, ar

    #     def _u_step(z_in, r_in, t_in):
    #         suf_toks, suf_mask, suf_ar = _suffix_tokens_meanflow_step(z_in, r_in, t_in)
    #         # how suffix attends to prefix + suffix
    #         suf_suf_mask = make_attn_mask(suf_mask, suf_ar)                        # (B, Suf, Suf)
    #         suf_pref_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suf_toks.shape[1])
    #         full_mask = jnp.concatenate([suf_pref_mask, suf_suf_mask], axis=-1)    # (B, Suf, P+Suf)
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1
    #         (p_out, s_out), _ = self.PaliGemma.llm([None, suf_toks],
    #                                             mask=full_mask,
    #                                             positions=positions,
    #                                             kv_cache=kv_cache)
    #         assert p_out is None
    #         return self.action_out_proj(s_out[:, -S:])  # (B,S,A)

    #     steps = int(num_steps)
    #     if steps <= 1:
    #         t_vec = jnp.ones((B,), dtype=jnp.float32)
    #         r_vec = jnp.zeros((B,), dtype=jnp.float32)
    #         u_e = _u_step(e, r_vec, t_vec)
    #         return e - u_e

    #     dt = 1.0 / steps

    #     def body(i, carry):
    #         z, t_scalar = carry
    #         r_scalar = t_scalar - dt
    #         # broadcast scalar times to (B,)
    #         t_vec = jnp.full((B,), t_scalar, dtype=jnp.float32)
    #         r_vec = jnp.full((B,), r_scalar, dtype=jnp.float32)
    #         u = _u_step(z, r_vec, t_vec)
    #         z_next = z - dt * u
    #         return (z_next, r_scalar)

    #     z_final, _ = jax.lax.fori_loop(0, steps, body, (e, jnp.array(1.0, dtype=jnp.float32)))
    #     return z_final
