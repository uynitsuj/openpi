#!/usr/bin/env python3
"""Per-anchor features from one forward pass of a trained pi0 checkpoint.

Directions A and B of curation_research/RESEARCH_DIRECTIONS_20260915.md share one pass.  For every
anchor visited (a stride over the corpus, sharded across GPUs) this emits:

  loss     the flow-matching loss at K noise/time draws, averaged        -> Sorscher difficulty (G)
  sketch   a random-sign sketch of the LAST-LAYER action-expert gradient  -> gradient compatibility (B)
  hidden   the mean pre-projection hidden state over the action tokens    -> the critic's state (A)
  gnorm    the norm of that last-layer gradient before sketching

Why the last layer.  The loss is mean over tokens and dims of (W h + b - u)^2, so its gradient
with respect to the output projection W is closed-form per sample: sum over tokens of the residual
times the hidden state.  That is a forward pass.  A full per-sample gradient through the expert is
a backward pass per sample and is not affordable over the corpus.  RoboDrop (arXiv 2609.10021)
scores full action-expert gradients; this is the last-layer variant of the same idea, and it is
named as such wherever the scores are used.

The dataset stack is the training one: create_torch_dataset + transform_dataset from the SAME data
config, with the RABC subset filter disabled so every anchor is visible.  preprocess_observation
runs with train=False, so no image augmentation: features are deterministic given the draws.

Flat index.  The dataset's flat index is recorded per row.  This codebase asserts (data_loader
precompute_valid_indices) that the flat order equals the parquet `index` column, so the join back
to (episode_index, frame_index) is done from the parquet at analysis time.
"""
from __future__ import annotations
import argparse, dataclasses, json, logging, pathlib, time
import numpy as np
import jax, jax.numpy as jnp
import torch

import openpi.training.config as _config
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.data_loader as _dl
from openpi.shared import nnx_utils

SKETCH_SEED = 20260915


def h19_features(self, rng, observation, actions):
    """Mirror of Pi0.compute_loss that also returns the residual and the hidden state."""
    observation = _model.preprocess_observation(None, observation, train=False)
    noise_rng, time_rng = jax.random.split(rng, 2)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    t = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    te = t[..., None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    h = suffix_out[:, -self.action_horizon:]                 # [B, H, W]
    v_t = self.action_out_proj(h)                            # [B, H, A]
    r = (v_t - u_t).astype(jnp.float32)                      # [B, H, A]
    loss = jnp.mean(jnp.square(r), axis=(-1, -2))            # [B]
    H, A = r.shape[1], r.shape[2]
    grad_w = jnp.einsum("bha,bhw->baw", r, h.astype(jnp.float32)) * (2.0 / (H * A))   # [B, A, W]
    return loss, grad_w.reshape(r.shape[0], -1), jnp.mean(h.astype(jnp.float32), axis=1), t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi0_put_bottles_mjwarp_no_rabc")
    ap.add_argument("--repo-id", default=None, help="override the data config repo_id; norm stats still come from the config")
    ap.add_argument("--ckpt-params", type=pathlib.Path, required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--sketch-dim", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: stop after this many anchors")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"create-only: {a.out} exists")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = _config.get_config(a.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    data_config = dataclasses.replace(data_config, reject_zero_weighted_samples=False)   # every anchor
    if a.repo_id:
        data_config = dataclasses.replace(data_config, repo_id=a.repo_id)
    ds = _dl.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    ds = _dl.transform_dataset(ds, data_config)
    N = len(ds)
    idx = np.arange(0, N, a.stride)[a.shard::a.nshards]
    if a.limit:
        idx = idx[: a.limit]
    logging.info(f"corpus {N:,} anchors; stride {a.stride}; shard {a.shard}/{a.nshards}; this shard {len(idx):,}")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()), batch_size=a.batch, shuffle=False, drop_last=False,
        num_workers=a.workers, collate_fn=_dl._collate_fn, persistent_workers=a.workers > 0,
    )

    model = cfg.model.load(_model.restore_params(a.ckpt_params, dtype=jnp.bfloat16))
    _pi0.Pi0.h19_features = h19_features
    feats = nnx_utils.module_jit(model.h19_features)

    S = None
    rng = jax.random.key(SKETCH_SEED + a.shard)
    out_loss, out_sketch, out_hidden, out_gnorm, out_idx = [], [], [], [], []
    t0 = time.time(); seen = 0
    for bi, batch in enumerate(loader):
        batch = jax.tree.map(np.asarray, batch)
        obs = _model.Observation.from_dict(batch)
        actions = jnp.asarray(batch["actions"])
        B = actions.shape[0]
        acc_loss = jnp.zeros((B,), jnp.float32); acc_g = None; acc_h = None
        for _ in range(a.draws):
            rng, sub = jax.random.split(rng)
            loss, g, h, _t = feats(sub, obs, actions)
            acc_loss = acc_loss + loss
            acc_g = g if acc_g is None else acc_g + g
            acc_h = h if acc_h is None else acc_h + h
        acc_loss = acc_loss / a.draws; acc_g = acc_g / a.draws; acc_h = acc_h / a.draws
        if S is None:
            D = int(acc_g.shape[1])
            S = jax.random.choice(jax.random.key(SKETCH_SEED), jnp.array([-1.0, 1.0], jnp.float32),
                                  shape=(D, a.sketch_dim)) / np.sqrt(a.sketch_dim)
            logging.info(f"last-layer gradient dim {D}; sketch to {a.sketch_dim}")
        gnorm = jnp.linalg.norm(acc_g, axis=1)
        sketch = acc_g @ S
        out_loss.append(np.asarray(acc_loss)); out_gnorm.append(np.asarray(gnorm))
        out_sketch.append(np.asarray(sketch, dtype=np.float16)); out_hidden.append(np.asarray(acc_h, dtype=np.float16))
        out_idx.append(idx[seen: seen + B]); seen += B
        if bi % 20 == 0:
            el = time.time() - t0
            logging.info(f"batch {bi}  anchors {seen:,}/{len(idx):,}  {seen / max(el, 1e-6):.1f}/s  loss {float(acc_loss.mean()):.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, flat_index=np.concatenate(out_idx), loss=np.concatenate(out_loss),
             gnorm=np.concatenate(out_gnorm), sketch=np.concatenate(out_sketch), hidden=np.concatenate(out_hidden),
             meta=json.dumps({"config": a.config, "repo_id": data_config.repo_id, "ckpt_params": str(a.ckpt_params), "stride": a.stride,
                              "shard": a.shard, "nshards": a.nshards, "draws": a.draws, "sketch_dim": a.sketch_dim,
                              "sketch_seed": SKETCH_SEED, "corpus_anchors": int(N), "rows": int(seen),
                              "gradient": "last-layer action_out_proj, closed form from residual x hidden",
                              "preprocess": "train=False, no augmentation"}))
    logging.info(f"wrote {a.out}  rows {seen:,}  wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
