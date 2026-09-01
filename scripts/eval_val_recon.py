"""abc-style validation recon error for siemens abc-layout checkpoints.

Mirrors abc_minimal/train_loop.py's val pass: run the full sampling path
(model.sample_actions, all denoising steps — the same path serving uses) over the
dataset's held-out val/ episodes and report MSE between sampled and ground-truth
actions in normalized space. Unlike the flow-matching val loss (a proxy averaged
over random noise levels), this measures the actions the policy would actually emit.

Reported:
  recon_mse_14   — over the 14 real yam dims (headline; abc's actions are 14-dim)
  recon_mse_32   — over the padded 32-dim vector (diluted by learned-zero pads)
  recon_mse_arm / recon_mse_gripper — 12 arm dims vs 2 gripper dims

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_val_recon.py \
        --config-name pi05_siemens_packing_abcloader_v3cc_bs128 \
        --params-dir /nfs_old/.../14999/params --step 14999 --out-json recon.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # yam: [l_arm(6), l_grip, r_arm(6), r_grip]
GRIPPER_DIMS = [6, 13]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--params-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from lerobot.utils.constants import HF_LEROBOT_HOME

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.training.data_loader as _dl
    from openpi.shared import nnx_utils
    from openpi.training.abc_layout_dataset import AbcLayoutDataset

    cfg = _config.get_config(args.config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    val_root = HF_LEROBOT_HOME / data_config.repo_id / "val"
    station_types = getattr(data_config, "abc_station_types", None)
    ood = False
    try:
        ds = AbcLayoutDataset(val_root, action_horizon=cfg.model.action_horizon, station_types=station_types)
    except ValueError:
        ood = True
        ds = AbcLayoutDataset(val_root, action_horizon=cfg.model.action_horizon, station_types=None)
    n_eps = len(ds._episodes)  # noqa: SLF001
    ds = _dl.transform_dataset(ds, data_config)
    n = len(ds)
    bs = args.batch_size
    n_batches = n // bs
    print(f"val: {n_eps} episodes, {n} frames -> {n_batches} batches of {bs}")

    def collate(items):
        return jax.tree.map(lambda *xs: np.stack(xs), *items)

    model = cfg.model.load(_model.restore_params(args.params_dir, dtype=jnp.bfloat16))
    sample_fn = nnx_utils.module_jit(model.sample_actions)  # default num_steps: serving behavior

    sums = {"14": 0.0, "32": 0.0, "arm": 0.0, "grip": 0.0}
    count = 0
    for i in range(n_batches):
        batch = collate([ds[j] for j in range(i * bs, (i + 1) * bs)])
        obs = _model.Observation.from_dict(batch)
        gt = np.asarray(batch["actions"], dtype=np.float64)  # (B, H, 32) normalized
        rng = jax.random.fold_in(jax.random.key(0), i)
        pred = np.asarray(sample_fn(rng, obs), dtype=np.float64)  # (B, H, 32)
        err2 = (pred - gt) ** 2
        sums["32"] += float(err2.mean(axis=(1, 2)).sum())
        sums["14"] += float(err2[..., :14].mean(axis=(1, 2)).sum())
        sums["arm"] += float(err2[..., ARM_DIMS].mean(axis=(1, 2)).sum())
        sums["grip"] += float(err2[..., GRIPPER_DIMS].mean(axis=(1, 2)).sum())
        count += err2.shape[0]
        if i % 20 == 0:
            print(f"batch {i}/{n_batches}: running recon_mse_14={sums['14'] / max(count, 1):.6f}")

    result = {
        "config": args.config_name,
        "step": args.step,
        "params_dir": args.params_dir,
        "recon_mse_14": sums["14"] / count,
        "recon_mse_32": sums["32"] / count,
        "recon_mse_arm": sums["arm"] / count,
        "recon_mse_gripper": sums["grip"] / count,
        "val_frames": count,
        "val_episodes": n_eps,
        "ood_d405": ood,
        "note": "normalized space; sample_actions with serving-default denoising steps; rng fold_in(0, batch)",
    }
    out = Path(args.out_json)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append(result)
    out.write_text(json.dumps(rows, indent=2))
    print(f"RESULT {args.config_name} step={args.step} recon_mse_14={result['recon_mse_14']:.6f} "
          f"arm={result['recon_mse_arm']:.6f} gripper={result['recon_mse_gripper']:.6f} ({count} frames)")


if __name__ == "__main__":
    main()
