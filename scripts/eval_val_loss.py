"""Post-hoc validation loss for siemens abc-layout checkpoints.

Evaluates model.compute_loss(train=False) over the dataset's held-out val/
episodes (the 8 episodes export_abc_layout_job.py reserves), using the exact
training transform pipeline (repack -> data -> Normalize(norm stats) -> model
transforms) so the numbers are comparable to the logged train loss. Station
filters from the config apply to val episodes too (a ZED-only policy is only
evaluated on ZED val episodes).

Deterministic: the flow-matching time/noise rng is fold_in(seed, batch_idx).

Usage (needs a free GPU; pin one with CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_val_loss.py \
        --config-name pi05_siemens_packing_abcloader_v2_zedonly_bs128 \
        --params-dir /nfs_old/.../14999/params --step 14999 \
        --out-json val_losses.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--params-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-json", required=True, help="results are appended to this JSON list")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="data pipeline only, no model")
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
    if not data_config.abc_layout:
        raise ValueError(f"{args.config_name} is not an abc-layout config; no val split exists")

    val_root = HF_LEROBOT_HOME / data_config.repo_id / "val"
    station_types = getattr(data_config, "abc_station_types", None)
    # The v2/v3 exports' 8 val episodes are all D405 (contiguous-slice split). For a
    # station-filtered (ZED-only) config that leaves zero in-distribution val episodes —
    # fall back to the unfiltered val set and mark the metric as OOD transfer.
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
    print(f"val: {n_eps} episodes, {n} frames -> {n_batches} batches of {bs} (station filter: {station_types})")

    def collate(items):
        return jax.tree.map(lambda *xs: np.stack(xs), *items)

    if args.dry_run:
        batch = collate([ds[j] for j in range(bs)])
        obs = _model.Observation.from_dict(batch)
        print("dry run OK; actions", batch["actions"].shape,
              "state", obs.state.shape, "images", {k: v.shape for k, v in obs.images.items()})
        return

    model = cfg.model.load(_model.restore_params(args.params_dir, dtype=jnp.bfloat16))
    loss_fn = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))

    total, count = 0.0, 0
    for i in range(n_batches):
        batch = collate([ds[j] for j in range(i * bs, (i + 1) * bs)])
        obs = _model.Observation.from_dict(batch)
        rng = jax.random.fold_in(jax.random.key(0), i)
        loss = loss_fn(rng, obs, batch["actions"], train=False)  # (b, action_horizon)
        loss = np.asarray(loss, dtype=np.float64)
        total += float(loss.mean(axis=-1).sum())
        count += loss.shape[0]
        if i % 20 == 0:
            print(f"batch {i}/{n_batches}: running val_loss={total / max(count, 1):.6f}")

    result = {
        "config": args.config_name,
        "step": args.step,
        "params_dir": args.params_dir,
        "val_loss": total / count,
        "val_frames": count,
        "val_episodes": n_eps,
        "batch_size": bs,
        "ood_d405": ood,
    }
    out = Path(args.out_json)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append(result)
    out.write_text(json.dumps(rows, indent=2))
    print(f"RESULT {args.config_name} step={args.step} val_loss={result['val_loss']:.6f} ({count} frames)")


if __name__ == "__main__":
    main()
