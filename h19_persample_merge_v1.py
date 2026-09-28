#!/usr/bin/env python3
"""Merge the per-GPU shards of h19_persample_features_v1.py into one file, checking the row count.

Runs on the node after the fan-out. EXPECTED_ROWS is the stride count over the corpus; the merge
refuses to write if any shard is missing rows or if a flat index appears twice.
"""
import glob, json, os, sys
import numpy as np

parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob("feats/shard_*.npz"))]
if not parts:
    sys.exit("no shards found")
idx = np.concatenate([p["flat_index"] for p in parts]); o = np.argsort(idx)
out = {k: np.concatenate([p[k] for p in parts])[o] for k in ("flat_index", "loss", "gnorm", "sketch", "hidden")}
exp = int(os.environ["EXPECTED_ROWS"]); n = len(out["flat_index"])
if n != exp:
    sys.exit(f"rows {n} != expected {exp}")
if len(np.unique(out["flat_index"])) != n:
    sys.exit("duplicate flat indices")
if not np.isfinite(out["loss"]).all():
    sys.exit("non-finite loss values")
np.savez("feats/persample_features.npz", **out, meta=parts[0]["meta"])
print(json.dumps({"rows": n, "loss_mean": float(out["loss"].mean()), "loss_sd": float(out["loss"].std()),
                  "gnorm_mean": float(out["gnorm"].mean()), "sketch_dim": int(out["sketch"].shape[1]),
                  "hidden_dim": int(out["hidden"].shape[1]), "shards": len(parts)}))
