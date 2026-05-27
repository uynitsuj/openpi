from collections.abc import Iterator, Sequence
import dataclasses
from dataclasses import dataclass, field, asdict
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
import time
import typing
from typing import List, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RejectionSamplingTransformedDataset(Dataset[T_co]):
    """TransformedDataset variant that rejects post-transform samples whose
    ``sample_weights`` is zero (or absent floats <= 0) and replaces them with a
    fresh uniform random draw from the dataset until acceptance, capped at
    ``max_retries``.

    Why: ``ComputeRABCWeights`` may emit ``sample_weights == 0`` for chunks
    rejected by the final-action / threshold gate. Without rejection, those
    samples still consume forward+backward FLOPs but contribute exactly zero
    gradient, and the *effective* batch size fluctuates step-to-step. With
    rejection, every sample in every batch has weight > 0 — stable effective
    batch size, no wasted GPU compute. No-op for samples that don't carry a
    ``sample_weights`` key (rabc disabled).
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        max_retries: int = 64,
        weight_key: str = "sample_weights",
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._max_retries = max_retries
        self._weight_key = weight_key
        self._n = len(dataset)
        # Per-worker numpy RNG — torch DataLoader reseeds workers via
        # worker_init_fn when set (and via the default torch seeding otherwise),
        # so identical workers won't all collide on the same indices.
        self._rng = np.random.default_rng()

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = int(index)
        sample = self._transform(self._dataset[idx])
        for _ in range(self._max_retries):
            w = sample.get(self._weight_key) if isinstance(sample, dict) else None
            if w is None:
                return sample  # No weight emitted (rabc disabled / non-rabc transform)
            wf = float(np.asarray(w).reshape(-1)[0])
            if wf > 0.0:
                return sample
            idx = int(self._rng.integers(0, self._n))
            sample = self._transform(self._dataset[idx])
        # Retry cap exceeded — return the last sample (gradient still ~0,
        # but training proceeds). Rare in practice; >max_retries consecutive
        # rejections suggests reject-rate is pathologically high.
        return sample

    def __len__(self) -> int:
        return self._n


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    delta_ts = [t / dataset_meta.fps for t in range(action_horizon)]
    delta_timestamps = {key: delta_ts for key in data_config.action_sequence_keys}
    # Extra keys (e.g., repromo_signed_magnitude) get a possibly-longer
    # window when extra_horizon_lookahead_frames > 0 (so RABC aggregators
    # like 'mean_lookahead' can see vel beyond the action chunk).
    extra_lookahead = getattr(data_config, "extra_horizon_lookahead_frames", 0)
    if extra_lookahead > 0:
        delta_ts_extra = [t / dataset_meta.fps for t in range(action_horizon + extra_lookahead)]
    else:
        delta_ts_extra = delta_ts
    for key in data_config.extra_horizon_keys:
        delta_timestamps[key] = delta_ts_extra
    episodes = list(data_config.episodes) if data_config.episodes is not None else None
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.04,  # 40ms tolerance for slight FPS mismatch (e.g., 29.58 vs 30)
        episodes=episodes,
    )

    if data_config.prompt_from_task:
        # v3 lerobot returns tasks as a pandas DataFrame (task as index, task_index as column);
        # PromptFromLeRobotTask expects dict[int, str]. Coerce both shapes here.
        tasks = dataset_meta.tasks
        if hasattr(tasks, "to_dict"):
            # DataFrame: task → task_index. Invert to {int(task_index): task_string}.
            tasks = {int(idx): str(task) for task, idx in tasks["task_index"].items()}
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def _find_rabc_transform(data_config: _config.DataConfig) -> _transforms.ComputeRABCWeights | None:
    """Locate the ComputeRABCWeights instance in data_config (if any)."""
    for t in data_config.data_transforms.inputs:
        if isinstance(t, _transforms.ComputeRABCWeights):
            return t
    return None


def _find_scizor_sidecar_transform(
    data_config: _config.DataConfig,
) -> _transforms.LoadScizorSidecar | None:
    """Locate the LoadScizorSidecar instance in data_config (if any)."""
    for t in data_config.data_transforms.inputs:
        if isinstance(t, _transforms.LoadScizorSidecar):
            return t
    return None


def _rabc_cache_key(
    repo_id: str,
    action_horizon: int,
    rabc: _transforms.ComputeRABCWeights,
    episodes: tuple[int, ...] | None,
    lookahead_frames: int = 0,
    scizor_sidecar_path: str | None = None,
) -> str:
    """Stable hash for the precomputed valid_indices file. Includes everything
    that changes which samples pass the gate: dataset, horizon, RABC params,
    the episode-filter tuple, the lookahead frame count, and (for
    mode='scizor_anchor') the sidecar parquet's path / mtime / size so
    swapping SCIZOR checkpoints invalidates the cache correctly."""
    payload: dict = {
        "repo_id": repo_id,
        "action_horizon": action_horizon,
        "rabc": dataclasses.asdict(rabc),
        "episodes": sorted(episodes) if episodes is not None else "all",
        "lookahead_frames": lookahead_frames,
    }
    if scizor_sidecar_path:
        resolved = str(pathlib.Path(scizor_sidecar_path).resolve())
        stat = pathlib.Path(resolved).stat()
        payload["scizor_sidecar"] = {
            "path": resolved, "mtime": stat.st_mtime, "size": stat.st_size,
        }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def precompute_valid_indices(
    repo_id: str,
    action_horizon: int,
    rabc: _transforms.ComputeRABCWeights,
    *,
    episodes: tuple[int, ...] | None = None,
    lookahead_frames: int = 0,
    cache_dir: pathlib.Path | None = None,
    spot_check: bool = True,
    scizor_sidecar_path: str | None = None,
    scizor_score_column: str = "scizor_score",
) -> np.ndarray:
    """Walk the lerobot v3 parquets, apply the RABC gate offline, and return
    the flat (post-episode-filter) indices of samples whose weight > 0.

    This is the cheap precompute path: it reads only the velocity (and quality,
    when needed) columns directly from parquet — no video decode, no image
    transforms. Output indices are positions into the *filtered* lerobot
    dataset (i.e. ``dataset[i]`` for ``i in valid_indices``), which composes
    cleanly with ``torch.utils.data.Subset``.

    Caches to ``<cache_dir>/<hash>.npy``. ``spot_check=True`` cross-validates
    a handful of indices against a fresh ``LeRobotDataset.__getitem__`` to
    catch any flat-indexing drift.

    Cache invalidation is keyed on (repo_id, action_horizon, rabc params,
    episodes tuple) — *not* parquet mtime. If you re-extract the velocity
    column or re-inject quality scores into the same repo_id, manually purge
    ``~/.cache/openpi/rabc_valid_indices/`` so the next launch recomputes.
    """
    if cache_dir is None:
        env_dir = os.environ.get("RABC_CACHE_DIR")
        cache_dir = (
            pathlib.Path(env_dir)
            if env_dir
            else pathlib.Path.home() / ".cache" / "openpi" / "rabc_valid_indices"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{_rabc_cache_key(repo_id, action_horizon, rabc, episodes, lookahead_frames, scizor_sidecar_path)}.npy"
    )
    logging.info(
        f"[rabc_precompute] {repo_id} H={action_horizon} mode={rabc.mode} "
        f"episodes={'all' if episodes is None else f'{len(episodes)} eps'} "
        f"cache={cache_path.name}"
    )
    if cache_path.exists():
        valid = np.load(cache_path)
        logging.info(
            f"[rabc_precompute] cache hit: {cache_path} ({len(valid):,} valid indices)"
        )
        return valid

    t0 = time.time()
    root = pathlib.Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id
    if not root.exists():
        raise FileNotFoundError(f"lerobot dataset not found at {root}")

    import pyarrow.parquet as _pq  # noqa: PLC0415

    # 1) Read episodes metadata: episode_index, length, dataset_from/to_index.
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_df = _pq.read_table(ep_files).to_pandas()
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)

    # ── SCIZOR sidecar branch ────────────────────────────────────────────
    # Read suboptimality scores from an external parquet (NOT the lerobot
    # dataset). Per-anchor-frame keep/drop matches the paper's filter
    # exactly (no aggregation across action_horizon).
    if rabc.mode == "scizor_anchor":
        if not scizor_sidecar_path:
            raise ValueError(
                "precompute_valid_indices: rabc.mode='scizor_anchor' requires "
                "scizor_sidecar_path to be set."
            )
        scores_by_ep = _transforms._load_scizor_sidecar(
            scizor_sidecar_path, scizor_score_column,
        )
        if episodes is None:
            ordered_eps = sorted(ep_df["episode_index"].astype(int).tolist())
        else:
            all_eps = set(ep_df["episode_index"].astype(int).tolist())
            missing = [e for e in episodes if e not in all_eps]
            if missing:
                raise ValueError(f"episodes not in dataset metadata: {missing[:10]}")
            ordered_eps = sorted(int(e) for e in episodes)
        ep_lookup = {int(r["episode_index"]): r for _, r in ep_df.iterrows()}
        filtered_indices: list[int] = []
        decide = rabc.decide_weight
        cursor = 0
        for ep in ordered_eps:
            meta = ep_lookup[int(ep)]
            L = int(meta["length"])
            ep_scores = scores_by_ep.get(int(ep))
            if ep_scores is None:
                # Episode missing from sidecar — drop the whole episode.
                cursor += L
                continue
            for offset in range(L):
                anchor_idx = min(offset, len(ep_scores) - 1)
                w = decide(np.asarray([ep_scores[anchor_idx]], dtype=np.float32))
                if w > 0.0:
                    filtered_indices.append(cursor)
                cursor += 1
        valid = np.asarray(filtered_indices, dtype=np.int64)
        n_filtered = cursor
        keep_frac = (len(valid) / n_filtered) if n_filtered > 0 else 0.0
        elapsed = time.time() - t0
        logging.info(
            f"[rabc_precompute][scizor] {repo_id} ε_s={rabc.threshold} "
            f"mode={rabc.scizor_weight_mode}: {len(valid):,}/{n_filtered:,} "
            f"kept ({keep_frac:.1%}) in {elapsed:.1f}s"
        )
        if spot_check and len(valid) > 0:
            _spot_check_scizor_valid_indices(
                repo_id, action_horizon, rabc, episodes, valid,
                scizor_sidecar_path, scizor_score_column,
            )
        np.save(cache_path, valid)
        logging.info(f"[rabc_precompute] cached → {cache_path}")
        return valid

    # 2) Read data parquets. Only pull the columns we need (velocity, optional
    # quality, and the parquet 'index' column to verify global flat ordering).
    needed_cols = ["index", "episode_index", "frame_index"]
    vel_col = None
    for c in ("repromo_signed_magnitude", "rorm_velocity"):
        if c in _pq.read_schema(sorted((root / "data").rglob("*.parquet"))[0]).names:
            vel_col = c
            break
    if vel_col is None:
        raise KeyError(
            f"No velocity column ('repromo_signed_magnitude' or 'rorm_velocity') in {root}"
        )
    needed_cols.append(vel_col)

    use_q = rabc.mode != "velocity_only" and rabc.q_min is not None and rabc.q_max is not None
    q_col = None
    if use_q:
        for c in ("repromo_quality", "rorm_q"):
            if c in _pq.read_schema(sorted((root / "data").rglob("*.parquet"))[0]).names:
                q_col = c
                break
        if q_col is not None:
            needed_cols.append(q_col)

    data_files = sorted((root / "data").rglob("*.parquet"))
    df = _pq.read_table(data_files, columns=needed_cols).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)
    # Sanity: index should be a contiguous 0..N-1 range across all episodes.
    n_global = len(df)
    if int(df["index"].iloc[0]) != 0 or int(df["index"].iloc[-1]) != n_global - 1:
        raise ValueError(
            f"parquet 'index' column is not a contiguous 0..N-1 range: "
            f"start={int(df['index'].iloc[0])}, end={int(df['index'].iloc[-1])}, n={n_global}"
        )
    vel_global = df[vel_col].to_numpy(dtype=np.float32)
    q_global = df[q_col].to_numpy(dtype=np.float32) if q_col is not None else None

    # 3) Build the global→filtered index mapping. lerobot iterates episodes in
    # the order given (or by episode_index when episodes is None).
    if episodes is None:
        ordered_eps = sorted(ep_df["episode_index"].astype(int).tolist())
    else:
        # lerobot sorts user-provided episodes internally before iterating, so
        # match that ordering — otherwise filtered flat indices won't line up
        # with dataset[k] when callers pass non-sorted tuples.
        all_eps = set(ep_df["episode_index"].astype(int).tolist())
        missing = [e for e in episodes if e not in all_eps]
        if missing:
            raise ValueError(f"episodes not in dataset metadata: {missing[:10]}")
        ordered_eps = sorted(int(e) for e in episodes)

    ep_lookup = {int(r["episode_index"]): r for _, r in ep_df.iterrows()}

    # Walk filtered episodes; for each frame, build the action_horizon window
    # the same way lerobot does (truncate at episode end + pad with last frame).
    filtered_indices: list[int] = []
    decide = rabc.decide_weight
    H = int(action_horizon) + int(lookahead_frames)
    cursor = 0  # filtered flat index cursor
    for ep in ordered_eps:
        meta = ep_lookup[int(ep)]
        g_from = int(meta["dataset_from_index"])
        g_to = int(meta["dataset_to_index"])
        L = int(meta["length"])
        if (g_to - g_from) != L:
            raise ValueError(
                f"ep {ep}: dataset_to-from={g_to-g_from} != length={L}"
            )
        ep_vel = vel_global[g_from:g_to]
        ep_q = q_global[g_from:g_to] if q_global is not None else None
        last_v = float(ep_vel[-1])
        for offset in range(L):
            end = offset + H
            if end <= L:
                window = ep_vel[offset:end]
            else:
                # Pad the tail with last-frame value, matching lerobot's
                # tolerance_s=0.04 end-of-episode behavior.
                pad = np.full(end - L, last_v, dtype=np.float32)
                window = np.concatenate([ep_vel[offset:L], pad])
            q_val = float(ep_q[offset]) if ep_q is not None else None
            w = decide(window, q_val)
            if w > 0.0:
                filtered_indices.append(cursor)
            cursor += 1

    valid = np.asarray(filtered_indices, dtype=np.int64)
    n_filtered = cursor
    keep_frac = (len(valid) / n_filtered) if n_filtered > 0 else 0.0
    elapsed = time.time() - t0
    logging.info(
        f"[rabc_precompute] {repo_id} H={H} mode={rabc.mode} "
        f"thr={rabc.threshold} fac={rabc.use_final_action_condition}: "
        f"{len(valid):,}/{n_filtered:,} kept ({keep_frac:.1%}) in {elapsed:.1f}s"
    )

    if spot_check and len(valid) > 0:
        _spot_check_valid_indices(repo_id, action_horizon, rabc, episodes, valid, lookahead_frames=lookahead_frames)

    np.save(cache_path, valid)
    logging.info(f"[rabc_precompute] cached → {cache_path}")
    return valid


def _spot_check_scizor_valid_indices(
    repo_id: str,
    action_horizon: int,
    rabc: _transforms.ComputeRABCWeights,
    episodes: tuple[int, ...] | None,
    valid: np.ndarray,
    scizor_sidecar_path: str,
    scizor_score_column: str,
    n_checks: int = 5,
) -> None:
    """Cross-validate the scizor precompute against a fresh LeRobot fetch +
    sidecar lookup. Cheap: each check is a single video frame decode plus a
    dict lookup."""
    delta_ts_actions = [t / lerobot_dataset.LeRobotDatasetMetadata(repo_id).fps for t in range(action_horizon)]
    ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={"actions": delta_ts_actions},
        tolerance_s=0.04,
        episodes=list(episodes) if episodes is not None else None,
    )
    scores_by_ep = _transforms._load_scizor_sidecar(scizor_sidecar_path, scizor_score_column)
    rng = np.random.default_rng(0)
    sampled = rng.choice(valid, size=min(n_checks, len(valid)), replace=False)
    for k in sampled:
        s = ds[int(k)]
        ep = int(np.asarray(s["episode_index"]).reshape(-1)[0])
        fi = int(np.asarray(s["frame_index"]).reshape(-1)[0])
        ep_scores = scores_by_ep[ep]
        anchor_score = float(ep_scores[min(fi, len(ep_scores) - 1)])
        w = rabc.decide_weight(np.asarray([anchor_score], dtype=np.float32))
        if w <= 0.0:
            raise AssertionError(
                f"[rabc_precompute][scizor] spot-check failed at idx {int(k)}: "
                f"ep={ep} frame={fi} score={anchor_score:.4f} ε_s={rabc.threshold} → w={w}"
            )
    logging.info(f"[rabc_precompute][scizor] spot-check OK on {len(sampled)} samples")


def _spot_check_valid_indices(
    repo_id: str,
    action_horizon: int,
    rabc: _transforms.ComputeRABCWeights,
    episodes: tuple[int, ...] | None,
    valid: np.ndarray,
    n_checks: int = 5,
    lookahead_frames: int = 0,
) -> None:
    """Cross-validate precomputed valid_indices against a fresh LeRobotDataset
    fetch. Asserts that ``decide_weight(dataset[k]['velocity'], q) > 0`` for
    sampled k. Cost: ~n_checks video decodes; pays for itself by catching
    flat-index mismatches before they silently corrupt training."""
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    H_total = action_horizon + lookahead_frames
    delta_ts_extra = [t / meta.fps for t in range(H_total)]
    delta_ts_actions = [t / meta.fps for t in range(action_horizon)]
    extra = ["repromo_signed_magnitude"]
    if rabc.mode != "velocity_only" and rabc.q_min is not None:
        extra.append("repromo_quality")
    delta_timestamps = {k: delta_ts_extra for k in extra}
    delta_timestamps["actions"] = delta_ts_actions
    ds = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.04,
        episodes=list(episodes) if episodes is not None else None,
    )
    rng = np.random.default_rng(0)
    sampled = rng.choice(valid, size=min(n_checks, len(valid)), replace=False)
    for k in sampled:
        s = ds[int(k)]
        vel_t = s.get("repromo_signed_magnitude")
        if vel_t is None:
            vel_t = s.get("rorm_velocity")
        vel = np.asarray(vel_t)
        q = None
        if rabc.mode != "velocity_only" and rabc.q_min is not None:
            qarr = s.get("repromo_quality")
            if qarr is None:
                qarr = s.get("rorm_q")
            if qarr is not None:
                q = float(np.asarray(qarr).reshape(-1)[0])
        w = rabc.decide_weight(vel, q)
        if w <= 0.0:
            raise AssertionError(
                f"[rabc_precompute] spot-check failed at filtered idx {int(k)}: "
                f"weight={w} (expected > 0). vel[:5]={vel[:5]}, q={q}"
            )
    logging.info(f"[rabc_precompute] spot-check OK on {len(sampled)} samples")


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    pipeline = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    # Only the per-getitem rejection mode wraps here; the subset mode does its
    # filtering downstream in create_torch_data_loader (so the existing
    # DistributedSampler / shuffle wiring composes cleanly with Subset).
    if (
        getattr(data_config, "reject_zero_weighted_samples", False)
        and getattr(data_config, "reject_zero_weighted_mode", "subset") == "rejection"
    ):
        return RejectionSamplingTransformedDataset(dataset, pipeline)
    return TransformedDataset(dataset, pipeline)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    # When RABC is disabled at the train-loop level (loss doesn't multiply by
    # sample_weights), skip the subset filter too — vanilla BC should see
    # every sample, not the rabc-gated subset that comes from the data
    # config's default rabc_threshold / use_final_action_condition.
    if not getattr(config, "rabc_enabled", False) and getattr(
        data_config, "reject_zero_weighted_samples", False
    ):
        import dataclasses as _dc
        data_config = _dc.replace(data_config, reject_zero_weighted_samples=False)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # RABC subset filtering: precompute the indices of samples whose weight > 0
    # offline (parquet-only, no video decode), then wrap in torch Subset so the
    # downstream sampler/DataLoader sees a dense, all-positive-weight dataset.
    # Composes with DistributedSampler / shuffle (the wrapped Subset just
    # renumbers 0..M-1).
    if (
        getattr(data_config, "reject_zero_weighted_samples", False)
        and getattr(data_config, "reject_zero_weighted_mode", "subset") == "subset"
    ):
        rabc = _find_rabc_transform(data_config)
        if rabc is not None and data_config.repo_id not in (None, "fake"):
            scizor_xf = _find_scizor_sidecar_transform(data_config)
            valid_indices = precompute_valid_indices(
                data_config.repo_id,
                action_horizon=action_horizon,
                rabc=rabc,
                episodes=data_config.episodes,
                lookahead_frames=getattr(data_config, "extra_horizon_lookahead_frames", 0),
                scizor_sidecar_path=scizor_xf.sidecar_path if scizor_xf is not None else None,
                scizor_score_column=scizor_xf.score_column if scizor_xf is not None else "scizor_score",
            )
            if len(valid_indices) == 0:
                raise RuntimeError(
                    f"RABC precompute kept zero samples for {data_config.repo_id} — "
                    "check rabc thresholds (every sample is being gated out)."
                )
            # Pass the numpy int64 array directly — torch.utils.data.Subset
            # accepts any sequence with __getitem__, and pickling a contiguous
            # numpy array to each spawn-worker is ~3× faster than a Python list
            # at multi-million-index scale.
            dataset = torch.utils.data.Subset(
                typing.cast(torch.utils.data.Dataset, dataset),
                indices=valid_indices,
            )
            logging.info(
                f"[rabc_subset] training over {len(dataset):,} valid samples"
            )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


# ---------------------------------------------------------------------------
# Online Reward Model data loader
# ---------------------------------------------------------------------------

@dataclass
class RMDatasetConfig:
    repo_id: str = "Qianzhong-Chen/tshirt_folding_10h_hlm_yam_white_0810"
    n_obs_steps: int = 8
    frame_gap: int = 30
    horizon: int = 8
    max_rewind_steps: int = 4
    image_names: List[str] = field(default_factory=lambda: ["top_camera-images-rgb"])
    dense_annotation: bool = False
    video_eval: bool = True


@dataclass
class XdofLerobotDatasetConfig:
    repo_id: str = "Qianzhong-Chen/tshirt_folding_10h_hlm_yam_white_0810"
    n_obs_steps: int = 1
    horizon: int = 26


def create_rm_dataset(data_config: _config.DataConfig, action_horizon: int) -> "Dataset":
    from openpi.training.rm_datasets import HybirdLeRobotDataset

    rm_cfg = RMDatasetConfig(repo_id=data_config.repo_id)
    xdof_cfg = XdofLerobotDatasetConfig(repo_id=data_config.repo_id, horizon=action_horizon + 1)

    rm = getattr(data_config, "reward_model", None)
    if rm is not None:
        rm_cfg.n_obs_steps = getattr(rm, "n_obs_steps", rm_cfg.n_obs_steps)
        rm_cfg.frame_gap = getattr(rm, "frame_gap", rm_cfg.frame_gap)
        rm_cfg.horizon = getattr(rm, "horizon", rm_cfg.horizon)
        rm_cfg.max_rewind_steps = getattr(rm, "max_rewind_steps", rm_cfg.max_rewind_steps)
        rm_cfg.image_names = list(getattr(rm, "camera_names", rm_cfg.image_names))
        rm_cfg.dense_annotation = getattr(rm, "dense_annotation", rm_cfg.dense_annotation)

    return HybirdLeRobotDataset(
        frame_gap_dataset_kwargs=asdict(rm_cfg),
        xdof_dataset_kwargs=asdict(xdof_cfg),
    )


class RMTorchDataLoader:
    """Data loader for the online reward model. Yields raw batch dicts (not Observations)."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield batch


class RMDataLoaderImpl:
    """Wrapper that yields raw RM batch dicts."""

    def __init__(self, data_config: _config.DataConfig, data_loader: RMTorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield batch


def create_rm_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> RMDataLoaderImpl:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_rm_dataset(data_config, config.model.action_horizon)

    data_loader = RMTorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return RMDataLoaderImpl(data_config, data_loader)
