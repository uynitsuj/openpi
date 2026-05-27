from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import pathlib
import re
import threading
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class ComputeRABCWeights(DataTransformFn):
    """Compute per-sample RABC weights from rorm_velocity (and optionally rorm_q).

    Velocity weight: v_weight = clip(mean(rorm_velocity over chunk), clip_min, clip_max)

    When q_min/q_max are provided, also normalizes rorm_q:
      q_norm = clip((rorm_q - q_min) / (q_max - q_min), 0, 1)

    Combined per ``mode``:
      "velocity_only"  (default): w = v_weight
      "multiplicative":           w = v_weight * q_norm
      "additive":                 w = 0.5 * (v_weight + q_norm)
      "q_threshold":              episode-Q-driven adaptive velocity threshold
                                  (see q_threshold_* params)

    Legacy velocity-only modes (threshold / use_final_action_condition) still
    work when mode="velocity_only".

    When ``threshold`` is set and mode="velocity_only", samples with integrated
    weight below the threshold are zeroed out instead of clipped to clip_min.

    When ``use_final_action_condition`` is True, skips integration and gates
    purely on the final-frame velocity:
      keep iff vel[-1] > ``threshold``
    Kept samples get weight = clip(vel[-1], None, clip_max); rejected get 0.
    ``threshold`` must be set in this mode. (Previous versions also kept
    samples whose final velocity was small but dv/dt > 0 — that ``cond_accel``
    branch was a heuristic that bypassed the threshold and has been removed
    so the gate matches its name.)

    ``sarm_progress_delta`` mode: reads absolute progress (``sarm_progress_key``)
    instead of velocity, computes ``reward = progress[-1] - progress[0]`` over
    the action horizon, then derives weight via the SARM-paper formula:
        w̃ = clip((reward - (μ - 2σ)) / (4σ + ε), 0, 1)
        w = 1 if reward > κ else (w̃ if reward ≥ 0 else 0)
    μ/σ come from ``sarm_reward_mu``/``sarm_reward_sigma`` (pre-computed from
    the dataset); κ is ``sarm_kappa``. Velocity/q params are ignored. Subset
    precompute is not supported in this mode (decide_weight reads velocity),
    so LeRobotRABCDataConfig forces ``rabc_reject_zero_weighted_mode='rejection'``
    when ``rabc_mode='sarm_progress_delta'``.

    q_threshold mode: compute q_norm via the same min-max normalization the
    multiplicative/additive modes use:
      q_norm = clip((rorm_q - q_min) / (q_max - q_min), 0, 1)
    Then map q_norm to a per-episode velocity threshold via ``q_threshold_shape``:
      "linear":  thr = q_threshold_low + (q_threshold_high - q_threshold_low) * q_norm
      "sigmoid": thr = q_threshold_high
                       + (q_threshold_low - q_threshold_high)
                       * (1 - sigmoid((q_norm - q_threshold_center) * q_threshold_steepness))
    With defaults q_threshold_low=1.0, q_threshold_high=0.0: best episode →
    threshold 0 (any positive velocity passes); worst episode → threshold 1.
    Kept chunks get weight = clip(mean(vel), None, clip_max); rejected get 0.

    Previously this mode required a precomputed rank-percentile lookup
    (InjectEpisodeQNorm) keyed on episode_index; that path is gone in favor
    of pure min-max so the threshold computation is self-contained per sample
    and behaves consistently with the multiplicative/additive modes.

    Expects `rorm_velocity` in the data dict as shape (action_horizon,).
    Produces `sample_weights` as a scalar float.
    """

    clip_min: float = 0.0
    clip_max: float = 1.0
    threshold: float | None = None
    use_final_action_condition: bool = False
    mode: str = "velocity_only"
    q_min: float | None = None
    q_max: float | None = None
    # q_threshold mode params.
    q_threshold_low: float = 1.0
    q_threshold_high: float = 0.0
    q_threshold_shape: str = "linear"  # "linear" | "sigmoid"
    q_threshold_center: float = 0.5
    q_threshold_steepness: float = 10.0
    # Window-aggregator: how to collapse the per-frame velocity vector into a
    # single scalar weight before clipping. "mean" (default) is the historical
    # behavior. "min" takes the lowest velocity in the chunk — penalizes any
    # frame in the window dipping into anti-progress, which is a stricter
    # criterion than averaging (one bad frame zeros out the weight after
    # clip_min). "max" takes the highest velocity — rewards windows whose
    # best frame is positive, useful when chunks straddle action boundaries.
    # "mean_lookahead": mean of vel[action_horizon:] only — the lookahead
    # portion of the velocity window, ignoring the action-chunk part. Requires
    # the data loader to fetch action_horizon + lookahead frames (set
    # ``extra_horizon_lookahead_frames`` on the DataConfig), and requires
    # ``action_horizon`` to be set on this transform so we know where to slice.
    velocity_aggregator: str = "mean"
    # Action chunk size, used by aggregators that need to know where the
    # action-chunk window ends and the lookahead begins ("mean_lookahead").
    # Set to model_config.action_horizon at config-build time.
    action_horizon: int = 0
    # Non-linear power applied to the weight AFTER gate decision but BEFORE
    # the final clip(weight, None, clip_max). With weight_power=2 + clip_max=1
    # this gives min(weight^2, 1) — quadratically suppresses medium-magnitude
    # weights while capping bursts, which empirically penalizes long episodes
    # (since their per-frame vel distribution is concentrated at ~0.5) by an
    # extra ~12% relative to linear weighting. weight_power=1 (default) is the
    # historical linear behavior.
    weight_power: float = 1.0
    # Multiplicative scale applied to the velocity vector before aggregation/
    # clipping. Useful when raw velocities are on a very different scale than
    # the [0, 1] range expected by clipping/thresholding (e.g. SARM per-frame
    # progress deltas are ~0.0005 per frame).
    velocity_scale: float = 1.0
    # sarm_progress_delta mode params. Reads absolute progress over the action
    # horizon and computes reward = progress[-1] - progress[0]. Weights are
    # derived via the SARM paper formula:
    #   w̃ = clip((r - (μ - 2σ)) / (4σ + ε), 0, 1)
    #   w = 1 if r > κ, else w̃
    # μ and σ are pre-computed from the dataset (or supplied explicitly).
    sarm_reward_mu: float = 0.0
    sarm_reward_sigma: float = 1.0
    sarm_kappa: float = 0.01
    # Column name for absolute progress values.
    sarm_progress_key: str = "sarm_dense_progress"
    # ── scizor_anchor mode params ──
    # Paper-faithful SCIZOR gating (Zhang et al. 2026, §3.2). Reads a scalar
    # per-sample anchor score from data["scizor_score"] (injected upstream by
    # LoadScizorSidecar from the SCIZOR sidecar parquet) and decides keep/drop
    # at the chunk's anchor frame only — no aggregation over the action
    # horizon. ``threshold`` is reused as ε_s (paper default 0.58).
    #   scizor_weight_mode='binary':     w = 1 if score <= ε_s else 0
    #   scizor_weight_mode='continuous': w = clip(1 - score, clip_min, clip_max)
    scizor_weight_mode: str = "binary"
    scizor_score_key: str = "scizor_score"

    def _aggregate_velocity(self, vel: np.ndarray) -> float:
        """Collapse the per-frame velocity vector to a scalar window weight."""
        n = max(len(vel), 1)
        if self.velocity_aggregator == "mean":
            return float(np.sum(vel) / n)
        if self.velocity_aggregator == "min":
            return float(np.min(vel))
        if self.velocity_aggregator == "max":
            return float(np.max(vel))
        if self.velocity_aggregator == "mean_lookahead":
            if self.action_horizon <= 0:
                raise ValueError(
                    "velocity_aggregator='mean_lookahead' requires action_horizon "
                    "to be set on ComputeRABCWeights (the action chunk length)."
                )
            tail = vel[self.action_horizon:]
            if len(tail) == 0:
                raise ValueError(
                    f"velocity_aggregator='mean_lookahead' but fetched vel has "
                    f"{len(vel)} frames <= action_horizon ({self.action_horizon}); "
                    "set ``extra_horizon_lookahead_frames`` on the DataConfig."
                )
            return float(np.mean(tail))
        raise ValueError(
            f"Unknown velocity_aggregator {self.velocity_aggregator!r}. "
            "Expected 'mean' | 'min' | 'max' | 'mean_lookahead'."
        )

    def _threshold_from_q_norm(self, q_norm: float) -> float:
        if self.q_threshold_shape == "linear":
            return float(self.q_threshold_low + (self.q_threshold_high - self.q_threshold_low) * q_norm)
        if self.q_threshold_shape == "sigmoid":
            x = (q_norm - self.q_threshold_center) * self.q_threshold_steepness
            sig = 1.0 / (1.0 + float(np.exp(-x)))
            return float(self.q_threshold_high + (self.q_threshold_low - self.q_threshold_high) * (1.0 - sig))
        raise ValueError(
            f"Unknown q_threshold_shape {self.q_threshold_shape!r}. Expected 'linear' or 'sigmoid'."
        )

    def decide_weight(self, vel: np.ndarray, q: float | None = None) -> float:
        """Pure gate logic: given a velocity window and (optional) per-frame
        quality scalar, return the sample weight. No data dict, no key
        migration — used by both ``__call__`` and the precompute path that
        builds ``valid_indices`` for SubsetRandomSampler / Subset.

        ``velocity_scale`` is applied here (not in ``__call__``) so the
        precompute filter and the runtime weight agree when the caller sets
        ``rabc_velocity_scale != 1.0``.
        """
        vel = np.asarray(vel, dtype=np.float32)
        if self.mode == "scizor_anchor":
            # Anchor-frame gate. The caller passes a length-1 array (or a
            # scalar broadcast to one); we read only vel[0]. velocity_scale
            # is intentionally NOT applied — SCIZOR scores are already in
            # [0, 1] and have a paper-fixed threshold (ε_s).
            if self.threshold is None:
                raise ValueError(
                    "mode='scizor_anchor' requires threshold (ε_s) to be set."
                )
            score = float(vel.reshape(-1)[0])
            if self.scizor_weight_mode == "binary":
                w = 1.0 if score <= self.threshold else 0.0
            elif self.scizor_weight_mode == "continuous":
                w = float(np.clip(1.0 - score, self.clip_min, self.clip_max))
            else:
                raise ValueError(
                    f"Unknown scizor_weight_mode {self.scizor_weight_mode!r}. "
                    "Expected 'binary' or 'continuous'."
                )
            return self._apply_power(w)
        if self.velocity_scale != 1.0:
            vel = vel * self.velocity_scale
        if self.mode == "q_threshold":
            if self.q_min is None or self.q_max is None:
                raise ValueError(
                    "mode='q_threshold' requires q_min and q_max for min-max "
                    "normalization. Set them on the config (typically via "
                    "rabc_q_min / rabc_q_max or autoload from rabc_stats.json)."
                )
            if q is None:
                raise ValueError(
                    "mode='q_threshold' requires a per-frame q value, but got None."
                )
            denom = max(self.q_max - self.q_min, 1e-8)
            q_norm = float(np.clip((float(q) - self.q_min) / denom, 0.0, 1.0))
            thr = self._threshold_from_q_norm(q_norm)
            if self.use_final_action_condition:
                final_vel = float(vel[-1])
                w = float(np.clip(final_vel, None, self.clip_max)) if final_vel > thr else 0.0
            else:
                agg_vel = self._aggregate_velocity(vel)
                w = 0.0 if agg_vel < thr else float(np.clip(agg_vel, None, self.clip_max))
            return self._apply_power(w)

        if self.use_final_action_condition:
            final_vel = float(vel[-1])
            if self.threshold is not None:
                weight = float(np.clip(final_vel, None, self.clip_max)) if final_vel > self.threshold else 0.0
            else:
                # No threshold: pass the final-frame velocity through as the
                # weight, clipped to [clip_min, clip_max]. With defaults
                # (clip_min=0, clip_max=inf) negative-motion samples get 0 and
                # are filtered by reject_zero_weighted_samples downstream.
                weight = float(np.clip(final_vel, self.clip_min, self.clip_max))
        else:
            weight = self._aggregate_velocity(vel)
            if self.threshold is not None:
                weight = 0.0 if weight < self.threshold else float(np.clip(weight, None, self.clip_max))
            else:
                weight = float(np.clip(weight, self.clip_min, self.clip_max))

        can_use_q = (
            self.mode != "velocity_only"
            and self.q_min is not None
            and self.q_max is not None
            and q is not None
        )
        if can_use_q:
            denom = max(self.q_max - self.q_min, 1e-8)
            q_norm = float(np.clip((float(q) - self.q_min) / denom, 0.0, 1.0))
            if self.mode == "multiplicative":
                weight = weight * q_norm
            elif self.mode == "additive":
                weight = 0.5 * (weight + q_norm)
            else:
                raise ValueError(
                    f"Unknown RABC mode {self.mode!r}. Expected 'velocity_only', "
                    "'multiplicative', 'additive', or 'q_threshold'."
                )
        return self._apply_power(float(weight))

    def _apply_power(self, weight: float) -> float:
        """Apply weight_power: w → min(max(w, 0)^p, clip_max). Default p=1 is
        a no-op modulo the cap re-application."""
        if self.weight_power == 1.0:
            return weight
        if weight <= 0.0:
            return 0.0
        return float(min(weight ** self.weight_power, self.clip_max))

    def __call__(self, data: DataDict) -> DataDict:
        # ── SCIZOR anchor-frame mode ─────────────────────────────────────
        # Paper-faithful filter: read the per-anchor scalar score injected
        # by LoadScizorSidecar and gate on ε_s. No aggregation, no velocity.
        if self.mode == "scizor_anchor":
            score_key = self.scizor_score_key
            if score_key not in data:
                # No sidecar lookup happened — leave sample_weights unset so
                # downstream code falls back to vanilla BC (matches the
                # behavior of velocity_only when no velocity column exists).
                return data
            score_val = float(np.asarray(data[score_key], dtype=np.float32).reshape(-1)[0])
            weight = self.decide_weight(np.asarray([score_val], dtype=np.float32))
            data = {**data, "sample_weights": np.float32(weight)}
            data.pop(score_key, None)
            return data

        # ── SARM progress-delta mode ──────────────────────────────────────
        # Computes reward = progress[-1] - progress[0] from absolute progress
        # predictions over the action horizon, then applies SARM-style soft
        # weighting:
        #   w̃ = clip((r - (μ-2σ)) / (4σ+ε), 0, 1)
        #   w = 1 if r > κ, w̃ if 0 ≤ r ≤ κ, 0 if r < 0
        if self.mode == "sarm_progress_delta":
            prog_key = self.sarm_progress_key
            if prog_key not in data:
                return data
            prog = np.asarray(data[prog_key], dtype=np.float32).ravel()
            if len(prog) < 2:
                data = {**data, "sample_weights": np.float32(0.0)}
                data.pop(prog_key, None)
                return data
            reward = float(prog[-1] - prog[0])
            # SARM paper: μ ← max(μ, 0) to prevent negative-mean datasets
            # from shifting the normalization window too far left.
            mu = max(self.sarm_reward_mu, 0.0)
            sigma = self.sarm_reward_sigma
            eps = 1e-6
            lo = mu - 2.0 * sigma
            denom = max(4.0 * sigma, eps)
            w_soft = float(np.clip((reward - lo) / denom, 0.0, 1.0))
            if reward > self.sarm_kappa:
                weight = 1.0
            elif reward >= 0.0:
                weight = w_soft
            else:
                weight = 0.0
            data = {**data, "sample_weights": np.float32(weight)}
            data.pop(prog_key, None)
            for k in ("sarm_dense_signed_magnitude", "sarm_dense_quality",
                       "sarm_sparse_progress", "sarm_sparse_signed_magnitude", "sarm_sparse_quality"):
                data.pop(k, None)
            return data

        # ── Velocity-based modes ──────────────────────────────────────────
        # Schema migration: read repromo_signed_magnitude (canonical, post
        # Repromo rename) or fall back to rorm_velocity (legacy, pre-rename).
        # Same for repromo_quality vs rorm_q on the quality side.
        vel_key = next(
            (k for k in ("repromo_signed_magnitude", "rorm_velocity", "sarm_dense_signed_magnitude") if k in data),
            None,
        )
        if vel_key is None:
            return data
        vel = np.asarray(data[vel_key], dtype=np.float32)
        if len(vel) == 0:
            return data
        # velocity_scale is applied inside decide_weight() so __call__ and the
        # subset-precompute path stay in sync; don't scale here.

        q_key = next(
            (k for k in ("repromo_quality", "rorm_q", "sarm_dense_quality") if k in data),
            None,
        )
        q_val: float | None = None
        if q_key is not None:
            q_val = float(np.asarray(data[q_key], dtype=np.float32).reshape(-1)[0])

        weight = self.decide_weight(vel, q_val)

        data = {**data, "sample_weights": np.float32(weight)}
        data.pop(vel_key, None)
        if q_key is not None:
            data.pop(q_key, None)
        data.pop("episode_q_norm", None)
        return data


# Module-level cache for sidecar parquet contents. Keyed on the sidecar path
# (resolved) + (mtime, size) so swapping sidecar files between runs in the
# same process invalidates correctly. Holds a (per_episode dict, score_column)
# pair: { episode_index → np.ndarray[length_of_episode, float32] }.
_SCIZOR_SIDECAR_CACHE: dict[
    tuple[str, float, int, str], dict[int, np.ndarray]
] = {}
_SCIZOR_SIDECAR_LOCK = threading.Lock()


def _load_scizor_sidecar(
    sidecar_path: str, score_column: str = "scizor_score"
) -> dict[int, np.ndarray]:
    """Read a SCIZOR sidecar parquet and return per-episode score arrays.

    The sidecar (output of
    ``SCIZOR_Baseline/curation/video_encoding/score_lerobot.py``) has columns
    ``episode_index, frame_index, scizor_score[, scizor_score_local,
    scizor_score_traj_mean]``. We materialise ``{episode_index:
    np.ndarray[length]}`` so that per-sample lookup is O(1) and memory is
    proportional to the dataset (≈4 bytes / frame).

    Cached at module level; loading is process-global and thread-safe.
    """
    import pyarrow.parquet as _pq  # local import: optional dep

    # Remote sidecars (s3://, gs://) are fetched once to a local cache. The
    # subset precompute in data_loader.precompute_valid_indices reads the
    # sidecar in the main process before the torch DataLoader spawns workers,
    # so this download happens exactly once per run; maybe_download is a
    # passthrough (no-op) for already-local paths.
    if "://" in sidecar_path:
        from openpi.shared import download as _download
        sidecar_path = str(_download.maybe_download(sidecar_path))

    resolved = str(pathlib.Path(sidecar_path).resolve())
    stat = pathlib.Path(resolved).stat()
    cache_key = (resolved, stat.st_mtime, stat.st_size, score_column)
    with _SCIZOR_SIDECAR_LOCK:
        cached = _SCIZOR_SIDECAR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        table = _pq.read_table(
            resolved, columns=["episode_index", "frame_index", score_column]
        )
        ep = table["episode_index"].to_numpy().astype(np.int64)
        fr = table["frame_index"].to_numpy().astype(np.int64)
        sc = table[score_column].to_numpy().astype(np.float32)

        out: dict[int, np.ndarray] = {}
        # Build per-episode dense arrays in one pass. Assumes each
        # (episode_index, frame_index) appears at most once (true of the
        # SCIZOR sidecar). Missing trailing frames (the classifier window
        # cannot reach frames within goal_time*fps of the episode end) are
        # padded with the last available score so anchor lookups near the
        # end of an episode never KeyError.
        order = np.lexsort((fr, ep))
        ep_s, fr_s, sc_s = ep[order], fr[order], sc[order]
        # Group bounds via diff on episode index.
        boundaries = np.flatnonzero(np.diff(ep_s)) + 1
        starts = np.concatenate([[0], boundaries])
        stops = np.concatenate([boundaries, [len(ep_s)]])
        for s, e in zip(starts, stops):
            ep_id = int(ep_s[s])
            length = int(fr_s[e - 1]) + 1
            arr = np.empty(length, dtype=np.float32)
            arr[fr_s[s:e]] = sc_s[s:e]
            # Fill any gaps (rare but defend) with the last seen score.
            # Simple forward-fill via cumulative-max-of-index trick:
            missing = np.setdiff1d(np.arange(length), fr_s[s:e], assume_unique=True)
            if len(missing) > 0:
                # Forward-fill from previous valid index; if at the start,
                # use the first valid score.
                valid_mask = np.zeros(length, dtype=bool)
                valid_mask[fr_s[s:e]] = True
                last_valid = -1
                first_valid_val = float(sc_s[s])
                for i in range(length):
                    if valid_mask[i]:
                        last_valid = i
                    else:
                        arr[i] = arr[last_valid] if last_valid >= 0 else first_valid_val
            out[ep_id] = arr
        _SCIZOR_SIDECAR_CACHE[cache_key] = out
        logging.info(
            "[scizor_sidecar] loaded %s (%d episodes, %d frames) col=%s",
            resolved, len(out), sum(len(v) for v in out.values()), score_column,
        )
        return out


@dataclasses.dataclass(frozen=True)
class LoadScizorSidecar(DataTransformFn):
    """Inject SCIZOR per-frame suboptimality scores into each training sample.

    Reads a sidecar parquet produced by SCIZOR's ``score_lerobot.py`` and
    writes ``data[score_key]`` as a scalar float32 — the score at the
    sample's *anchor frame* (the chunk's first frame). Designed to be
    composed with ``ComputeRABCWeights(mode='scizor_anchor')``, which
    consumes the same key.

    The transform does NOT modify the underlying LeRobot dataset — scores
    live in a separate parquet and are joined by (episode_index,
    frame_index) at sample time. This makes head-to-head comparison of
    different SCIZOR checkpoints a pure config swap.

    Anchor-frame semantics match SCIZOR's reference octo pipeline
    (``octo/octo/data/dataset.py:612``): the per-frame mask attached to
    each chunk is read at the chunk's start frame `t`, not aggregated over
    the action horizon. See the openpi RA-BC scizor docs for the full
    methodology.
    """

    sidecar_path: str
    score_key: str = "scizor_score"
    score_column: str = "scizor_score"

    def __call__(self, data: DataDict) -> DataDict:
        ep_idx = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        frame_idx = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        scores_by_ep = _load_scizor_sidecar(self.sidecar_path, self.score_column)
        ep_scores = scores_by_ep.get(ep_idx)
        if ep_scores is None:
            # Defensive: episode missing from sidecar → zero-weight sample.
            # decide_weight will gate it out cleanly downstream.
            data = {**data, self.score_key: np.float32(np.inf)}
            return data
        # Clamp frame_idx to last valid score (handles the trailing-window
        # tail where the classifier could not produce a real score).
        fi = min(frame_idx, len(ep_scores) - 1)
        data = {**data, self.score_key: np.float32(ep_scores[fi])}
        return data


@dataclasses.dataclass(frozen=True)
class InjectEpisodeQNorm(DataTransformFn):
    """Deprecated. q_threshold mode now uses min-max normalization on the
    per-frame `repromo_quality` directly, the same as multiplicative/additive
    modes — no precomputed rank lookup is needed. This class is kept only so
    pickled configs from prior runs still import; remove once nothing in flight
    depends on it.
    """

    episode_q_norm_pairs: tuple[tuple[int, float], ...] = ()
    default: float = 0.0

    def __post_init__(self) -> None:
        lookup = {int(k): float(v) for k, v in self.episode_q_norm_pairs}
        object.__setattr__(self, "_lookup", lookup)

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data:
            return data
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        q = self._lookup.get(ep, self.default)
        return {**data, "episode_q_norm": np.float32(q)}


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
