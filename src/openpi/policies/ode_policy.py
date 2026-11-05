"""Data transforms and server-side state for the ODE action head (openpi.models.pi0_ode).

Row convention (paper, docs/PORTING.md and ode_policy/real/README.md): a sample anchored at frame t
integrates from a_0 = action[t-1] (the reference the controller holds when frame t is observed;
the measured state at the first frame of an episode) and is supervised by action[t .. t+H-1].
At deployment a_0 is the last reference the robot actually executed, which only the caller
(here: the policy server) knows -> ODECarry keeps it per world.
"""

from collections.abc import Callable
import dataclasses
import logging

import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class ODESplitPrevAction(_transforms.DataTransformFn):
    """Raw (unnormalised) units, before the robot input transform (YamInputs).

    Training: the loader fetched H + 1 action rows (offsets -1 .. H-1, DataConfig.ode_head). Row 0 becomes
    `ode_start` (the measured state if it lies before the episode start), rows 1..H the target chunk, and
    `action_valid_mask` marks the rows inside the episode.
    Inference (no "actions"): `ode_start` = the measured state; ODECarry replaces it by the carried reference.
    """

    action_horizon: int

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        state = np.asarray(data["state"], dtype=np.float32)
        if "actions" not in data:
            data["ode_start"] = state.copy()
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[0] != self.action_horizon + 1:
            raise ValueError(
                f"ODE data expects action_horizon + 1 = {self.action_horizon + 1} action rows (offsets -1..H-1), "
                f"got {actions.shape[0]}; is DataConfig.ode_head set?"
            )
        is_pad = data.pop("actions_is_pad", None)
        is_pad = np.zeros(actions.shape[0], bool) if is_pad is None else np.asarray(is_pad, dtype=bool)
        if is_pad.shape != (actions.shape[0],):
            raise ValueError(f"actions_is_pad has shape {is_pad.shape}, expected ({actions.shape[0]},)")
        data["ode_start"] = state.copy() if is_pad[0] else actions[0].copy()
        data["actions"] = actions[1:]
        data["action_valid_mask"] = ~is_pad[1:]
        return data


def _unit_slope_and_normalize(stats: _transforms.NormStats, use_quantiles: bool):
    """Return (normalize(x), d normalized / d raw) matching openpi's Normalize exactly."""
    if use_quantiles:
        q01, q99 = np.asarray(stats.q01, np.float32), np.asarray(stats.q99, np.float32)
        span = q99 - q01
        slope = np.where(span > 0, 2.0 / (span + 1e-6), 0.0).astype(np.float32)

        def normalize(x):
            n = x.shape[-1]
            return (x - q01[:n]) / (q99[:n] - q01[:n] + 1e-6) * 2.0 - 1.0

    else:
        mean, std = np.asarray(stats.mean, np.float32), np.asarray(stats.std, np.float32)
        slope = np.where(std > 0, 1.0 / (std + 1e-6), 0.0).astype(np.float32)

        def normalize(x):
            n = x.shape[-1]
            return (x - mean[:n]) / (std[:n] + 1e-6)

    return normalize, slope


@dataclasses.dataclass(frozen=True)
class ODENormalizeStart(_transforms.DataTransformFn):
    """Right after Normalize: express `ode_start` in the ACTION normalisation (it is a reference, not a state,
    even when it was filled from the measured state) and add `ode_unit_scale` (d normalised / d raw)."""

    norm_stats: dict
    use_quantiles: bool = False

    def __call__(self, data: dict) -> dict:
        if self.norm_stats is None or "actions" not in self.norm_stats:
            raise ValueError("ODE head needs action norm stats")
        normalize, slope = _unit_slope_and_normalize(self.norm_stats["actions"], self.use_quantiles)
        data = dict(data)
        if "ode_start" in data:
            start = np.asarray(data["ode_start"], dtype=np.float32)
            data["ode_start"] = normalize(start).astype(np.float32)
            data["ode_unit_scale"] = _transforms.pad_to_dim(slope[: start.shape[-1]], start.shape[-1])
        return data


class ODECarry:
    """Per-world carried reference for a batched, stateless caller (server side).

    Wraps the jitted `sample_actions(rng, observation, **kw)` of a Pi0ODE policy. Row b of every call is
    world b (the abc-rabc batched runner sends all worlds, in a fixed order, on every replan). For each row
    the chunk starts from the last EXECUTED reference of that world -- row (executed_steps - 1) of the chunk
    returned on the previous call -- instead of the measured state, except when the world starts an episode:
      * first call, or the batch size changed           -> all rows start from the measured state;
      * reset() / reset(rows) (Policy.reset, or obs["reset"] in Policy.infer)       -> those rows;
      * a row's measured arm joints are > reset_jump_rad away from its carried reference (a new episode
        whose initial pose differs; the batched runner never needs this) -> that row.
    Everything is in normalised action units; `action_slope` (d normalised / d raw) converts the jump test to rad.
    """

    def __init__(
        self,
        sample_fn: Callable,
        *,
        executed_steps: int,
        action_slope: np.ndarray,
        jump_dims: tuple[int, ...],
        reset_jump_rad: float,
    ):
        self._fn = sample_fn
        self.executed_steps = int(executed_steps)
        self._slope = np.asarray(action_slope, np.float32)
        self._jump_dims = np.asarray(jump_dims, np.int64)
        self.reset_jump_rad = float(reset_jump_rad)
        self._ref: np.ndarray | None = None  # [B, action_dim], normalised
        self._valid: np.ndarray | None = None  # [B] bool
        self.stats = {"calls": 0, "carried_rows": 0, "fresh_rows": 0, "jump_resets": 0}

    def reset(self, rows=None) -> None:
        if rows is None or self._valid is None:
            self._ref, self._valid = None, None
        else:
            self._valid[np.asarray(rows)] = False

    def start_rows(self, measured_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(start [B, A], carried [B] bool) for a batch whose measured-state start is `measured_start`."""
        measured_start = np.asarray(measured_start, np.float32)
        batch = measured_start.shape[0]
        if self._ref is None or self._ref.shape != measured_start.shape:
            if self._ref is not None:
                logger.info("ODECarry: batch shape %s -> %s, resetting all rows", self._ref.shape, measured_start.shape)
            self._ref, self._valid = None, None
            return measured_start, np.zeros(batch, bool)
        carried = self._valid.copy()
        slope = self._slope[self._jump_dims]
        gap_rad = np.abs(measured_start[:, self._jump_dims] - self._ref[:, self._jump_dims]) / np.where(
            slope > 0, slope, np.inf
        )
        jump = carried & (gap_rad.max(axis=-1) > self.reset_jump_rad)
        if jump.any():
            logger.info(
                "ODECarry: rows %s jumped > %.2f rad from their reference; restarting them",
                np.flatnonzero(jump).tolist(),
                self.reset_jump_rad,
            )
            self.stats["jump_resets"] += int(jump.sum())
        carried &= ~jump
        return np.where(carried[:, None], self._ref, measured_start), carried

    def __call__(self, rng, observation, **kwargs):
        start, carried = self.start_rows(np.asarray(observation.ode_start))
        observation = dataclasses.replace(observation, ode_start=jnp.asarray(start, dtype=jnp.float32))
        actions = self._fn(rng, observation, **kwargs)
        self._ref = np.array(actions[:, self.executed_steps - 1, :], dtype=np.float32)
        self._valid = np.ones(self._ref.shape[0], bool)
        self.stats["calls"] += 1
        self.stats["carried_rows"] += int(carried.sum())
        self.stats["fresh_rows"] += int((~carried).sum())
        return actions


def yam_arm_dims() -> tuple[int, ...]:
    """The 12 arm joints of the 14-D YAM vector [L j1..j6, L grip, R j1..j6, R grip]."""
    return (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
