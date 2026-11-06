import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import ode_policy
from openpi.policies import yam_policy
from openpi.shared import normalize as _normalize


def _stats(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.r_[rng.normal(size=14), np.zeros(dim - 14)].astype(np.float32)
    std = np.r_[rng.uniform(0.2, 0.6, size=14), np.zeros(dim - 14)].astype(np.float32)
    return _normalize.NormStats(mean=mean, std=std, q01=mean - 2 * std, q99=mean + 2 * std)


NORM = {"state": _stats(seed=0), "actions": _stats(seed=1)}


# ----------------------------------------------------------------------------------------------- data transforms
def test_split_prev_action_mid_episode_start_and_end():
    split = ode_policy.ODESplitPrevAction(action_horizon=4)
    state = np.arange(14, dtype=np.float32)
    actions = np.arange(5 * 14, dtype=np.float32).reshape(5, 14) + 100
    mid = split({"state": state, "actions": actions, "actions_is_pad": np.zeros(5, bool)})
    np.testing.assert_array_equal(mid["ode_start"], actions[0])  # a0 = action[t-1]
    np.testing.assert_array_equal(mid["actions"], actions[1:])
    np.testing.assert_array_equal(mid["action_valid_mask"], [True] * 4)
    assert "actions_is_pad" not in mid
    first = split({"state": state, "actions": actions, "actions_is_pad": np.array([True] + [False] * 4)})
    np.testing.assert_array_equal(first["ode_start"], state)  # episode start: measured state
    end = split({"state": state, "actions": actions, "actions_is_pad": np.array([False, False, True, True, True])})
    np.testing.assert_array_equal(end["action_valid_mask"], [True, False, False, False])
    infer = split({"state": state})
    np.testing.assert_array_equal(infer["ode_start"], state)
    with pytest.raises(ValueError):
        split({"state": state, "actions": actions[:4]})


def test_yam_inputs_pass_ode_fields_and_normalize_start_in_action_units():
    data = ode_policy.ODESplitPrevAction(action_horizon=4)(
        {
            "state": np.full(14, 0.3, np.float32),
            "actions": np.full((5, 14), 0.7, np.float32),
            "actions_is_pad": np.zeros(5, bool),
            "top_camera-images-rgb": np.zeros((3, 8, 8), np.uint8),
        }
    )
    data = yam_policy.YamInputs(action_dim=32)(data)
    assert data["ode_start"].shape == (32,) and data["action_valid_mask"].shape == (4,)
    data = _transforms.Normalize(NORM)(data)
    data = ode_policy.ODENormalizeStart(NORM)(data)
    # the start (a reference) is normalised exactly like the actions, not like the state
    np.testing.assert_allclose(data["ode_start"][:14], data["actions"][0, :14], rtol=1e-6)
    np.testing.assert_allclose(data["ode_unit_scale"][:14], 1 / (NORM["actions"].std[:14] + 1e-6), rtol=1e-6)
    assert np.all(data["ode_unit_scale"][14:] == 0) and np.all(data["ode_start"][14:] == 0)
    q = ode_policy.ODENormalizeStart(NORM, use_quantiles=True)(
        {"ode_start": np.full(32, 0.7, np.float32) * (np.arange(32) < 14)}
    )
    ref = _transforms.Normalize(NORM, use_quantiles=True)(
        {"actions": np.full((1, 32), 0.7, np.float32) * (np.arange(32) < 14)}
    )
    np.testing.assert_allclose(q["ode_start"][:14], ref["actions"][0, :14], rtol=1e-5)


# ----------------------------------------------------------------------------------------------- per-world carry
class _FakeSample:
    """Returns a chunk whose row k is start + (k + 1) * 0.01 (normalised units); records the starts it saw."""

    def __init__(self, horizon=30, dim=32):
        self.h, self.dim, self.starts = horizon, dim, []

    def __call__(self, rng, observation, **kw):
        start = np.asarray(observation.ode_start)
        self.starts.append(start.copy())
        k = np.arange(1, self.h + 1, dtype=np.float32)[None, :, None]
        return jnp.asarray(start[:, None, :] + 0.01 * k)


def _obs(start):
    start = jnp.asarray(start, jnp.float32)
    b = start.shape[0]
    return _model.Observation(
        images={"base_0_rgb": jnp.zeros((b, 2, 2, 3))},
        image_masks={"base_0_rgb": jnp.ones((b,), bool)},
        state=start,
        ode_start=start,
    )


def _carry(fake, executed=30, jump=0.5):
    slope = np.r_[1 / (NORM["actions"].std[:14] + 1e-6), np.zeros(18)].astype(np.float32)
    return ode_policy.ODECarry(
        fake, executed_steps=executed, action_slope=slope, jump_dims=ode_policy.yam_arm_dims(), reset_jump_rad=jump
    )


def test_carry_starts_from_last_executed_reference_per_world():
    fake = _FakeSample()
    carry = _carry(fake)
    m0 = np.zeros((3, 32), np.float32)
    a1 = np.asarray(carry(None, _obs(m0)))
    np.testing.assert_array_equal(fake.starts[0], m0)  # first chunk: measured state
    m1 = m0 + 0.05  # arm lags / moved: the measured state is NOT the reference
    carry(None, _obs(m1))
    np.testing.assert_allclose(fake.starts[1], a1[:, 29, :])  # second chunk: row 29 of the first, per world
    # executed_steps < H: carry that row instead
    fake2 = _FakeSample()
    carry2 = _carry(fake2, executed=10)
    a = np.asarray(carry2(None, _obs(m0)))
    carry2(None, _obs(m1))
    np.testing.assert_allclose(fake2.starts[1], a[:, 9, :])


def test_carry_reset_paths():
    fake = _FakeSample()
    carry = _carry(fake)
    m = np.zeros((3, 32), np.float32)
    first = np.asarray(carry(None, _obs(m)))
    carry.reset(rows=[1])  # explicit per-row reset
    carry(None, _obs(m + 0.02))
    np.testing.assert_allclose(fake.starts[1][[0, 2]], first[[0, 2], 29])
    np.testing.assert_allclose(fake.starts[1][1], m[1] + 0.02)
    # a measured arm-joint jump > 0.5 rad from the carried reference restarts only that world
    prev = np.asarray(fake.starts[1]) + 0.3  # = the references carried into the next call
    m2 = prev.copy()
    m2[2, 3] += 0.6 / (NORM["actions"].std[3] + 1e-6)  # 0.6 rad on arm joint 4 of the left arm
    m2[0, 6] += 5.0  # a big GRIPPER gap (grasp: commanded closed, measured open) must NOT reset
    carry(None, _obs(m2))
    np.testing.assert_allclose(fake.starts[2][2], m2[2])
    np.testing.assert_allclose(fake.starts[2][[0, 1]], prev[[0, 1]], atol=1e-6)
    assert carry.stats["jump_resets"] == 1
    # batch size change (new runner / new set of worlds) resets everything
    carry(None, _obs(np.ones((2, 32), np.float32)))
    np.testing.assert_array_equal(fake.starts[3], np.ones((2, 32)))
    carry.reset()
    carry(None, _obs(np.full((2, 32), 2.0, np.float32)))
    np.testing.assert_array_equal(fake.starts[4], np.full((2, 32), 2.0))


# ----------------------------------------------------------------------------------------------- serve + infer
@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    """create_trained_policy on a tiny Pi0ODE checkpoint written to disk, with the real serving config transforms."""
    from flax import nnx
    import orbax.checkpoint as ocp

    from openpi.models import pi0_ode as _p
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_cfg = dataclasses.replace(
        _config.get_config("pi0_ode_put_bottles_infer"),
        model=_p.Pi0ODEConfig(paligemma_variant="dummy", action_horizon=30),
    )
    model = train_cfg.model.create(jax.random.key(0))
    # non-zero field output so actions actually move
    model.ode_field.out.kernel.value = (
        jax.random.normal(jax.random.key(1), model.ode_field.out.kernel.value.shape) * 0.05
    )
    ckpt = pathlib.Path(tmp_path_factory.mktemp("ckpt"))
    # bf16 on disk (the serving path restores in bf16 anyway): the dummy model still carries SigLIP So400m
    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), nnx.state(model).to_pure_dict())
    with ocp.PyTreeCheckpointer() as c:
        c.save(ckpt / "params", {"params": params})
    _normalize.save(ckpt / "assets" / "sim_put_bottles_mjwarp_rmperobj", NORM)
    policy = policy_config.create_trained_policy(train_cfg, ckpt)
    yield policy
    import shutil

    shutil.rmtree(ckpt, ignore_errors=True)


def _world_obs(rng, b):
    return [
        {
            "state": rng.uniform(-0.5, 0.5, 14).astype(np.float32),
            "prompt": "Put the plastic bottles in the bin",
            "top_camera-images-rgb": rng.integers(0, 255, (3, 168, 224), dtype=np.uint8),
            "left_camera-images-rgb": rng.integers(0, 255, (3, 168, 224), dtype=np.uint8),
            "right_camera-images-rgb": rng.integers(0, 255, (3, 168, 224), dtype=np.uint8),
        }
        for _ in range(b)
    ]


def _batched_infer(p, obs_list):
    """Exactly abc-rabc@33d6d6ce batched_policy.BatchedPi0.infer_batch."""
    from openpi.models import model as _m

    ins = [p._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]
    batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *ins)
    p._rng, rng = jax.random.split(p._rng)
    actions = np.asarray(p._sample_actions(rng, _m.Observation.from_dict(batch), **p._sample_kwargs))
    state_in = np.asarray(batch["state"])
    return np.stack(
        [
            np.asarray(p._output_transform({"state": state_in[w], "actions": actions[w]})["actions"])
            for w in range(len(obs_list))
        ]
    )


def test_batched_serve_round_trip_carries_per_world(tiny_policy):
    p = tiny_policy
    rng = np.random.default_rng(0)
    obs = _world_obs(rng, 4)
    a1 = _batched_infer(p, obs)
    assert a1.shape == (4, 30, 14) and np.all(np.isfinite(a1))
    carry = p._ode_carry
    # chunk 1 started from the measured state: row 0 is one integration step away from it, not a jump
    assert np.max(np.abs(a1[:, 0] - np.stack([o["state"] for o in obs]))) < 0.2
    # chunk 2: worlds moved a bit (lagging arm); the integration must start at chunk 1's executed end, per world
    obs2 = _world_obs(rng, 4)
    for o, a in zip(obs2, a1, strict=True):
        o["state"] = a[-1] + 0.03
    a2 = _batched_infer(p, obs2)
    assert carry.stats == {"calls": 2, "carried_rows": 4, "fresh_rows": 4, "jump_resets": 0}
    # same check in raw units: chunk 2 continues from chunk 1's last row (step-sized change, no restart)
    assert np.max(np.abs(a2[:, 0] - a1[:, -1])) < 0.2
    # a new episode in world 1 (pose jumps by > 0.5 rad): only world 1 restarts
    obs3 = _world_obs(rng, 4)
    for w, o in enumerate(obs3):
        o["state"] = a2[w, -1].copy()
    obs3[1]["state"][:6] += 1.0
    _batched_infer(p, obs3)
    assert carry.stats["jump_resets"] == 1 and carry.stats["fresh_rows"] == 5


def test_single_obs_infer_and_reset_flag(tiny_policy):
    p = tiny_policy
    p.reset()
    obs = _world_obs(np.random.default_rng(1), 1)[0]
    out = p.infer(dict(obs))
    assert out["actions"].shape == (30, 14)
    # the arm tracked the chunk (measured = executed end + small lag): the next chunk continues from the reference
    obs2 = {**obs, "state": out["actions"][-1] + 0.02}
    before = dict(p._ode_carry.stats)
    out2 = p.infer(dict(obs2))
    assert p._ode_carry.stats["jump_resets"] == before["jump_resets"]
    assert p._ode_carry.stats["carried_rows"] == before["carried_rows"] + 1
    assert (
        np.max(np.abs(out2["actions"][0] - out["actions"][-1]))
        < np.max(np.abs(out2["actions"][0] - obs2["state"])) + 1e-6
    )
    # obs["reset"] marks a new episode: the chunk starts from the measured state again (same as a fresh policy)
    out3 = p.infer({**obs, "reset": True})
    np.testing.assert_allclose(out3["actions"], out["actions"], atol=1e-5)


def test_websocket_round_trip(tiny_policy):
    import threading

    from openpi_client import websocket_client_policy

    from openpi.serving import websocket_policy_server

    p = tiny_policy
    p.reset()
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = websocket_policy_server.WebsocketPolicyServer(policy=p, host="127.0.0.1", port=port, metadata={})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
    obs = _world_obs(np.random.default_rng(2), 1)[0]
    out = client.infer(obs)
    assert np.asarray(out["actions"]).shape == (30, 14)


def test_carry_snaps_integrated_gripper_reference():
    fake = _FakeSample()
    slope = np.r_[1 / (NORM["actions"].std[:14] + 1e-6), np.zeros(18)].astype(np.float32)
    carry = ode_policy.ODECarry(
        fake, executed_steps=30, action_slope=slope, jump_dims=ode_policy.yam_arm_dims(), reset_jump_rad=0.5,
        snap_dims=(6, 13), snap_levels=np.array([[-1.0, 2.0], [-3.0, 0.5]], np.float32),
    )  # fmt: skip
    a1 = np.asarray(carry(None, _obs(np.zeros((3, 32), np.float32))))
    carry(None, _obs(np.zeros((3, 32), np.float32)))
    start = np.asarray(fake.starts[1])
    arm = list(ode_policy.yam_arm_dims())
    np.testing.assert_allclose(start[:, arm], a1[:, 29, arm])  # arm joints carried as is
    for d, (lo, hi) in zip((6, 13), ((-1.0, 2.0), (-3.0, 0.5)), strict=True):
        expect = np.where(np.abs(a1[:, 29, d] - lo) <= np.abs(a1[:, 29, d] - hi), lo, hi)
        np.testing.assert_allclose(start[:, d], expect)


def test_truncate_chunk():
    fake = _FakeSample()
    out = ode_policy.TruncateChunk(fake, 8)(None, _obs(np.zeros((2, 32), np.float32)))
    assert np.asarray(out).shape[1] == 8


def test_gripper_head_policy_execute_8_of_30(tmp_path, monkeypatch):
    """Serving the gripper-head model with OPENPI_EXECUTE_STEPS=8: 8-row chunks, carry from row 7 (per world)."""
    from flax import nnx
    import orbax.checkpoint as ocp

    from openpi.models import pi0_ode as _p
    from openpi.policies import policy_config
    from openpi.training import config as _config

    monkeypatch.setenv("OPENPI_EXECUTE_STEPS", "8")
    train_cfg = dataclasses.replace(
        _config.get_config("pi0_ode_put_bottles_grip_infer"),
        model=_p.Pi0ODEConfig(paligemma_variant="dummy", action_horizon=30, gripper_head_dims=(6, 13)),
    )
    model = train_cfg.model.create(jax.random.key(0))
    model.ode_field.out.kernel.value = (
        jax.random.normal(jax.random.key(1), model.ode_field.out.kernel.value.shape) * 0.05
    )
    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), nnx.state(model).to_pure_dict())
    with ocp.PyTreeCheckpointer() as c:
        c.save(tmp_path / "params", {"params": params})
    _normalize.save(tmp_path / "assets" / "sim_put_bottles_mjwarp_rmperobj", NORM)
    p = policy_config.create_trained_policy(train_cfg, tmp_path)
    assert p._ode_carry.executed_steps == 8
    rng = np.random.default_rng(0)
    obs = _world_obs(rng, 3)
    a1 = _batched_infer(p, obs)
    assert a1.shape == (3, 8, 14)
    obs2 = _world_obs(rng, 3)
    for o, a in zip(obs2, a1, strict=True):
        o["state"] = a[-1] + 0.01
    a2 = _batched_infer(p, obs2)
    arm = list(ode_policy.yam_arm_dims())
    assert p._ode_carry.stats["carried_rows"] == 3 and p._ode_carry.stats["jump_resets"] == 0
    # continues from row 7 per world: the boundary step is an ordinary integration step, not a restart
    step_in = np.abs(np.diff(a2[:, :, arm], axis=1)).max()
    assert np.abs(a2[:, 0, arm] - a1[:, -1, arm]).max() <= 1.5 * step_in + 1e-4
