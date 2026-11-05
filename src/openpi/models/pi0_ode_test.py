import dataclasses

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma as _gemma
from openpi.models import pi0
from openpi.models import pi0_ode
from openpi.training import ode_weight_loader


@pytest.fixture(scope="module")
def tiny():
    cfg = pi0_ode.Pi0ODEConfig(paligemma_variant="dummy", action_horizon=30)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(2)
    obs = dataclasses.replace(
        obs,
        ode_start=jax.random.normal(jax.random.key(1), obs.ode_start.shape),
        state=jax.random.normal(jax.random.key(2), obs.state.shape),
    )
    actions = jax.random.normal(jax.random.key(3), cfg.fake_act(2).shape)
    return cfg, model, obs, actions


def test_config_defaults_and_validation():
    cfg = pi0_ode.Pi0ODEConfig()
    assert cfg.action_horizon == 30 and cfg.ode_dim == 14 and not cfg.pi05 and cfg.field_width == 512
    assert cfg.model_type.value == "pi0"  # same transforms / tokenizer / z-score norm as the pi0 baseline
    assert len(cfg.perturb_std) == 14
    with pytest.raises(ValueError):
        pi0_ode.Pi0ODEConfig(pi05=True)
    with pytest.raises(ValueError):
        pi0_ode.Pi0ODEConfig(ode_dim=40)
    with pytest.raises(ValueError):
        pi0_ode.Pi0ODEConfig(perturb_std=(0.1, 0.2))


def test_parameter_groups_have_no_action_expert(tiny):
    _, model, _, _ = tiny
    keys = set(flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/"))
    roots = {k.split("/")[0] for k in keys}
    assert roots == {"PaliGemma", "ode_query", "ode_field"}
    assert not any(ode_weight_loader.is_expert_key(k) for k in keys)
    assert model.ode_query.value.shape == (1, 1, _gemma.get_config("dummy").width)


def test_zero_init_returns_start_and_shapes(tiny):
    cfg, model, obs, _ = tiny
    actions = model.sample_actions(jax.random.key(0), obs)
    assert actions.shape == (2, 30, 32)
    np.testing.assert_array_equal(
        actions[:, :, :14], np.broadcast_to(np.asarray(obs.ode_start[:, None, :14]), (2, 30, 14))
    )
    assert np.all(np.asarray(actions[:, :, 14:]) == 0)


def test_loss_shape_masking_and_gradients(tiny):
    cfg, model, obs, actions = tiny
    loss = model.compute_loss(jax.random.key(0), obs, actions, train=True)
    assert loss.shape == (2, 30) and np.all(np.isfinite(loss))
    # padded steps: changing their targets does not change the loss
    valid = np.ones((2, 30), bool)
    valid[1, 10:] = False
    obs_m = dataclasses.replace(obs, action_valid_mask=jnp.asarray(valid))
    l1 = model.compute_loss(jax.random.key(0), obs_m, actions, train=False)
    l2 = model.compute_loss(jax.random.key(0), obs_m, actions.at[1, 10:].add(50.0), train=False)
    np.testing.assert_allclose(l1, l2, rtol=1e-6)
    assert np.all(np.asarray(l1[1, 10:]) == 0)
    # padded dims (14..31) never enter the loss
    l3 = model.compute_loss(jax.random.key(0), obs_m, actions.at[:, :, 14:].add(50.0), train=False)
    np.testing.assert_allclose(l1, l3, rtol=1e-6)

    # With the zero-initialised output only the output layer gets gradient (exactly as in the paper); once it is
    # non-zero, the gradient reaches the field, the query token and the VLM / image encoder through c.
    def grad_norms(m):
        g = nnx.grad(lambda mm: jnp.mean(mm.compute_loss(jax.random.key(0), obs, actions, train=True)))(m)
        flat = flatten_dict(g.to_pure_dict(), sep="/")
        return {
            grp: sum(float(jnp.sum(jnp.square(v))) for k, v in flat.items() if k.startswith(grp))
            for grp in ("ode_query", "ode_field/out", "ode_field/inp", "PaliGemma/llm", "PaliGemma/img")
        }

    g0 = grad_norms(model)
    assert g0["ode_field/out"] > 0 and g0["ode_query"] == 0 and g0["PaliGemma/llm"] == 0
    graphdef, state = nnx.split(model)
    m2 = nnx.merge(graphdef, state)
    m2.ode_field.out.kernel.value = jax.random.normal(jax.random.key(5), m2.ode_field.out.kernel.value.shape) * 0.05
    g1 = grad_norms(m2)
    assert all(v > 0 for v in g1.values()), g1


def test_condition_ignores_state_and_start(tiny):
    """Proprioception goes to the field only (paper); the VLM condition sees images + prompt."""
    _, model, obs, _ = tiny
    c1 = model.condition(obs)
    c2 = model.condition(dataclasses.replace(obs, state=obs.state * 0 + 7.0, ode_start=obs.ode_start * 0 - 3.0))
    np.testing.assert_array_equal(c1, c2)
    assert c1.dtype == jnp.float32 and c1.shape == (2, model.ode_query.value.shape[-1])


def test_query_mask_prefix_does_not_attend_query():
    prefix_mask = jnp.array([[True, True, False, True]])
    mask = jnp.concatenate([prefix_mask, jnp.ones((1, 1), bool)], axis=1)
    ar = jnp.array([False, False, False, False, True])
    attn = np.asarray(pi0.make_attn_mask(mask, ar))[0]
    assert not attn[:4, 4].any()  # prefix rows never see the query
    np.testing.assert_array_equal(attn[4], [True, True, False, True, True])  # query sees the valid prefix + itself


def test_sample_actions_jits_and_matches_eager(tiny):
    _, model, obs, _ = tiny
    from openpi.shared import nnx_utils

    jitted = nnx_utils.module_jit(model.sample_actions)
    np.testing.assert_allclose(jitted(jax.random.key(0), obs), model.sample_actions(jax.random.key(0), obs), atol=1e-6)
