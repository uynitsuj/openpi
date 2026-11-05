import ast
import os
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import ode_field


def _field(cond=5, state=14, dim=14, width=16, layers=4, seed=0):
    return ode_field.VectorField(cond, state, dim, width, num_layers=layers, rngs=nnx.Rngs(seed))


def _randomize_output(field, seed=1, std=0.05):
    field.out.kernel.value = jax.random.normal(jax.random.key(seed), field.out.kernel.value.shape) * std
    field.out.bias.value = jax.random.normal(jax.random.key(seed + 1), field.out.bias.value.shape) * std


def test_field_shapes_and_zero_initialised_output():
    field = _field()
    assert field.inp.kernel.value.shape == (28, 16)
    assert field.out.kernel.value.shape == (16, 14)
    assert np.all(field.out.kernel.value == 0) and np.all(field.out.bias.value == 0)
    film = field.precompute(jnp.ones((3, 5)))
    assert len(film) == 4 and film[0][0].shape == (3, 16)
    assert field(jnp.ones((3, 14)), jnp.ones((3, 14)), film).shape == (3, 14)
    # PyTorch-default init bounds 1/sqrt(fan_in) (paper's author_linear)
    assert np.max(np.abs(field.inp.kernel.value)) <= 28**-0.5


def test_zero_init_field_returns_start_state():
    field = _field()
    a0 = jax.random.normal(jax.random.key(0), (3, 14))
    path = ode_field.integrate(field, a0, jnp.ones((3, 14)), jnp.ones((3, 5)), horizon=30, vel_scale=0.05)
    assert path.shape == (3, 30, 14)
    np.testing.assert_array_equal(path, np.broadcast_to(np.asarray(a0)[:, None], (3, 30, 14)))


class _LinearField:
    """da/dt = vel_scale * (M a + b): integrate() only needs precompute() and __call__."""

    def __init__(self, m, b):
        self.m, self.b = m, b

    def precompute(self, c):
        return None

    def __call__(self, a, s, film):
        return a @ self.m.T + self.b


def test_rk4_matches_analytic_linear_field():
    # Diagonal stable linear field: exact solution a(t) = a* + (a0 - a*) exp(lambda vel t), a* = -b / lambda.
    lam = np.array([-1.0, -3.0, -0.2, -6.0], np.float32)
    b = np.array([0.5, -1.0, 0.1, 2.0], np.float32)
    field = _LinearField(jnp.diag(lam), jnp.asarray(b))
    vel, horizon = 0.05, 30
    a0 = np.array([[1.0, -2.0, 0.3, 0.0]], np.float32)
    path = np.asarray(ode_field.integrate(field, jnp.asarray(a0), None, None, horizon=horizon, vel_scale=vel))
    t = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
    star = -b / lam
    exact = star + (a0[0] - star) * np.exp(lam * vel * t)
    np.testing.assert_allclose(path[0], exact, atol=5e-5)  # RK4 local error ~ z^5/120, z = -0.3 on the stiffest dim
    # ... and exactly the RK4 amplification factor 1 + z + z^2/2 + z^3/6 + z^4/24 per step (z = lambda * vel)
    z = lam.astype(np.float64) * vel
    r = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    rk4 = star + (a0[0] - star) * r**t
    np.testing.assert_allclose(path[0], rk4, atol=2e-6)


def test_rk4_backprop_matches_finite_differences():
    field = _field(width=8, layers=2)
    _randomize_output(field, std=0.3)
    s, c = jnp.ones((2, 14)) * 0.3, jnp.ones((2, 5)) * 0.2
    a0 = jax.random.normal(jax.random.key(3), (2, 14))

    def loss(a):
        return jnp.sum(jnp.sin(ode_field.integrate(field, a, s, c, horizon=6, vel_scale=0.5)))

    g = np.asarray(jax.grad(loss)(a0), np.float64)
    eps = 1e-3
    for i, j in [(0, 0), (1, 5), (0, 13)]:
        e = jnp.zeros_like(a0).at[i, j].set(eps)
        fd = (float(loss(a0 + e)) - float(loss(a0 - e))) / (2 * eps)
        np.testing.assert_allclose(g[i, j], fd, rtol=2e-2, atol=2e-3)


def test_path_loss_masks_padding_and_averages_valid_steps():
    rng = np.random.default_rng(0)
    path, target = (rng.normal(size=(2, 30, 14)).astype(np.float32) for _ in range(2))
    valid = np.ones((2, 30), bool)
    valid[1, 20:] = False
    loss = np.asarray(ode_field.path_loss(path, target, valid, vel_scale=0.05))
    per_step = np.mean(((path - target) / 0.05) ** 2, -1)
    # mean over H of the returned [B, H] == per-example mean over VALID steps (paper's sum_valid / n_valid)
    np.testing.assert_allclose(loss.mean(-1), [per_step[0].mean(), per_step[1, :20].mean()], rtol=1e-5)
    assert np.all(loss[1, 20:] == 0)
    # changing the target on padded steps changes nothing
    target2 = target.copy()
    target2[1, 20:] += 100
    np.testing.assert_array_equal(loss, ode_field.path_loss(path, target2, valid, vel_scale=0.05))
    # a fully padded example contributes zero, not NaN
    assert np.all(np.asarray(ode_field.path_loss(path, target, np.zeros((2, 30), bool), vel_scale=0.05)) == 0)


def test_start_perturbation_is_contractive():
    a0 = jnp.zeros((256, 14))
    target = jnp.ones((256, 30, 14))
    std = jnp.full((14,), 0.1)
    a1, t1 = ode_field.perturb_start(jax.random.key(0), a0, target, std, prob=0.5, decay=0.7)
    delta = np.asarray(a1 - a0)
    np.testing.assert_allclose(
        np.asarray(t1 - target), delta[:, None, :] * (0.7 ** np.arange(1, 31))[None, :, None], atol=1e-6
    )
    on = np.any(delta != 0, axis=-1)
    assert 0.3 < on.mean() < 0.7  # about half the samples
    assert 0.07 < np.std(delta[on]) < 0.13
    a2, t2 = ode_field.perturb_start(jax.random.key(0), a0, target, std, prob=0.0)
    np.testing.assert_array_equal(a2, a0)
    np.testing.assert_array_equal(t2, target)


def test_std_vector():
    assert ode_field.as_std_vector(0.01, 3) == (0.01, 0.01, 0.01)
    with pytest.raises(ValueError):
        ode_field.as_std_vector((0.1, 0.2), 3)


def test_pure_dict_roundtrip():
    field = _field()
    state = nnx.state(field)
    state.replace_by_pure_dict(state.to_pure_dict())
    assert nnx.graphdef(_field()) == nnx.graphdef(_field())


def _author_policy_file():
    root = os.environ.get("ODE_POLICY_ROOT")
    return pathlib.Path(root) / "ode_policy/policies/ode/ode_image_policy.py" if root else None


@pytest.mark.skipif(_author_policy_file() is None, reason="set ODE_POLICY_ROOT to the NTUMARS/ODE-POLICY checkout")
def test_matches_author_pytorch_field_and_rk4():
    """Same check as the paper patch's ode_field_test, at our sizes (14-D state incl. grippers, H = 30)."""
    torch = pytest.importorskip("torch")
    source = _author_policy_file()
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("_FiLMLayer", "VectorField")]
    scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "POSE_DIM": 6}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    torch.manual_seed(0)
    author = scope["VectorField"](7, 14, 12, 4, out_dim=14, pose_dim=14)
    torch.nn.init.normal_(author.out.weight, std=0.05)
    port = ode_field.VectorField(7, 14, 14, 12, rngs=nnx.Rngs(0))
    pairs = [(author.inp, port.inp), (author.out, port.out)]
    for a, b in zip(author.layers, port.layers.values(), strict=True):
        pairs += [(a.lin, b.lin), (a.film, b.film)]
    for a, b in pairs:
        b.kernel.value = jnp.asarray(a.weight.detach().numpy().T)
        b.bias.value = jnp.asarray(a.bias.detach().numpy())
    rng = np.random.default_rng(0)
    a, s, c = (rng.normal(size=(2, d)).astype(np.float32) for d in (14, 14, 7))
    ta, ts, tc = (torch.tensor(x) for x in (a, s, c))
    film = author.precompute(tc)
    x, states = ta, []
    for _ in range(30):

        def f(z):
            return 0.05 * author(z, ts, tc, film)

        k1 = f(x)
        k2 = f(x + 0.5 * k1)
        k3 = f(x + 0.5 * k2)
        k4 = f(x + k3)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(x)
    expected = torch.stack(states, 1).detach().numpy()
    got = ode_field.integrate(port, jnp.asarray(a), jnp.asarray(s), jnp.asarray(c), horizon=30, vel_scale=0.05)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-4)
