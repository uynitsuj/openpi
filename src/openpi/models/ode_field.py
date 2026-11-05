"""ODE action head: vector field, RK4 integration, start perturbation and path loss.

JAX port of the ODE policy's head (github.com/NTUMARS/ODE-POLICY, branch vla-pi05,
vla/pi05_ode/openpi_ode.patch: src/openpi/models/ode_field.py), generalised from the
6-D end-effector pose to an arbitrary action state of `field_dim` entries (here: the
14 absolute joint-position references of the YAM bimanual arm, grippers included).

    da/dt = vel_scale * f(a, s | c)

`a` is the action state (the joint reference the controller is sent to), `s` the measured
proprioception, `c` the per-chunk condition read out of the VLM. `c` and `s` are held fixed
over the chunk. Time unit = one control step (one row of the action chunk), so the path
returned by `integrate` holds the state after control steps 1..H.
"""

from collections.abc import Sequence
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class AuthorUniform:
    """PyTorch nn.Linear default initialisation U(-1/sqrt(fan_in), 1/sqrt(fan_in)).

    A frozen dataclass (not a closure) so the initializer is identical static graph metadata
    across nnx.eval_shape and real init (see the paper patch's ode_gripper_test).
    """

    bound: float

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-self.bound, maxval=self.bound)


def author_linear(in_features: int, out_features: int, *, rngs: nnx.Rngs, zero: bool = False) -> nnx.Linear:
    init = jax.nn.initializers.zeros if zero else AuthorUniform(in_features**-0.5)
    return nnx.Linear(in_features, out_features, kernel_init=init, bias_init=init, rngs=rngs)


class FiLMLayer(nnx.Module):
    def __init__(self, width: int, cond_dim: int, *, rngs: nnx.Rngs):
        self.lin = author_linear(width, width, rngs=rngs)
        self.film = author_linear(cond_dim, 2 * width, rngs=rngs)

    def __call__(self, h, gb):
        gain, bias = gb
        return h + jax.nn.silu(self.lin(h) * (1 + gain) + bias)


class VectorField(nnx.Module):
    """FiLM residual MLP f(a, s | c) with a zero-initialised output projection.

    At initialisation f == 0, so the integrated path stays at the start state.
    """

    def __init__(
        self, cond_dim: int, state_dim: int, field_dim: int, width: int, *, rngs: nnx.Rngs, num_layers: int = 4
    ):
        self.inp = author_linear(field_dim + state_dim, width, rngs=rngs)
        # String-keyed dict (not a list): numeric keys break nnx pure-dict round trips (paper patch,
        # ode_state_keys_test) and openpi's '/'-joined checkpoint keys.
        self.layers = nnx.Dict({f"layer_{i}": FiLMLayer(width, cond_dim, rngs=rngs) for i in range(num_layers)})
        self.out = author_linear(width, field_dim, zero=True, rngs=rngs)

    def precompute(self, c):
        """FiLM gains/biases depend on c only: compute them once per chunk."""
        return [jnp.split(layer.film(c), 2, axis=-1) for layer in self.layers.values()]

    def __call__(self, a, s, film):
        h = self.inp(jnp.concatenate([a, s], axis=-1))
        for layer, gb in zip(self.layers.values(), film, strict=True):
            h = layer(h, gb)
        return self.out(h)


def integrate(field: VectorField, a0, s, c, *, horizon: int, vel_scale: float):
    """Fixed-step RK4 with the condition held fixed: four field calls per control step.

    a0: [B, D] start state, s: [B, S], c: [B, C]. Returns the path [B, horizon, D]
    (state after each control step; the start state itself is not included).
    """
    film = field.precompute(c)

    def f(x):
        return vel_scale * field(x, s, film)

    def step(a, _):
        k1 = f(a)
        k2 = f(a + 0.5 * k1)
        k3 = f(a + 0.5 * k2)
        k4 = f(a + k3)
        a = a + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return a, a

    _, path = jax.lax.scan(step, a0, None, length=horizon)
    return jnp.swapaxes(path, 0, 1)


def perturb_start(key, a0, target, std, *, prob: float = 0.5, decay: float = 0.7):
    """Contractive start perturbation (paper): on a fraction `prob` of the samples shift the
    start by delta ~ N(0, std) and ask the path to return to the demonstration,
    target_k + delta * decay**k (k = 1..H).

    std: [B, D] or [D], already in the normalised units of a0 / target.
    """
    gate_key, noise_key = jax.random.split(key)
    on = jax.random.bernoulli(gate_key, prob, (a0.shape[0], 1)).astype(a0.dtype)
    delta = jax.random.normal(noise_key, a0.shape, a0.dtype) * std * on
    k = jnp.arange(1, target.shape[1] + 1, dtype=a0.dtype)
    return a0 + delta, target + delta[:, None, :] * (decay**k)[None, :, None]


def path_loss(path, target, valid, *, vel_scale: float):
    """Path MSE in units of one nominal step (the paper's (path - target) / vel_scale), padding masked.

    Returns [B, H] per-step losses scaled so that their mean over H equals the paper's per-example
    loss sum_valid / n_valid; openpi's train step averages over H (and over the batch).
    """
    per_step = jnp.mean(jnp.square((path - target) / vel_scale), axis=-1)
    valid = valid.astype(per_step.dtype)
    n_valid = jnp.maximum(jnp.sum(valid, axis=-1, keepdims=True), 1.0)
    return per_step * valid * (valid.shape[-1] / n_valid)


def as_std_vector(std: float | Sequence[float], dim: int) -> tuple[float, ...]:
    """Broadcast a scalar / per-dimension perturbation std to `dim` entries."""
    if isinstance(std, int | float):
        return (float(std),) * dim
    std = tuple(float(x) for x in std)
    if len(std) != dim:
        raise ValueError(f"perturb_std: expected 1 or {dim} values, got {len(std)}")
    return std
