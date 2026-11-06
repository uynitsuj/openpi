import json
import os

from flax import nnx
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import pi0_ode
from openpi.training import ode_weight_loader as owl


def _shapes(cfg):
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    return flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")


@pytest.fixture(scope="module")
def full_size():
    src = _shapes(pi0_config.Pi0Config(action_horizon=30))  # = pi0_base's structure
    ref = _shapes(pi0_ode.Pi0ODEConfig())
    return src, ref


def test_full_size_parameter_sets(full_size):
    src, ref = full_size
    inherited, removed, new = owl.plan({k: v.shape for k, v in src.items()}, {k: v.shape for k, v in ref.items()})
    # everything of PaliGemma except the action-expert stream is inherited; the img encoder entirely
    assert {k for k in src if k.startswith("PaliGemma/img/")} <= inherited
    assert {k for k in src if k.startswith("PaliGemma/llm/") and not owl.is_expert_key(k)} <= inherited
    assert inherited == {k for k in ref if k.startswith("PaliGemma/")}
    assert removed == {k for k in src if owl.is_expert_key(k)} | owl.PI0_HEAD_KEYS
    assert len({k for k in removed if owl.is_expert_key(k)}) == 8  # q, kv, attn_vec, mlp gating + linear, 3 norms
    assert {k.split("/")[0] for k in new} == {"ode_query", "ode_field"}
    n_new = sum(int(np.prod(ref[k].shape)) for k in new)
    n_vlm = sum(int(np.prod(ref[k].shape)) for k in inherited)
    # Gemma 2B + SigLIP; query (2,048) + 4 x 512 FiLM field (~9.4M, mostly the 2048 -> 2 x 512 FiLM projections)
    assert 2.8e9 < n_vlm < 3.0e9 and 9e6 < n_new < 1e7, (n_vlm, n_new)


def test_pi0_base_checkpoint_keys_match(full_size):
    """pi0_base's own orbax metadata (55 KB, no weights) has exactly the pi0 parameter set used above."""
    path = os.environ.get("PI0_BASE_METADATA")
    if path is None:
        fsspec = pytest.importorskip("fsspec")
        try:
            with fsspec.open("gs://openpi-assets/checkpoints/pi0_base/params/_METADATA", token="anon") as f:
                meta = json.load(f)
        except Exception as e:  # offline
            pytest.skip(f"pi0_base metadata unavailable: {e}")
    else:
        meta = json.load(open(path))
    keys = set()
    for k in meta["tree_metadata"]:
        parts = [p.strip(" '") for p in k.strip("()").split(",") if p.strip(" '")]
        assert parts[0] == "params"
        keys.add("/".join(parts[1:]))
    src, _ = full_size
    assert keys == set(src)


@pytest.fixture(scope="module")
def tiny_pair():
    src_model = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=30
    ).create(jax.random.key(1))
    src = nnx.state(src_model, nnx.Param).to_pure_dict()
    ref_cfg = pi0_ode.Pi0ODEConfig(paligemma_variant="dummy", action_horizon=30)
    ref = nnx.state(nnx.eval_shape(ref_cfg.create, jax.random.key(0)), nnx.Param).to_pure_dict()
    return src, ref, ref_cfg


def test_migrate_copies_vlm_and_keeps_new_params_fresh(tiny_pair):
    src, ref, ref_cfg = tiny_pair
    out = flatten_dict(owl.migrate(src, ref), sep="/")
    fsrc = flatten_dict(src, sep="/")
    for k, v in out.items():
        if k.startswith("PaliGemma/"):
            np.testing.assert_array_equal(np.asarray(v), np.asarray(fsrc[k]).astype(v.dtype))
        else:
            assert k.split("/")[0] in ("ode_query", "ode_field") and isinstance(v, jax.ShapeDtypeStruct)
    # train.py's flow: drop the ShapeDtypeStructs, merge into a freshly initialised model
    partial = unflatten_dict({k: v for k, v in out.items() if not isinstance(v, jax.ShapeDtypeStruct)}, sep="/")
    model = ref_cfg.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(partial)
    model = nnx.merge(graphdef, state)
    np.testing.assert_array_equal(
        model.PaliGemma.llm.embedder["input_embedding"].value, fsrc["PaliGemma/llm/embedder/input_embedding"]
    )
    assert np.all(np.asarray(model.ode_field.out.kernel.value) == 0)


@pytest.mark.parametrize("mutation", ["extra", "missing_vlm", "missing_head", "shape", "no_new_group"])
def test_migration_is_strict(tiny_pair, mutation):
    src, ref, _ = tiny_pair
    fsrc, fref = dict(flatten_dict(src, sep="/")), dict(flatten_dict(ref, sep="/"))
    if mutation == "extra":
        fsrc["surprise/kernel"] = np.zeros(1)
    elif mutation == "missing_vlm":
        del fsrc["PaliGemma/img/pos_embedding"]
    elif mutation == "missing_head":
        del fsrc["state_proj/kernel"]
    elif mutation == "shape":
        fsrc["PaliGemma/llm/final_norm/scale"] = np.zeros(3)
    elif mutation == "no_new_group":
        fref = {k: v for k, v in fref.items() if not k.startswith("ode_query")}
    with pytest.raises(ValueError):
        owl.migrate(unflatten_dict(fsrc, sep="/"), unflatten_dict(fref, sep="/"))


def test_migrate_with_gripper_head(tiny_pair):
    src, _, _ = tiny_pair
    cfg = pi0_ode.Pi0ODEConfig(paligemma_variant="dummy", action_horizon=8, gripper_head_dims=(6, 13))
    ref = nnx.state(nnx.eval_shape(cfg.create, jax.random.key(0)), nnx.Param).to_pure_dict()
    out = flatten_dict(owl.migrate(src, ref), sep="/")
    new = {k.split("/")[0] for k, v in out.items() if isinstance(v, jax.ShapeDtypeStruct)}
    assert new == {"ode_query", "ode_field", "ode_gripper"}
