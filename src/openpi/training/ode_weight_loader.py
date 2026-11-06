"""Strict pi0_base -> Pi0ODE parameter migration (after the paper's vlmquery_weight_loader).

Every parameter of the source checkpoint is either inherited unchanged (PaliGemma: SigLIP image encoder and the
Gemma 2B VLM stream) or explicitly dropped (pi0's action expert stream `*_1` inside the LLM, and pi0's action /
state / time projections). The only target parameters that may be missing from the source are the new ODE head
groups (`ode_query`, `ode_field`, and `ode_gripper` when the model has a gripper head), which keep their fresh initialisation. Anything else -- an unexpected extra or
missing key, a shape mismatch -- raises. This loader is for initialisation from pi0_base only; resuming or serving
a trained Pi0ODE checkpoint goes through the normal checkpoint restore.
"""

import dataclasses
import logging

from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import download

logger = logging.getLogger("openpi")

# pi0's action-expert modules inside the two-stream Gemma (gemma.py names the second stream "<name>_1").
EXPERT_MODULES = frozenset(
    (
        "attn_vec_einsum_1",
        "q_einsum_1",
        "kv_einsum_1",
        "mlp_1",
        "pre_attention_norm_1",
        "pre_ffw_norm_1",
        "final_norm_1",
    )
)
# pi0 (non-pi05) heads outside the LLM.
PI0_HEADS = frozenset(("state_proj", "action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj"))
PI0_HEAD_KEYS = frozenset(f"{h}/{p}" for h in PI0_HEADS for p in ("kernel", "bias"))
NEW_GROUPS = frozenset(("ode_query", "ode_field"))
# Optional new groups (present only for some Pi0ODEConfig settings): the direct gripper head (gripper_head_dims).
OPTIONAL_NEW_GROUPS = frozenset(("ode_gripper",))


def is_expert_key(key: str) -> bool:
    return key.startswith("PaliGemma/llm/") and bool(set(key.split("/")) & EXPERT_MODULES)


def plan(src_shapes: dict[str, tuple], ref_shapes: dict[str, tuple]) -> tuple[set[str], set[str], set[str]]:
    """Validate a migration on flat {key: shape} dicts; returns (inherited, removed, new) key sets or raises."""
    src, ref = set(src_shapes), set(ref_shapes)
    expert = {k for k in src if is_expert_key(k)}
    if not expert:
        raise ValueError("Expected a pi0 checkpoint with action-expert (*_1) parameters")
    if not src >= PI0_HEAD_KEYS:
        raise ValueError(f"Expected pi0 (non-pi05) head parameters, missing {sorted(PI0_HEAD_KEYS - src)}")
    removed = src - ref
    if removed != expert | PI0_HEAD_KEYS:
        raise ValueError(f"Unexpected removed parameters: {sorted(removed ^ (expert | PI0_HEAD_KEYS))}")
    if any(is_expert_key(k) or k.split("/")[0] in PI0_HEADS for k in ref):
        raise ValueError("Target model unexpectedly has action-expert / pi0 head parameters")
    new = ref - src
    groups = {k.split("/")[0] for k in new}
    if not NEW_GROUPS <= groups <= NEW_GROUPS | OPTIONAL_NEW_GROUPS:
        raise ValueError(
            f"Unexpected missing parameter groups {sorted(groups)} "
            f"(expected {sorted(NEW_GROUPS)} + optionally {sorted(OPTIONAL_NEW_GROUPS)})"
        )
    inherited = src & ref
    for k in sorted(inherited):
        if tuple(src_shapes[k]) != tuple(ref_shapes[k]):
            raise ValueError(f"Shape mismatch for {k}: {tuple(src_shapes[k])} != {tuple(ref_shapes[k])}")
    return inherited, removed, new


def migrate(loaded: at.Params, reference: at.Params) -> at.Params:
    """loaded: pi0_base params (pure dict); reference: the Pi0ODE param tree (arrays or ShapeDtypeStructs).

    Returns the reference tree with every inherited entry replaced by the checkpoint value (cast to the reference
    dtype); the new ODE entries keep the reference leaf (openpi's train.py then keeps their fresh init).
    """
    src, ref = flatten_dict(loaded, sep="/"), flatten_dict(reference, sep="/")
    inherited, removed, new = plan({k: v.shape for k, v in src.items()}, {k: v.shape for k, v in ref.items()})
    result = dict(ref)
    for k in inherited:
        result[k] = np.asarray(src[k]).astype(ref[k].dtype)
    logger.info(
        "Pi0ODE migration from pi0_base: inherited %d, dropped %d (action expert + pi0 heads), new %d %s",
        len(inherited), len(removed), len(new), sorted({k.split("/")[0] for k in new}),
    )  # fmt: skip
    return unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Pi0ODEWeightLoader:
    """openpi WeightLoader: pi0_base (VLM + image encoder) -> Pi0ODE; query and field start fresh."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return migrate(loaded, params)
