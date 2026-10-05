"""EgoMI (XMI) policy transforms for bimanual YAM data.

Ported from xdofai/openpi dev/justinyu/xmi_rby src/openpi/policies/xmi_rby_policy.py
(the 20D/29D EgoMI version). Kept as a separate module so the older 20D
XmiRbyInputs on this branch, used by other configs, is unchanged. The
keyframe-history path (past_head_images) is not ported; top_camera is a single frame.

Dataset keys (convert_to_lerobot.py in passive-xmi-vis): left_camera-images-rgb,
right_camera-images-rgb, top_camera-images-rgb, state (29D), actions (29D), prompt.
"""

import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xmi_yam_example() -> dict:
    return {
        "left_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "top_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(29),
        "prompt": "put the plastic bottles in the bin",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


RetargetMode = Literal["20D-relative", "20D-intergripper-relative", "29D-relative", "29D-intergripper-relative"]


def layout_dim(retarget_mode: str) -> int:
    if "20D" in retarget_mode:
        return 20
    if "29D" in retarget_mode:
        return 29
    raise ValueError(f"Unsupported retarget mode: {retarget_mode}")


@dataclasses.dataclass(frozen=True)
class XmiYamInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    retarget_mode: RetargetMode = "29D-intergripper-relative"
    use_top_camera: bool = True

    def __call__(self, data: dict) -> dict:
        d = layout_dim(self.retarget_mode)
        state = transforms.pad_to_dim(np.asarray(data["state"])[:d], self.action_dim)

        left = _parse_image(data["left_camera-images-rgb"])
        right = _parse_image(data["right_camera-images-rgb"])
        top = _parse_image(data["top_camera-images-rgb"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        if self.use_top_camera:
            images, image_masks = (top, left, right), (np.True_, np.True_, np.True_)
        else:
            images = (np.zeros_like(top), left, right)
            # pi0 masks the missing view; pi0-FAST is never masked (as in the reference).
            image_masks = (np.bool_(self.model_type != _model.ModelType.PI0), np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(np.asarray(data["actions"])[:, :d], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class XmiYamOutputs(transforms.DataTransformFn):
    """Inference only: return the first 20 or 29 action dims (absolute, after the
    data config's AbsoluteActions output transform)."""

    action_out_dim: int = 29

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_out_dim])}
