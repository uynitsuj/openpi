import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xmi_rby_example() -> dict:
    """Creates a random input example for the XMI RBY policy."""
    return {
        "left_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "top_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(20),  # [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif len(image.shape) == 4:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XmiRbyInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for XMI RBY robot.
    
    The XMI data uses end-effector poses with 6D rotation representation:
    - State format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper] = 20D
    - Three camera views: left, right (exterior), and top
    """
    
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    retarget_mode: Literal["20D-relative", "20D-intergripper-relative", "29D-relative", "29D-intergripper-relative"] = "29D-relative"
    use_top_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Extract the 20D end-effector state vector (29D if retargeting head with top camera)
        # Format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
        if "20D" in self.retarget_mode:
            state = data["state"][:20]
        elif "29D" in self.retarget_mode:
            state = data["state"][:29]
        else:
            raise ValueError(f"Unsupported retarget mode: {self.retarget_mode}")
        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse images to uint8 (H,W,C) format
        exterior_left_image = _parse_image(data["left_camera-images-rgb"])
        exterior_right_image = _parse_image(data["right_camera-images-rgb"])
        top_image = _parse_image(data["top_camera-images-rgb"])
        past_head_images = None

        match self.model_type:
            case _model.ModelType.PI0:
                # Pi0 models support three image inputs: one third-person view and two wrist views
                # For XMI, we use: base (top view), left wrist (left exterior), right wrist (right exterior)
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                if not self.use_top_camera:
                    images = (np.zeros_like(top_image), exterior_left_image, exterior_right_image)
                    image_masks = (np.False_, np.True_, np.True_) # For now skip top camera (XMI head and RBY head visual gap)
                else:
                    if len(top_image.shape) == 3:
                        images = (top_image, exterior_left_image, exterior_right_image)
                    elif len(top_image.shape) == 4: # multi tstep history for top camera
                        images = (top_image[0], exterior_left_image, exterior_right_image)
                        past_head_images = top_image[1:]

                    else:
                        raise ValueError(f"Unsupported top camera image shape: {top_image.shape}")

                    image_masks = (np.True_, np.True_, np.True_)
                
            case _model.ModelType.PI0_FAST:
                # Pi0-FAST uses: base_0, base_1, wrist_0
                # We'll use top as base_0, left exterior as base_1, right exterior as wrist_0
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                if not self.use_top_camera:
                    images = (np.zeros_like(top_image), exterior_left_image, exterior_right_image)
                else:
                    images = (top_image, exterior_left_image, exterior_right_image)
                # We don't mask out images for FAST models
                image_masks = (np.True_, np.True_, np.True_)
                
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "past_head_images": past_head_images,
        }

        # Add actions if available (during training)
        if "actions" in data:
            if "20D" in self.retarget_mode:
                actions = np.asarray(data["actions"])[:, :20]
            elif "29D" in self.retarget_mode:
                actions = np.asarray(data["actions"])[:, :29]
            else:
                raise ValueError(f"Unsupported retarget mode: {self.retarget_mode}")
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        # Add language instruction if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XmiRbyOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the XMI dataset specific format.
    It is used for inference only.
    """
    action_out_dim: int = 20
    
    def __call__(self, data: dict) -> dict:
        # Return the first 20 actions (end-effector pose deltas in 6D rotation + position format)
        # Format: [left_6d_rot_delta, left_3d_pos_delta, left_1d_gripper_abs, right_6d_rot_delta, right_3d_pos_delta, right_1d_gripper_abs]
        return {"actions": np.asarray(data["actions"][:, :self.action_out_dim])}
