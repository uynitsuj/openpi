import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xmi_rby_example() -> dict:
    """Creates a random input example for the XMI RBY policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_3_top": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(20),  # [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
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

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Extract the 20D end-effector state vector
        # Format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper]
        state = data["observation/state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse images to uint8 (H,W,C) format
        exterior_left_image = _parse_image(data["observation/exterior_image_1_left"])
        exterior_right_image = _parse_image(data["observation/exterior_image_2_right"])
        top_image = _parse_image(data["observation/exterior_image_3_top"])

        match self.model_type:
            case _model.ModelType.PI0:
                # Pi0 models support three image inputs: one third-person view and two wrist views
                # For XMI, we use: base (top view), left wrist (left exterior), right wrist (right exterior)
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (top_image, exterior_left_image, exterior_right_image)
                image_masks = (np.True_, np.True_, np.True_)
                
            case _model.ModelType.PI0_FAST:
                # Pi0-FAST uses: base_0, base_1, wrist_0
                # We'll use top as base_0, left exterior as base_1, right exterior as wrist_0
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (top_image, exterior_left_image, exterior_right_image)
                # We don't mask out images for FAST models
                image_masks = (np.True_, np.True_, np.True_)
                
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Add actions if available (during training)
        if "actions" in data:
            actions = np.asarray(data["actions"])
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
    
    def __call__(self, data: dict) -> dict:
        # Return the first 20 actions (end-effector pose deltas in 6D rotation + position format)
        # Format: [left_6d_rot_delta, left_3d_pos_delta, left_1d_gripper_abs, right_6d_rot_delta, right_3d_pos_delta, right_1d_gripper_abs]
        return {"actions": np.asarray(data["actions"][:, :20])}
