import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "state": np.ones((14,)),
        "left_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "right_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "top_camera-images-rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs for the YAM policy.

    Expected inputs:
    - left_camera-images-rgb: [channel, height, width] - Left camera
    - right_camera-images-rgb: [channel, height, width] - Right camera  
    - top_camera-images-rgb: [channel, height, width] - Top camera
    - state: [14] - Joint positions for both arms (6 joints + 1 gripper per arm)
    - actions: [action_horizon, 14] - Joint actions for both arms (absolute joint space)
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected camera names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb")

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Process images
        in_images = {}
        for camera_key in ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]:
            if camera_key in data:
                in_images[camera_key] = data[camera_key]

        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain subset of {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Convert images to the expected format
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel] if needed.
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        # Process available images
        processed_images = {}
        for camera_key in in_images:
            processed_images[camera_key] = convert_image(in_images[camera_key])

        # Use the first available image as base image, or create a default
        if processed_images:
            base_image = next(iter(processed_images.values()))
        else:
            base_image = np.zeros((224, 224, 3), dtype=np.uint8)

        match self.model_type:
            case _model.ModelType.PI0:
                # Map YAM cameras to standard PI0 camera names
                images = {
                    "base_0_rgb": processed_images.get("top_camera-images-rgb", np.zeros_like(base_image)),  # Top camera as base
                    "left_wrist_0_rgb": processed_images.get("left_camera-images-rgb", np.zeros_like(base_image)),  # Left camera
                    "right_wrist_0_rgb": processed_images.get("right_camera-images-rgb", np.zeros_like(base_image)),  # Right camera
                }
                image_masks = {
                    "base_0_rgb": "top_camera-images-rgb" in processed_images,
                    "left_wrist_0_rgb": "left_camera-images-rgb" in processed_images,
                    "right_wrist_0_rgb": "right_camera-images-rgb" in processed_images,
                }
            case _model.ModelType.PI0_FAST:
                # For FAST models, we don't mask out padding images
                images = {
                    "base_0_rgb": processed_images.get("top_camera-images-rgb", np.zeros_like(base_image)),
                    "base_1_rgb": processed_images.get("left_camera-images-rgb", np.zeros_like(base_image)),
                    "wrist_0_rgb": processed_images.get("right_camera-images-rgb", np.zeros_like(base_image)),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # YAM uses absolute joint actions - no conversion needed
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs for the YAM policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims (YAM joint space).
        actions = np.asarray(data["actions"][:, :14])
        # YAM uses absolute joint actions - no conversion needed
        return {"actions": actions}
