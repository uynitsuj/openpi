import numpy as np

from openpi.models import model as _model
from openpi.policies import yam_policy
from openpi.training import config as training_config
import openpi.transforms as _transforms


def test_yam_inputs_supports_pi05_rabc_weights():
    transform = yam_policy.YamInputs(action_dim=32, model_type=_model.ModelType.PI05)
    data = {
        "state": np.arange(14, dtype=np.float32),
        "actions": np.arange(28, dtype=np.float32).reshape(2, 14),
        "top_camera-images-rgb": np.zeros((3, 4, 5), dtype=np.uint8),
        "prompt": "fold the shirt",
        "sample_weights": np.float64(0.75),
    }

    result = transform(data)

    assert result["state"].shape == (32,)
    assert result["actions"].shape == (2, 32)
    assert result["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert result["sample_weights"].shape == ()
    assert result["sample_weights"].dtype == np.float32
    assert result["sample_weights"] == np.float32(0.75)


def test_pi05_rabc_config_uses_yam_pi05_adapter(tmp_path):
    train_config = training_config.get_config("pi05_yam_tshirt_rabc")

    data_config = train_config.data.create(tmp_path, train_config.model)

    assert train_config.model.model_type == _model.ModelType.PI05
    assert train_config.rabc_enabled
    assert isinstance(data_config.data_transforms.inputs[0], yam_policy.YamInputs)
    assert data_config.data_transforms.inputs[0].model_type == _model.ModelType.PI05


def test_mjwarp310_sim_rabc_config_gates_on_final_action(tmp_path):
    train_config = training_config.get_config("pi0_sim_turn_mug_rabc_finalaction_thr100_nomax")

    data_config = train_config.data.create(tmp_path, train_config.model)

    assert train_config.model.model_type == _model.ModelType.PI0
    assert train_config.rabc_enabled
    assert train_config.num_train_steps == 30_000
    assert data_config.repo_id == "sim_turn_the_mug_right_side_up"
    # The velocity column must be fetched over the action-horizon window so the
    # final-action gate can read vel[-1].
    assert data_config.extra_horizon_keys == ("warp_rm_signed_magnitude",)

    rabc_transform, yam_inputs = data_config.data_transforms.inputs
    assert isinstance(rabc_transform, _transforms.ComputeRABCWeights)
    assert rabc_transform.use_final_action_condition
    assert rabc_transform.threshold == 1.0
    assert rabc_transform.clip_max == float("inf")
    assert rabc_transform.velocity_keys == ("warp_rm_signed_magnitude",)
    assert isinstance(yam_inputs, yam_policy.YamInputs)
    assert yam_inputs.model_type == _model.ModelType.PI0


def test_mjwarp310_sim_vanilla_bc_config(tmp_path):
    train_config = training_config.get_config("pi0_sim_turn_mug_no_rabc")

    data_config = train_config.data.create(tmp_path, train_config.model)

    assert train_config.model.model_type == _model.ModelType.PI0
    assert not train_config.rabc_enabled
    assert train_config.num_train_steps == 30_000
    assert data_config.repo_id == "sim_turn_the_mug_right_side_up"
    # Vanilla BC does not fetch the reward column at all.
    assert data_config.extra_horizon_keys == ()
    assert isinstance(data_config.data_transforms.inputs[0], yam_policy.YamInputs)
