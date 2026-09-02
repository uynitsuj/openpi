"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, List, Literal

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.xmi_rby_policy as xmi_rby_policy
import openpi.policies.yam_policy as yam_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # RABC: when True, the dataloader wraps the transformed dataset with
    # RejectionSamplingTransformedDataset so chunks emitting
    # sample_weights == 0 (e.g., final-action gate failed) are replaced by a
    # fresh uniform draw rather than consuming a batch slot with zero gradient.
    # Default True matches the behavior on origin/rorm-rabc (cherry-picked
    # 2026-05-18). No-op if no transform emits sample_weights.
    rabc_reject_zero_weighted: bool = True

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # Additional keys to fetch with the same delta_timestamps as actions (e.g., "rorm_velocity").
    # These will be available in the data dict for transforms to consume.
    extra_horizon_keys: Sequence[str] = ()

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, repo_id points at an ABC training layout (MCAP export: states_actions.bin +
    # strict CFR combined video) under HF_LEROBOT_HOME, loaded with abc's random-access
    # approach instead of LeRobotDataset. See openpi/training/abc_layout_dataset.py.
    abc_layout: bool = False
    # ABC layout only: keep episodes from these station_types (episode_metadata.json).
    # None trains on everything. E.g. ("yam_zed_0_61",) for a ZED-only policy.
    abc_station_types: Sequence[str] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # Episode filtering for LeRobot datasets. If set, only these episode indices are used for training.
    episodes: tuple[int, ...] | None = None
    # Held-out validation episodes. Used by val dataloader if set.
    val_episodes: tuple[int, ...] | None = None

    # If True, the data loader filters out samples whose RABC weight is 0 so
    # every batch has fully positive weights. ``reject_zero_weighted_mode``
    # picks the implementation:
    #   - "subset" (default): precompute the valid flat indices once from
    #     parquet (no video decode), wrap the transformed dataset in
    #     torch.utils.data.Subset. Single decision per index, no runtime cost.
    #   - "rejection": per-getitem rejection sampling — re-decodes video on
    #     each retry. Kept as a fallback for cases where the subset precompute
    #     can't be trusted (custom transforms, dynamic weights).
    # No-op when no `sample_weights` key is produced (rabc disabled).
    reject_zero_weighted_samples: bool = True
    reject_zero_weighted_mode: str = "subset"

    # When >0, the loader fetches action_horizon + this-many frames for each
    # key in ``extra_horizon_keys``. Lets RABC aggregators look beyond the
    # action chunk (e.g. velocity_aggregator='mean_lookahead' averages over
    # the trailing portion only). Default 0 = no lookahead.
    extra_horizon_lookahead_frames: int = 0


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXmiRbyDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms for the XMI RBY bimanual robot dataset.
    
    The XMI data uses end-effector poses with 6D rotation representation:
    - State format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper] = 20D
    - Three camera views: left exterior, right exterior, and top
    - Actions are delta end-effector poses with absolute gripper positions
    """
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform to map dataset keys to policy keys
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_right": "exterior_image_2_right",
                        "observation/exterior_image_3_top": "exterior_image_3_top",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms using XMI RBY policy transforms
        data_transforms = _transforms.Group(
            inputs=[xmi_rby_policy.XmiRbyInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[xmi_rby_policy.XmiRbyOutputs()],
        )

        # XMI data uses delta actions for rotations/positions, but absolute gripper positions
        # The conversion script already produces the correct format, but we may need delta conversion
        # for the rotations and positions (indices 0:6, 6:9, 10:16, 16:19) while keeping
        # grippers absolute (indices 9, 19)
        delta_action_mask = _transforms.make_bool_mask(
            6, 3, -1,  # left: 6d_rot (delta), 3d_pos (delta), gripper (absolute)
            6, 3, -1   # right: 6d_rot (delta), 3d_pos (delta), gripper (absolute) 
        )
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms for tokenization and image processing
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

def _total_episodes(repo_id: str) -> int:
    """Read total_episodes from the dataset's meta/info.json."""
    import json
    from lerobot.utils.constants import HF_LEROBOT_HOME
    info = json.loads((HF_LEROBOT_HOME / repo_id / "meta" / "info.json").read_text())
    return int(info["total_episodes"])


def _split_val_episodes(
    repo_id: str, val_frac: float, val_seed: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Hold out ``val_frac`` of episodes for validation.

    Returns (val_episodes, non_val_episodes). Deterministic across configs that
    share ``val_seed`` so topq/rabc runs all see the same held-out val set.
    """
    import numpy as np
    if val_frac <= 0:
        return (), tuple(range(_total_episodes(repo_id)))
    n = _total_episodes(repo_id)
    rng = np.random.default_rng(val_seed)
    k = max(1, int(round(val_frac * n)))
    perm = rng.permutation(n)
    val = tuple(sorted(int(x) for x in perm[:k]))
    non_val = tuple(sorted(int(x) for x in perm[k:]))
    return val, non_val


def _episode_mean_q(repo_id: str) -> dict[int, float]:
    """Scan the dataset's parquet files and return {episode_index: mean Q}.

    Reads ``repromo_quality`` (canonical) or falls back to legacy ``rorm_q``.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = HF_LEROBOT_HOME / repo_id
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {root}/data")

    ep_sum: dict[int, float] = {}
    ep_count: dict[int, int] = {}
    for f in parquet_files:
        schema_cols = set(pq.read_schema(f).names)
        q_col = (
            "repromo_quality" if "repromo_quality" in schema_cols
            else "rorm_q" if "rorm_q" in schema_cols
            else "sarm_dense_quality" if "sarm_dense_quality" in schema_cols
            else None
        )
        if q_col is None:
            raise KeyError(
                f"Neither 'repromo_quality', 'rorm_q', nor 'sarm_dense_quality' in {f}; "
                f"has the dataset been injected with reward annotations?"
            )
        t = pq.read_table(f, columns=["episode_index", q_col])
        eps = np.asarray(t["episode_index"]).astype(np.int64).ravel()
        qs = np.asarray(t[q_col]).astype(np.float64).ravel()
        for e, q in zip(eps, qs):
            e = int(e)
            ep_sum[e] = ep_sum.get(e, 0.0) + q
            ep_count[e] = ep_count.get(e, 0) + 1
    return {e: ep_sum[e] / ep_count[e] for e in ep_sum}


def _top_q_episodes(
    repo_id: str, frac: float, exclude_eps: tuple[int, ...] = (),
) -> list[int]:
    """Return sorted episode indices whose mean rorm_q is in the top ``frac`` of the pool."""
    if not 0 < frac <= 1:
        raise ValueError(f"top_q_frac must be in (0, 1], got {frac}")
    import numpy as np

    ep_to_mean = _episode_mean_q(repo_id)
    excluded = set(exclude_eps)
    candidate_ids = [e for e in ep_to_mean if e not in excluded]
    ep_ids = np.array(candidate_ids)
    mean_q = np.array([ep_to_mean[e] for e in ep_ids])
    order = np.argsort(mean_q)[::-1]
    total = len(ep_to_mean)
    k = max(1, int(round(frac * total)))
    k = min(k, len(ep_ids))
    kept = sorted(int(ep_ids[i]) for i in order[:k])
    logging.info(
        f"top_q_frac={frac}: keeping {k}/{total} episodes "
        f"(q >= {float(mean_q[order[k - 1]]):.4f}, excluded {len(excluded)} val eps)"
    )
    return kept


def _shortest_episodes(
    repo_id: str, frac: float, exclude_eps: tuple[int, ...] = (),
) -> list[int]:
    """Return sorted episode indices with the shortest lengths in the bottom ``frac`` of the pool.

    Supports both lerobot v2.1 (meta/episodes.jsonl) and v3.0 (meta/episodes/chunk-*/file-*.parquet).
    """
    if not 0 < frac <= 1:
        raise ValueError(f"top_shortest_frac must be in (0, 1], got {frac}")
    import json
    import numpy as np
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = HF_LEROBOT_HOME / repo_id
    ep_lengths: dict[int, int] = {}
    legacy_path = root / "meta" / "episodes.jsonl"
    v3_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if v3_files:
        import pyarrow.parquet as pq
        for f in v3_files:
            t = pq.read_table(f, columns=["episode_index", "length"])
            for ep_idx, length in zip(t["episode_index"].to_pylist(), t["length"].to_pylist()):
                ep_lengths[int(ep_idx)] = int(length)
    elif legacy_path.exists():
        with open(legacy_path) as f:
            for line in f:
                rec = json.loads(line)
                ep_lengths[int(rec["episode_index"])] = int(rec["length"])
    else:
        raise FileNotFoundError(f"no episode metadata under {root}/meta")

    excluded = set(exclude_eps)
    candidate_ids = [e for e in ep_lengths if e not in excluded]
    ep_ids = np.array(candidate_ids)
    lengths = np.array([ep_lengths[e] for e in ep_ids])
    order = np.argsort(lengths)  # ascending: shortest first
    total = len(ep_lengths)
    k = max(1, int(round(frac * total)))
    k = min(k, len(ep_ids))
    kept = sorted(int(ep_ids[i]) for i in order[:k])
    logging.info(
        f"top_shortest_frac={frac}: keeping {k}/{total} episodes "
        f"(length <= {int(lengths[order[k - 1]])}, excluded {len(excluded)} val eps)"
    )
    return kept


def _load_rabc_q_range(repo_id: str) -> tuple[float | None, float | None]:
    """Load q_min/q_max for the per-frame quality column.

    Prefers ``meta/rabc_stats.json`` (cheap, precomputed). Falls back to a one-
    pass parquet scan over ``repromo_quality`` (or legacy ``rorm_q``) so configs
    work on datasets that haven't had the sidecar file generated yet.
    """
    import json
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = HF_LEROBOT_HOME / repo_id
    p = root / "meta" / "rabc_stats.json"
    if p.exists():
        stats = json.loads(p.read_text())
        if stats.get("q_min") is not None and stats.get("q_max") is not None:
            return stats["q_min"], stats["q_max"]

    # Fallback: scan parquets for per-frame quality range.
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        return None, None
    schema_cols = set(pq.read_schema(parquet_files[0]).names)
    q_col = (
        "repromo_quality" if "repromo_quality" in schema_cols
        else "rorm_q" if "rorm_q" in schema_cols
        else "sarm_dense_quality" if "sarm_dense_quality" in schema_cols
        else None
    )
    if q_col is None:
        return None, None
    q_min, q_max = float("inf"), float("-inf")
    for f in parquet_files:
        col = np.asarray(pq.read_table(f, columns=[q_col])[q_col]).astype(np.float64).ravel()
        if col.size == 0:
            continue
        q_min = min(q_min, float(col.min()))
        q_max = max(q_max, float(col.max()))
    if not np.isfinite(q_min) or not np.isfinite(q_max):
        return None, None
    logging.info(f"_load_rabc_q_range: scanned parquets for {repo_id}, q ∈ [{q_min:.4f}, {q_max:.4f}]")
    return q_min, q_max


def _compute_sarm_reward_stats(
    repo_id: str, action_horizon: int, progress_col: str = "sarm_dense_progress",
) -> tuple[float, float]:
    """Compute mean and std of progress-delta rewards for SARM weighting.

    Prefers ``meta/sarm_reward_stats.json`` (keyed by horizon) for fast startup.
    Falls back to a full parquet scan computing reward = progress[t + horizon - 1] - progress[t]
    for each valid frame, then returns (μ, σ) for use in the SARM weighting formula.
    """
    import json
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = HF_LEROBOT_HOME / repo_id

    # Fast path: check pre-computed cache.
    cache_path = root / "meta" / "sarm_reward_stats.json"
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())
        key = f"{action_horizon}_{progress_col}"
        # Fall back to horizon-only key only for the default column
        # (backward compat with pre-column-aware caches).
        if key not in stats and progress_col == "sarm_dense_progress":
            key = str(action_horizon)
        if key in stats and stats[key].get("mu") is not None and stats[key].get("sigma") is not None:
            mu, sigma = float(stats[key]["mu"]), float(stats[key]["sigma"])
            logging.info(f"_compute_sarm_reward_stats: loaded from cache ({key}): μ={mu:.6f}, σ={sigma:.6f}")
            return mu, sigma

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {root}/data")

    all_rewards: list[np.ndarray] = []
    for f in parquet_files:
        schema_cols = set(pq.read_schema(f).names)
        if progress_col not in schema_cols:
            continue
        t = pq.read_table(f, columns=["episode_index", "frame_index", progress_col])
        eps = np.asarray(t["episode_index"]).astype(np.int64).ravel()
        frames = np.asarray(t["frame_index"]).astype(np.int64).ravel()
        prog = np.asarray(t[progress_col]).astype(np.float64).ravel()
        # Sort by (episode, frame) to guarantee temporal order.
        order = np.lexsort((frames, eps))
        eps, prog = eps[order], prog[order]
        unique_eps = np.unique(eps)
        for ep in unique_eps:
            mask = eps == ep
            ep_prog = prog[mask]
            n = len(ep_prog)
            if n < action_horizon:
                continue
            n_chunks = n - action_horizon + 1
            rewards = ep_prog[action_horizon - 1:] - ep_prog[:n_chunks]
            all_rewards.append(rewards)

    if not all_rewards:
        logging.warning(f"No valid rewards computed for {repo_id}; returning defaults (0, 1)")
        return 0.0, 1.0

    rewards = np.concatenate(all_rewards)
    mu = float(rewards.mean())
    sigma = float(rewards.std())
    logging.info(
        f"_compute_sarm_reward_stats: {repo_id}, horizon={action_horizon}, "
        f"μ={mu:.6f}, σ={sigma:.6f}, n={len(rewards)}"
    )
    return mu, sigma


@dataclasses.dataclass(frozen=True)
class LeRobotYamDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms for the YAM bimanual robot dataset.
    
    The YAM data uses absolute joint positions:
    - State format: [left_6_joints, left_1_gripper, right_6_joints, right_1_gripper] = 14D
    - Three camera views: left exterior, right exterior, and top
    - Actions are absolute joint positions
    """
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "left_camera-images-rgb": "left_camera-images-rgb",
                        "right_camera-images-rgb": "right_camera-images-rgb", 
                        "top_camera-images-rgb": "top_camera-images-rgb",
                        "state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # Data transforms using YAM policy transforms
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[yam_policy.YamOutputs()],
        )

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )



@dataclasses.dataclass(frozen=True)
class AbcLayoutYamDataConfig(LeRobotYamDataConfig):
    """YAM transforms over the ABC training layout (abc's mcap-export format).

    Identical transform stack to LeRobotYamDataConfig; only the storage backend
    differs (abc_layout_dataset.AbcLayoutDataset instead of LeRobotDataset).
    NOTE the data conventions also follow abc: raw MCAP joint order (no flip) and
    commanded actions — norm stats and serving must use this config's assets.
    """

    # Optional station_type filter (see DataConfig.abc_station_types).
    station_types: tuple[str, ...] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            super().create(assets_dirs, model_config),
            abc_layout=True,
            abc_station_types=self.station_types,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYamRormDataConfig(DataConfigFactory):
    """
    YAM dataset with RORM reward weights for RABC / AWR training.

    Same as LeRobotYamDataConfig but also loads `rorm_velocity` (and optionally
    `rorm_q`) from the dataset and computes per-sample RABC weights.

    Modes (rabc_mode):
      "velocity_only"  — classic velocity-integrated weight (default)
      "multiplicative" — v_weight * q_norm  (requires rorm_q in dataset)
      "additive"       — 0.5 * (v_weight + q_norm)  (requires rorm_q in dataset)
      "sarm_progress_delta" — SARM-paper weighting from absolute progress.
          Computes reward = progress[-1] - progress[0] over the action horizon,
          then applies soft weighting with running-stats normalization.

    For Q-based modes, q_min/q_max are loaded from meta/rabc_stats.json unless
    explicitly provided via rabc_q_min / rabc_q_max.

    top_q_frac: if set, hard-filters to the top fraction of episodes by mean
    rorm_q score (no soft weighting). val_frac episodes are held out first
    using val_seed so all ablation configs share the same validation set.
    """

    default_prompt: str | None = None
    rabc_clip_min: float = 0.0
    rabc_clip_max: float = 1.0
    # Default flipped 2026-05-07 — finalaction gating beat mean integration
    # 11/12 vs 0/12 on real-world tshirt-fold eval; future configs inherit it.
    rabc_threshold: float | None = 0.50
    rabc_use_final_action_condition: bool = True
    # Filter zero-weighted chunks so the effective batch size matches the
    # nominal one (no GPU FLOPs spent on samples whose gradient is exactly
    # zero). Set False to revert to the legacy "weight-by-zero" loss-mask
    # behavior. See DataConfig.reject_zero_weighted_samples.
    rabc_reject_zero_weighted: bool = True
    # Implementation: "subset" precomputes valid indices from parquet once and
    # wraps the dataset in torch.utils.data.Subset (zero runtime overhead).
    # "rejection" does per-getitem retry (re-decodes video on each rejection).
    rabc_reject_zero_weighted_mode: str = "subset"
    # Q-based reweighting mode. One of {"velocity_only", "multiplicative", "additive", "q_threshold"}.
    rabc_mode: str = "velocity_only"
    # Explicit Q normalization range. Auto-loaded from rabc_stats.json when None.
    rabc_q_min: float | None = None
    rabc_q_max: float | None = None
    # q_threshold mode params. ``q_threshold_low`` is the velocity threshold
    # applied to the lowest-quality episode (q_norm=0); ``q_threshold_high`` to
    # the highest (q_norm=1). With defaults best→0, worst→1.
    q_threshold_low: float = 1.0
    q_threshold_high: float = 0.0
    q_threshold_shape: str = "linear"  # "linear" | "sigmoid"
    # For sigmoid shape: center is the q_norm at the midpoint of the transition.
    # Use rank-based percentiles, e.g. 0.95 for "top 5%", 0.90 for "top 10%".
    q_threshold_center: float = 0.5
    q_threshold_steepness: float = 10.0
    # How to collapse the per-frame velocity vector into a single window
    # weight before clipping. "mean" (default) is the historical behavior.
    # "min" picks the lowest velocity in the chunk — stricter, penalizes any
    # frame in the window dipping into anti-progress. "max" picks the highest.
    rabc_velocity_aggregator: str = "mean"
    # Extend the velocity window beyond action_horizon by this many frames so
    # gates like velocity_aggregator='mean_lookahead' can see 1s of motion
    # past the action chunk. Default 0 = no lookahead.
    rabc_lookahead_frames: int = 0
    # Non-linear post-process: weight → min(max(weight, 0)^p, clip_max). With
    # weight_power=2 + clip_max=1 you get min(weight^2, 1) — suppresses
    # medium-magnitude samples more than high-magnitude ones, indirectly
    # penalizing long episodes whose per-frame vel sits at ~0.5. Default 1.0
    # = linear (historical behavior).
    rabc_weight_power: float = 1.0
    # Scale factor applied to velocity before aggregation/clipping. Useful
    # when raw velocities are on a different scale than the [0, 1] range
    # expected by clipping/thresholding.
    rabc_velocity_scale: float = 1.0
    # ── sarm_progress_delta mode params ──
    # Pre-computed reward statistics for SARM soft weighting. When None and
    # mode is sarm_progress_delta, auto-computed by scanning the dataset.
    sarm_reward_mu: float | None = None
    sarm_reward_sigma: float | None = None
    # Hard prior-override threshold (κ in the SARM paper). Rewards above κ
    # get weight=1 unconditionally. Paper default: 0.01.
    sarm_kappa: float = 0.01
    # Column name for absolute progress values.
    sarm_progress_key: str = "sarm_dense_progress"
    # Hard Q-filter: keep only the top fraction of episodes by mean rorm_q.
    top_q_frac: float | None = None
    # Length filter: keep only the shortest fraction of episodes by frame count.
    top_shortest_frac: float | None = None
    # Deminf-curation baseline: path to a JSON file with an explicit list of
    # episode indices to keep (see assets/deminf_baselines/*.json). When set,
    # this overrides val/top_q/top_shortest selection — the listed episodes
    # are used directly as the training set (val_eps are still excluded if
    # val_frac > 0). Intended for "deminf-as-curation" baselines that match a
    # WARP-BC keep-fraction at matched training-sample count.
    deminf_keep_episodes_path: str | None = None
    val_frac: float = 0.0
    val_seed: int = 0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        is_sarm = self.rabc_mode == "sarm_progress_delta"
        use_q = self.rabc_mode not in ("velocity_only", "sarm_progress_delta")
        # Inspect the dataset parquet schema so we only request columns that exist.
        # Carry both canonical (repromo_*) and legacy (rorm_*) names through the
        # repack when present so ComputeRABCWeights can resolve whichever the
        # parquet uses; RepackTransform is a strict lookup and would KeyError on
        # a missing column.
        from lerobot.utils.constants import HF_LEROBOT_HOME
        import pyarrow.parquet as pq
        schema_cols: set[str] = set()
        if self.repo_id is not None:
            data_dir = HF_LEROBOT_HOME / self.repo_id / "data"
            try:
                first_parquet = next(data_dir.rglob("*.parquet"))
                schema_cols = set(pq.read_schema(first_parquet).names)
            except (StopIteration, FileNotFoundError):
                schema_cols = set()
        repack_keys = {
            "left_camera-images-rgb": "left_camera-images-rgb",
            "right_camera-images-rgb": "right_camera-images-rgb",
            "top_camera-images-rgb": "top_camera-images-rgb",
            "state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }

        if is_sarm:
            # SARM progress-delta mode: only need absolute progress column.
            if self.sarm_progress_key in schema_cols:
                repack_keys[self.sarm_progress_key] = self.sarm_progress_key
        else:
            # Pick the FIRST available velocity column (transform falls back
            # among them via the same precedence). Adding all that exist
            # forces lerobot to query every one per-frame, which crashes if
            # any legacy column (e.g. rorm_velocity) has null rows.
            for col in ("warp_rm_signed_magnitude", "repromo_signed_magnitude", "rorm_velocity", "sarm_dense_signed_magnitude"):
                if col in schema_cols:
                    repack_keys[col] = col
                    break
            if use_q:
                for col in ("repromo_quality", "rorm_q", "sarm_dense_quality"):
                    if col in schema_cols:
                        repack_keys[col] = col

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_keys)]
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        q_min, q_max = self.rabc_q_min, self.rabc_q_max
        if use_q and (q_min is None or q_max is None) and self.repo_id is not None:
            q_min, q_max = _load_rabc_q_range(self.repo_id)

        # SARM: auto-compute reward statistics if not explicitly provided.
        sarm_mu, sarm_sigma = self.sarm_reward_mu, self.sarm_reward_sigma
        if is_sarm and self.repo_id is not None:
            info_path = HF_LEROBOT_HOME / self.repo_id / "meta" / "info.json"
            if info_path.exists() and (sarm_mu is None or sarm_sigma is None):
                sarm_mu, sarm_sigma = _compute_sarm_reward_stats(
                    self.repo_id, model_config.action_horizon, self.sarm_progress_key,
                )

        rabc_inputs: list[_transforms.DataTransformFn] = [
            _transforms.ComputeRABCWeights(
                clip_min=self.rabc_clip_min,
                clip_max=self.rabc_clip_max,
                threshold=self.rabc_threshold,
                use_final_action_condition=self.rabc_use_final_action_condition,
                mode=self.rabc_mode,
                q_min=q_min,
                q_max=q_max,
                q_threshold_low=self.q_threshold_low,
                q_threshold_high=self.q_threshold_high,
                q_threshold_shape=self.q_threshold_shape,
                q_threshold_center=self.q_threshold_center,
                q_threshold_steepness=self.q_threshold_steepness,
                velocity_aggregator=self.rabc_velocity_aggregator,
                action_horizon=getattr(model_config, "action_horizon", 0),
                weight_power=self.rabc_weight_power,
                velocity_scale=self.rabc_velocity_scale,
                sarm_reward_mu=sarm_mu if sarm_mu is not None else 0.0,
                sarm_reward_sigma=sarm_sigma if sarm_sigma is not None else 1.0,
                sarm_kappa=self.sarm_kappa,
                sarm_progress_key=self.sarm_progress_key,
            ),
            yam_policy.YamInputs(action_dim=model_config.action_dim, model_type=model_config.model_type),
        ]

        data_transforms = _transforms.Group(
            inputs=rabc_inputs,
            outputs=[yam_policy.YamOutputs()],
        )

        # Stack velocity / quality / progress over action_horizon so RABC
        # transforms see a ``(horizon,)`` array.
        def _pick(*candidates: str) -> str | None:
            return next((c for c in candidates if c in schema_cols), None)

        if is_sarm:
            # Only need the progress column stacked over the horizon.
            progress_key = _pick(self.sarm_progress_key)
            extra_horizon_keys = (progress_key,) if progress_key else ()
        else:
            vel_key_for_horizon = _pick("warp_rm_signed_magnitude", "repromo_signed_magnitude", "rorm_velocity", "sarm_dense_signed_magnitude")
            q_key_for_horizon = _pick("repromo_quality", "rorm_q", "sarm_dense_quality") if use_q else None
            extra_horizon_keys = tuple(k for k in (vel_key_for_horizon, q_key_for_horizon) if k is not None)

        episodes: tuple[int, ...] | None = None
        val_episodes: tuple[int, ...] | None = None
        # Episode resolution is training-only; skip when the dataset isn't on disk
        # (e.g., during inference / serving) so the config still materializes.
        info_path = HF_LEROBOT_HOME / self.repo_id / "meta" / "info.json" if self.repo_id is not None else None
        if info_path is not None and info_path.exists():
            val_eps, non_val_eps = _split_val_episodes(self.repo_id, self.val_frac, self.val_seed)
            val_episodes = val_eps if val_eps else None
            if self.deminf_keep_episodes_path is not None:
                import json as _json
                with open(self.deminf_keep_episodes_path) as _f:
                    _blob = _json.load(_f)
                _kept = set(int(e) for e in _blob["episodes"])
                _val_set = set(val_eps or ())
                episodes = tuple(sorted(_kept - _val_set))
                logging.info(
                    f"[deminf_keep] {self.deminf_keep_episodes_path}: "
                    f"loaded {len(_kept)} episodes ({_blob.get('achieved_sample_frac', '?')} of samples); "
                    f"after val-exclusion training on {len(episodes)} eps."
                )
            elif self.top_q_frac is not None:
                episodes = tuple(_top_q_episodes(self.repo_id, self.top_q_frac, exclude_eps=val_eps))
            elif self.top_shortest_frac is not None:
                episodes = tuple(_shortest_episodes(self.repo_id, self.top_shortest_frac, exclude_eps=val_eps))
            elif non_val_eps:
                episodes = non_val_eps
        elif self.repo_id is not None:
            logging.info(
                f"Skipping episode split for {self.repo_id!r}: dataset not present at {info_path}. "
                f"Assumed inference-only context."
            )

        # SARM progress-delta mode reads progress (not velocity), so the
        # "subset" precompute path — which calls rabc.decide_weight() on
        # velocity windows — would silently mis-filter. Force per-getitem
        # "rejection" mode here so SARM configs don't need to set this
        # explicitly.
        reject_mode = self.rabc_reject_zero_weighted_mode
        if is_sarm and reject_mode != "rejection":
            logging.info(
                f"SARM mode ({self.rabc_mode}) requires per-getitem rejection; "
                f"overriding rabc_reject_zero_weighted_mode={reject_mode!r} → 'rejection'."
            )
            reject_mode = "rejection"

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            extra_horizon_keys=extra_horizon_keys,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            episodes=episodes,
            val_episodes=val_episodes,
            reject_zero_weighted_samples=self.rabc_reject_zero_weighted,
            reject_zero_weighted_mode=reject_mode,
            extra_horizon_lookahead_frames=self.rabc_lookahead_frames,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotScizorSidecarDataConfig(LeRobotYamRormDataConfig):
    """Paper-faithful SCIZOR gating for RA-BC training.

    Reads per-frame SCIZOR suboptimality scores from a sidecar parquet
    (``scizor_predictions.parquet``, output of SCIZOR's ``score_lerobot.py``)
    and gates each (obs_window, action_chunk) BC sample at its anchor frame
    using the paper threshold ε_s. The underlying LeRobot dataset is NOT
    modified — scores are joined at sample time via (episode_index,
    frame_index).

    See ``docs/scizor_sidecar_rabc.md`` for the methodology writeup.

    Fields:
      scizor_sidecar_path: path to ``scizor_predictions.parquet``.
      scizor_eps_s:        keep iff score <= ε_s. Paper default 0.58 (App. A.1).
      scizor_weight_mode:  "binary" (paper) or "continuous" (1 - score).
      scizor_score_column: which sidecar column to read. Default
                           ``scizor_score`` (paper-final: α-mixed with
                           trajectory mean). ``scizor_score_local`` is the
                           pre-mix per-frame score (for ablations).
    """

    scizor_sidecar_path: str = ""
    scizor_eps_s: float = 0.58
    scizor_weight_mode: str = "binary"
    scizor_score_column: str = "scizor_score"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.scizor_sidecar_path:
            raise ValueError(
                "LeRobotScizorSidecarDataConfig requires scizor_sidecar_path "
                "to point at a scizor_predictions.parquet file."
            )
        # Build a minimal repack that carries episode_index/frame_index (needed
        # by LoadScizorSidecar for its anchor lookup) and skips all RORM
        # velocity/quality columns (we read nothing from the dataset parquets).
        repack_keys = {
            "left_camera-images-rgb": "left_camera-images-rgb",
            "right_camera-images-rgb": "right_camera-images-rgb",
            "top_camera-images-rgb": "top_camera-images-rgb",
            "state": "state",
            "actions": "actions",
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_keys)]
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.LoadScizorSidecar(
                    sidecar_path=self.scizor_sidecar_path,
                    score_column=self.scizor_score_column,
                ),
                _transforms.ComputeRABCWeights(
                    mode="scizor_anchor",
                    threshold=self.scizor_eps_s,
                    scizor_weight_mode=self.scizor_weight_mode,
                    clip_min=0.0,
                    clip_max=1.0,
                    action_horizon=getattr(model_config, "action_horizon", 0),
                ),
                yam_policy.YamInputs(action_dim=model_config.action_dim, model_type=model_config.model_type),
            ],
            outputs=[yam_policy.YamOutputs()],
        )

        # Episode split (val/train) handling — copied from the parent so we
        # honor val_frac / val_seed / top_q_frac / top_shortest_frac the same
        # way and so val episodes stay comparable across ablation conditions.
        from lerobot.utils.constants import HF_LEROBOT_HOME
        episodes: tuple[int, ...] | None = None
        val_episodes: tuple[int, ...] | None = None
        info_path = HF_LEROBOT_HOME / self.repo_id / "meta" / "info.json" if self.repo_id is not None else None
        if info_path is not None and info_path.exists():
            val_eps, non_val_eps = _split_val_episodes(self.repo_id, self.val_frac, self.val_seed)
            val_episodes = val_eps if val_eps else None
            if self.top_q_frac is not None:
                episodes = tuple(_top_q_episodes(self.repo_id, self.top_q_frac, exclude_eps=val_eps))
            elif self.top_shortest_frac is not None:
                episodes = tuple(_shortest_episodes(self.repo_id, self.top_shortest_frac, exclude_eps=val_eps))
            elif non_val_eps:
                episodes = non_val_eps

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            extra_horizon_keys=(),  # sidecar reads one scalar per sample, no horizon fetch
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            episodes=episodes,
            val_episodes=val_episodes,
            reject_zero_weighted_samples=self.rabc_reject_zero_weighted,
            # Force subset mode — matches SCIZOR's dataset.filter(...) semantics
            # (dropped samples contribute 0 FLOPs, no rejection-sampling retry).
            reject_zero_weighted_mode="subset",
            extra_horizon_lookahead_frames=0,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotVelocitySidecarDataConfig(LeRobotYamRormDataConfig):
    """RABC velocity gating from a sidecar parquet instead of a dataset column.

    Same gate math as the parent (final-action / aggregator modes via
    ComputeRABCWeights), but the velocity comes from a standalone parquet with
    per-frame ``episode_index, frame_index, <column>`` rows — e.g. the
    ``frame_signals.parquet`` written by icrrt's RM scorers. The underlying
    LeRobot dataset is NOT modified or copied; comparing two reward models is
    a pure config swap. Local paths or s3:// URLs both work.

    The parent's velocity-column autodetection and horizon fetch are skipped;
    a LoadVelocitySidecar transform builds each sample's velocity window
    (lerobot-identical tail clamping), and the subset precompute reads the
    same sidecar (see data_loader.precompute_valid_indices).
    """

    velocity_sidecar_path: str = ""
    velocity_sidecar_column: str = "velocity"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.velocity_sidecar_path:
            raise ValueError(
                "LeRobotVelocitySidecarDataConfig requires velocity_sidecar_path "
                "to point at a parquet with episode_index/frame_index/velocity columns."
            )
        if self.rabc_mode != "velocity_only":
            raise ValueError("velocity sidecar supports rabc_mode='velocity_only' only")
        # Minimal repack: carry episode_index/frame_index for the sidecar
        # lookup; read no velocity/quality columns from the dataset parquets.
        repack_keys = {
            "left_camera-images-rgb": "left_camera-images-rgb",
            "right_camera-images-rgb": "right_camera-images-rgb",
            "top_camera-images-rgb": "top_camera-images-rgb",
            "state": "state",
            "actions": "actions",
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_keys)]
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.LoadVelocitySidecar(
                    sidecar_path=self.velocity_sidecar_path,
                    action_horizon=getattr(model_config, "action_horizon", 0),
                    lookahead_frames=self.rabc_lookahead_frames,
                    velocity_column=self.velocity_sidecar_column,
                ),
                _transforms.ComputeRABCWeights(
                    clip_min=self.rabc_clip_min,
                    clip_max=self.rabc_clip_max,
                    threshold=self.rabc_threshold,
                    use_final_action_condition=self.rabc_use_final_action_condition,
                    mode="velocity_only",
                    velocity_aggregator=self.rabc_velocity_aggregator,
                    action_horizon=getattr(model_config, "action_horizon", 0),
                    weight_power=self.rabc_weight_power,
                    velocity_scale=self.rabc_velocity_scale,
                ),
                yam_policy.YamInputs(action_dim=model_config.action_dim, model_type=model_config.model_type),
            ],
            outputs=[yam_policy.YamOutputs()],
        )

        # Episode split — same semantics as the parent.
        from lerobot.utils.constants import HF_LEROBOT_HOME
        episodes: tuple[int, ...] | None = None
        val_episodes: tuple[int, ...] | None = None
        info_path = HF_LEROBOT_HOME / self.repo_id / "meta" / "info.json" if self.repo_id is not None else None
        if info_path is not None and info_path.exists():
            val_eps, non_val_eps = _split_val_episodes(self.repo_id, self.val_frac, self.val_seed)
            val_episodes = val_eps if val_eps else None
            if self.top_q_frac is not None:
                episodes = tuple(_top_q_episodes(self.repo_id, self.top_q_frac, exclude_eps=val_eps))
            elif self.top_shortest_frac is not None:
                episodes = tuple(_shortest_episodes(self.repo_id, self.top_shortest_frac, exclude_eps=val_eps))
            elif non_val_eps:
                episodes = non_val_eps

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            extra_horizon_keys=(),  # velocity window comes from the sidecar
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            episodes=episodes,
            val_episodes=val_episodes,
            reject_zero_weighted_samples=self.rabc_reject_zero_weighted,
            reject_zero_weighted_mode=self.rabc_reject_zero_weighted_mode,
            extra_horizon_lookahead_frames=0,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "/home/justinyu/checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to run a validation-loss pass. 0 disables it (default).
    # Only abc-layout datasets carry a val/ split; other configs log a warning and skip.
    val_interval: int = 0
    # Number of full batches (of batch_size) evaluated per validation pass, drawn
    # evenly-spaced across the val split with a fixed rng — deterministic curve.
    num_val_batches: int = 8
    # How often (in steps) to save checkpoints.
    save_interval: int = 10000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 10000

    # If set, each checkpoint step is synced to this S3 path via ``aws s3 sync``.
    s3_checkpoint_path: str | None = None

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # ── RABC / AWR weighting ─────────────────────────────────────────────
    # When enabled, per-sample loss is weighted by the integrated RORM velocity
    # over the action horizon. Requires `rorm_velocity` in the dataset.
    rabc_enabled: bool = False
    # Clip range for the per-sample RABC weight after integration + normalization.
    rabc_clip_min: float = 0.0
    rabc_clip_max: float = 1.0
    # When True, normalize the weighted loss by sum of weights (SARM paper):
    #   L = Σ(w_i * ℓ_i) / (Σw_i + ε)  instead of  mean(w_i * ℓ_i)
    rabc_normalize_weights: bool = False

    # ── Online Reward Model RABC ─────────────────────────────────────────
    # Runs a PyTorch reward model (HybridRM) at training time to compute
    # per-sample weights. Mutually exclusive with rabc_enabled (pre-computed).
    online_rm_enabled: bool = False
    online_rm_weight_method: str = "float"  # "binary" or "float"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# WARP-BC / vanilla-BC task table for the 06_10 d405 deliveries + tshirt folding.
# short-name -> (base repo_id, default prompt, pi0 init checkpoint).
# box/bottles init from base pi0 pretrained; tshirt from the tshirt-folding pi0 base.
_WARPBC_TASKS = {
    "box":     ("fold_the_paper_box_d405_v021",                 "Fold the paper box",                 "gs://openpi-assets/checkpoints/pi0_base/params"),
    "bottles": ("put_the_plastic_bottles_in_the_bin_d405_v021", "Put the plastic bottles in the bin", "gs://openpi-assets/checkpoints/pi0_base/params"),
    "tshirt":  ("tshirt_folding_d405_v010_20260420_gop10",      "Folding tshirt pile and stacking",   "s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
}

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning XMI RBY configs.
    #
    TrainConfig(
        name="pi0_xmi_rby",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotXmiRbyDataConfig(
            repo_id="uynitsuj/xmi_bimanual_testing",
            default_prompt="testing",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_xmi_rby",
        model=pi0_fast.Pi0FASTConfig(action_dim=20, action_horizon=10, max_token_len=250),
        data=LeRobotXmiRbyDataConfig(
            repo_id="uynitsuj/xmi_bimanual_testing",
            default_prompt="testing",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_xmi_rby_low_mem_finetune",
        model=pi0_config.Pi0Config(action_horizon=10, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotXmiRbyDataConfig(
            repo_id="uynitsuj/xmi_bimanual_testing",
            default_prompt="testing",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(action_horizon=10, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # Fine-tuning YAM configs.
    TrainConfig(
        name="pi0_yam",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotYamDataConfig(
            repo_id="uynitsuj/yam_bimanual_load_dishes_absolute",
            default_prompt="Load dishes onto tabletop dishrack",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_yam_low_mem_finetune",
        model=pi0_config.Pi0Config(action_horizon=10, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotYamDataConfig(
            repo_id="uynitsuj/yam_bimanual_load_dishes_absolute",
            default_prompt="Load dishes onto tabletop dishrack",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    #
    # Siemens industrial packing: pi0.5 on the YAM dataset built from DataEngine job
    # 01a01dc5-bef9-7233-b409-4db2d832ac91 (716 episodes after the >=10s duration filter)
    # by scripts/yam_data/convert_xdof_mcap_job.py. Uses the validated bs128 speedup
    # recipe (docs/speedup): bs128/fsdp2 + OPENPI_REMAT_POLICY, cosine decay over 15k.
    #
    TrainConfig(
        name="pi05_siemens_industrial_packing_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYamDataConfig(
            repo_id="industrial_packing_yam",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    #
    # v2: the full 1435-episode pool (1346 yam_zed_0_61 + 89 yam_0_61/D405 stations),
    # ZED top cameras cropped to the D405 reference FOV (see docs/siemens_packing_runs.md).
    # Same recipe; 15k x bs128 is ~1 epoch of the 1.85M-frame pool (matches the
    # sample budget philosophy of the validated bottles recipe).
    #
    TrainConfig(
        name="pi05_siemens_industrial_packing_v2_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYamDataConfig(
            repo_id="industrial_packing_yam_v2",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    TrainConfig(
        name="pi05_siemens_packing_abcloader_v2_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224_v2",
            default_prompt="industrial packing",
            base_config=DataConfig(),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    # v3 (2026-08-29): +33 episodes over v2 — 30 new D405 (sz_44/20260828) and 3 ZED
    # on the new sz_04 station; 1468 episodes total after the >=10s filter.
    TrainConfig(
        name="pi05_siemens_packing_abcloader_v3_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224_v3",
            default_prompt="industrial packing",
            base_config=DataConfig(),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    TrainConfig(
        name="pi05_siemens_packing_abcloader_v3_zedonly_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224_v3",
            default_prompt="industrial packing",
            base_config=DataConfig(),
            station_types=("yam_zed_0_61",),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    # v3cc ablation (2026-08-29): identical episodes to v3 but the dataset is baked with
    # center-crop resize instead of letterbox padding (full pixel budget, narrower FOV).
    # First run with the live val pass (val_interval) — same 8 val episodes as v3, so
    # padded-vs-center-crop val curves compare directly.
    TrainConfig(
        name="pi05_siemens_packing_abcloader_v3cc_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224_v3cc",
            default_prompt="industrial packing",
            base_config=DataConfig(),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        val_interval=1_000,
    ),
    # ZED-only ablation: same dataset, station filter keeps the 1346 yam_zed_0_61
    # episodes (cropped ZED top) and drops the 89 D405 ones. Norm stats are computed
    # per-config, so this arm normalizes over its own subset.
    TrainConfig(
        name="pi05_siemens_packing_abcloader_v2_zedonly_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224_v2",
            default_prompt="industrial packing",
            base_config=DataConfig(),
            station_types=("yam_zed_0_61",),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    #
    # Same pi0.5 recipe but trained through abc's dataloader approach on the
    # MCAP-exported ABC layout (industrial_packing_abc224): random access into the
    # strict-CFR combined video + states_actions.bin, commanded actions, raw joint
    # order. Comparison arm against pi05_siemens_industrial_packing_bs128.
    #
    TrainConfig(
        name="pi05_siemens_packing_abcloader_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=AbcLayoutYamDataConfig(
            repo_id="industrial_packing_abc224",
            default_prompt="industrial packing",
            base_config=DataConfig(),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    #
    # Siemens "simple" D405-only job (DataEngine job 01a046a8-5ed0-7ea1-9064-f173a747688f,
    # 2106 episodes on sz_44, all yam_0_61/D405 stations). Plain LeRobot yam pipeline —
    # same conversion + transform stack as pi05_siemens_industrial_packing_bs128 (and the
    # bottles/warp-rm lineage): convert_xdof_mcap_job.py, 224 resize-with-pad, absolute
    # joint state=actions, flipped joint order. No ZED crop applies (no ZED stations).
    #
    TrainConfig(
        name="pi05_siemens_simple_d405_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        # LeRobotYamRormDataConfig for the val_frac split machinery only: this
        # dataset has no velocity column, so the repack stays vanilla and
        # ComputeRABCWeights no-ops (same shape as the Table-I vanilla BC arms).
        # val_frac = exactly 10 held-out episodes (k = round(frac * 2105)).
        data=LeRobotYamRormDataConfig(
            repo_id="siemens_simple_d405",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
            val_frac=10 / 2105,
            val_seed=0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        val_interval=1_000,
        project_name="siemens-industrial-packing",
    ),
    #
    # Two-stage curriculum (2026-09-01): 5k on the mixed industrial-packing v3 pool
    # (LeRobot rebuild of job_episodes_v3.csv, 1468 eps: ZED tops FOV-cropped to the
    # D405 reference + D405 stations), then 15k fine-tune on the refreshed simple-D405
    # job (siemens_simple_d405_v2, 2694 eps incl. the 590 new 20260901 episodes).
    #
    TrainConfig(
        name="pi05_siemens_packing_yam_v3_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="industrial_packing_yam_v3",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
            val_frac=10 / 1468,  # exactly 10 held-out episodes
            val_seed=0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=5_000),
        num_train_steps=5_000,
        save_interval=5_000,
        keep_period=5_000,
        val_interval=1_000,
        project_name="siemens-industrial-packing",
    ),
    TrainConfig(
        name="pi05_siemens_simple_d405_v2_ft_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="siemens_simple_d405_v2",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
            val_frac=10 / 2694,  # exactly 10 held-out episodes
            val_seed=0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        # Stage-1 output: the 5k industrial-packing-v3 checkpoint (final step 4999) on NFS.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/nfs_old/karim/siemens_tmp_ckpts/pi05_siemens_packing_yam_v3_bs128/siemens_packing_yam_v3_5k_20260901/4999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        val_interval=1_000,
        project_name="siemens-industrial-packing",
    ),
    # Center-crop ablation of the simple-D405 task (2026-09-02): same 2894-episode
    # pool as siemens_simple_d405_v2 but baked with --resize-mode center_crop
    # (640x480 -> center 480x480 -> 224, no letterbox — the bottles/sss45 look).
    # Serving must center-crop to match. NOTE: episode numbering is scrambled per
    # build, so this config's val episodes differ from v2's despite the same pool.
    TrainConfig(
        name="pi05_siemens_simple_d405_cc_bs128",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="siemens_simple_d405_cc",
            default_prompt="industrial packing",
            base_config=DataConfig(prompt_from_task=True),
            val_frac=10 / 2892,
            val_seed=0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        val_interval=1_000,
        project_name="siemens-industrial-packing",
    ),
    #
    # RABC / AWR weighted YAM tshirt folding configs.
    #
    TrainConfig(
        name="pi0_yam_tshirt_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # SCIZOR sidecar gating — paper-faithful (anchor-frame, ε_s=0.58, binary
    # weights). Reads scores from a separate parquet (no LeRobot mutation).
    # See docs/scizor_sidecar_rabc.md for methodology.
    TrainConfig(
        name="pi0_yam_tshirt_scizor_sidecar_110612",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotScizorSidecarDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_singlefold_gop10",
            default_prompt="Folding tshirt pile and stacking",
            scizor_sidecar_path="s3://xdof-internal-research/repromo/baselines/scizor/tshirt_singlefold_110612/scizor_predictions.parquet",
            scizor_eps_s=0.58,
            scizor_weight_mode="binary",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_scizor_sidecar_122320",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotScizorSidecarDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_singlefold_gop10",
            default_prompt="Folding tshirt pile and stacking",
            scizor_sidecar_path="s3://xdof-internal-research/repromo/baselines/scizor/tshirt_singlefold_122320/scizor_predictions.parquet",
            scizor_eps_s=0.58,
            scizor_weight_mode="binary",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # ── SCIZOR baseline on the SARM centered_with_d405 datasets ──────────
    # Same datasets as the SARM-RABC / deminf / WARP-BC baselines (under60s,
    # under90s), rescored by SCIZOR so all curation methods compare on an
    # identical demonstration pool. Sidecars live on S3 at
    # repromo/baselines/scizor/<repo_id>/scizor_predictions.parquet (same
    # filename convention as the singlefold sidecars); _load_scizor_sidecar
    # fetches the s3:// path to a local cache once in the main process.
    # ε_s=0.58, binary gate, no_rabc pretrained init, 60k steps — matched to
    # the other baselines.
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_scizor_sidecar_sarm_{tag}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotScizorSidecarDataConfig(
                repo_id=f"sarm_dense_and_sparse_centered_with_d405_{tag}_gop10",
                default_prompt="Folding tshirt pile and stacking",
                scizor_sidecar_path=f"s3://xdof-internal-research/repromo/baselines/scizor/sarm_dense_and_sparse_centered_with_d405_{tag}_gop10/scizor_predictions.parquet",
                scizor_eps_s=0.58,
                scizor_weight_mode="binary",
                base_config=DataConfig(prompt_from_task=True),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for tag in ("under60s", "under90s")
    ],
    TrainConfig(
        name="pi0_yam_tshirt_no_rabc_d405",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # hlm_tshirt_reward_select — counterpart to the d405 rabc/no_rabc pair on
    # the human-led-manipulation dataset. Reuses the same pi0 base ckpt as
    # the d405 configs. repo_id targets the gop10-reencoded variant for
    # faster random-access decode during training (run
    # `openpi/scripts/reencode_dense_keyframes.py` once if not yet on disk).
    TrainConfig(
        name="pi0_hlm_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_tshirt_reward_select_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_hlm_no_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_tshirt_reward_select_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # Merged hlm + d405-under-60s — counterpart to pi0_hlm_{rabc,no_rabc}
    # but on the 2427-episode merge that adds short d405 demos to the hlm
    # base. Same prompt + base ckpt; only repo_id changes. RABC variants
    # (uniform-shape vs piecewise-shape) come from re-injecting the
    # repromo_progress column with the appropriate RM checkpoint between
    # launches.
    TrainConfig(
        name="pi0_merged_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under60s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            rabc_use_final_action_condition=False,
            rabc_threshold=None,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_merged_no_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under60s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # Wider merge — hlm + d405 episodes ≤ 90s (4124 episodes total) instead
    # of the under-60s 2427. More d405 demonstrations added to the training
    # mix; downstream RABC + no-RABC pair to sweep whether the wider data
    # window helps under the fs=2 RM signal.
    TrainConfig(
        name="pi0_merged90_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=False,  # preserve original mean-aggregator behavior
            rabc_threshold=None,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Threshold sweep: same data + RM as Run-1 but mean-aggregated RABC weight
    # is zeroed when integrated < threshold; otherwise no upper cap. Three
    # thresholds tested.
    *[
        TrainConfig(
            name=f"pi0_merged90_rabc_thr{int(thr * 100):03d}_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="hlm_plus_d405_under90s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=False,
                rabc_threshold=thr,
                rabc_clip_max=float("inf"),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for thr in (0.50, 0.60, 0.75)
    ],
    # hlm+all-d405 (singlefold-pruned, no time filter, 7679 ep). Same data +
    # injected RM as Run-4 (pi0_merged_singlefold_rabc); final-action gating
    # with two thresholds.
    *[
        TrainConfig(
            name=f"pi0_merged_singlefold_rabc_finalaction_thr{int(thr * 100):03d}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="hlm_plus_d405_singlefold_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=thr,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for thr in (0.50, 0.75)
    ],
    # 120s-cap vanilla BC baseline (no rabc). Counterpart to pi0_merged_no_rabc
    # (under60s) and pi0_merged90_no_rabc (under90s).
    TrainConfig(
        name="pi0_merged120_no_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under120s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # sarm dataset variant: fs2_sss45-RM-injected dense+sparse subset, strict
    # finalaction thr=1.00, no-max clip (raw RM magnitude flows to loss).
    TrainConfig(
        name="pi0_sarm_dense_sparse_rabc_finalaction_thr100_nomax",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_only_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_sarm_rabc_dense_progress_20260510",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_only_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=90_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # mean2s / mean1s_offset gates on merged90/120: aggregate vel over the
    # action chunk + 1s after it (60-frame window = 2s @ 30fps), or only the
    # 1s lookahead portion. Tests whether the model wants to be told about
    # what comes AFTER the action it's predicting, not just within it.
    # Defaults: thr=0.75 NOMAX (kept-weight = raw mean; no upper cap).
    *[
        TrainConfig(
            name=f"pi0_merged{cap}_rabc_{aggname}_thr{int(thr*100):03d}_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"hlm_plus_d405_under{cap}s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=False,
                rabc_threshold=thr,
                rabc_clip_max=float("inf"),
                rabc_velocity_aggregator=agg,
                rabc_lookahead_frames=30,  # 1s lookahead at fps=30
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for cap in (90, 120)
        for agg, aggname in (("mean", "mean2s"), ("mean_lookahead", "mean1s_offset"))
        for thr in (0.75, 1.00)
    ],
    # min(weight^2, 1) variants for merged90 / merged120 × {finalaction,
    # mean2s, mean1s_offset}. Same as the nothr_nomax recipe except
    # weight_power=2 and clip_max=1.0 (capped at 1 by construction). Suppresses
    # medium-magnitude samples (which concentrate in long episodes) by ~12%
    # more than the linear version per simulation.
    *[
        TrainConfig(
            name=f"pi0_merged{cap}_rabc_{aggname}_sqclip",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"hlm_plus_d405_under{cap}s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=use_fac,
                rabc_threshold=None,
                rabc_clip_max=1.0,
                rabc_velocity_aggregator=agg,
                rabc_lookahead_frames=lookahead,
                rabc_weight_power=2.0,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for cap in (90, 120)
        for use_fac, agg, aggname, lookahead in (
            (True, "mean", "finalaction", 0),
            (False, "mean", "mean2s", 30),
            (False, "mean_lookahead", "mean1s_offset", 30),
        )
    ],
    # sarm dataset × {finalaction, mean2s, mean1s_offset}, all thr=None NOMAX.
    # Re-uses the d405-short25 RM injection (separately scored via launch_score
    # on sarm — uses repromo_signed_magnitude / repromo_quality columns).
    *[
        TrainConfig(
            name=f"pi0_sarm_dense_sparse_rabc_{aggname}_nothr_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="sarm_dense_and_sparse_only_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=use_fac,
                rabc_threshold=None,
                rabc_clip_max=float("inf"),
                rabc_velocity_aggregator=agg,
                rabc_lookahead_frames=lookahead,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for use_fac, agg, aggname, lookahead in (
            (True, "mean", "finalaction", 0),
            (False, "mean", "mean2s", 30),
            (False, "mean_lookahead", "mean1s_offset", 30),
        )
    ],
    # No-threshold + no-max-cap variants for merged60/90/120 × {finalaction,
    # mean2s, mean1s_offset}. With threshold=None and clip_max=inf the weight
    # is raw vel passed through (clipped at clip_min=0 floor so negative
    # motion → 0 → filtered by subset). Tests whether removing the threshold
    # entirely and just letting magnitude flow is better than gating.
    *[
        TrainConfig(
            name=f"pi0_merged{cap}_rabc_{aggname}_nothr_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"hlm_plus_d405_under{cap}s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=use_fac,
                rabc_threshold=None,
                rabc_clip_max=float("inf"),
                rabc_velocity_aggregator=agg,
                rabc_lookahead_frames=lookahead,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for cap in (60, 90, 120)
        for use_fac, agg, aggname, lookahead in (
            (True, "mean", "finalaction", 0),  # vel[-1] passthrough
            (False, "mean", "mean2s", 30),     # mean over 60-frame window
            (False, "mean_lookahead", "mean1s_offset", 30),  # mean over lookahead 30 frames
        )
    ],
    # 60s-cap variant: thr=1.00 strict finalaction on hlm + d405<60s.
    TrainConfig(
        name="pi0_merged60_rabc_finalaction_thr100",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under60s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # 120s-cap variants: hlm + d405<120s, d405-short25-RM injected, final-action
    # gating. thr=0.75 keeps ~52% of frames; thr=1.00 keeps ~22% (long-episode
    # frames almost entirely drop). thr=0.50 retired 2026-05-08 — the strict
    # gate without cond_accel kept ~80% which under-filtered long episodes
    # relative to RM-prediction shape.
    *[
        TrainConfig(
            name=f"pi0_merged120_rabc_finalaction_thr{int(thr * 100):03d}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="hlm_plus_d405_under120s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=thr,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for thr in (0.75, 1.00)
    ],
    # No-max-cap variants of the 120s strict-gate trains. clip_max=inf lets
    # the kept-sample weight reflect raw RM magnitude (vel can be > 1.0)
    # rather than saturating at 1.0. Tests whether the cap was suppressing
    # the gradient signal on the most-confident-progress frames.
    *[
        TrainConfig(
            name=f"pi0_merged120_rabc_finalaction_thr{int(thr * 100):03d}_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="hlm_plus_d405_under120s_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=thr,
                rabc_clip_max=float("inf"),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for thr in (0.75, 1.00)
    ],
    # Run-5b variant: same as finalaction, but multiply weight by q_norm
    # (min-max from injected pinned Q). Q comes from injected repromo_quality.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction_mult",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=0.50,
            rabc_mode="multiplicative",
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Run-5 variant: pi0 on merged90 with d405-short25-RM, RABC weight
    # additionally gated by the final-action condition. Same data + RM as
    # Run-1 (pi0_merged90_rabc) — only the transform flag differs.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=0.50,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # NOMAX (clip_max=inf) sibling of thr=1.0 strict on merged90: keeps the
    # same ~34% of frames but lets kept-weight magnitude pass through raw
    # rather than saturating at 1.0.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction_thr100_nomax",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # IID-RM ablation twin: same recipe as pi0_merged90_rabc_finalaction_thr100_nomax,
    # but repo_id points at the duplicated merged90 dataset whose
    # `repromo_signed_magnitude` was injected with the IID-trained RM
    # (repromo_full_tshirt_folding_d405_v010_20260420_shortest25_win32_iid_15k)
    # rather than the canonical AR(1)-trained RM. Used to isolate the
    # contribution of the AR(1) speed-process correlation in the RM training
    # sampler on downstream policy quality.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction_thr100_nomax_iidrm",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_iidrm_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Mean-over-chunk sibling of the above: same dataset and thr/nomax, but
    # uses the default velocity_aggregator="mean" path instead of the final-
    # action gate. Keep iff mean(vel[t:t+H]) > 1.0; kept weight = clip(mean,
    # None, inf) = mean. clip_max=inf so high-velocity stretches retain their
    # magnitude rather than saturating at 1.0.
    TrainConfig(
        name="pi0_merged90_rabc_mean_thr100_nomax",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=False,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # NOMAX sibling for merged60 thr=1.0 strict (same dataset as
    # pi0_merged60_rabc_finalaction_thr100 but no upper cap).
    TrainConfig(
        name="pi0_merged60_rabc_finalaction_thr100_nomax",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under60s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # thr=1.0 strict variant — the most aggressive filter; long-episode frames
    # are nearly all dropped. Expected keep ~34% on under90s.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction_thr100",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # thr=0.75 strict variant of pi0_merged90_rabc_finalaction. Drops cond_accel
    # rescue (handled in transforms.py) and raises threshold so long-episode
    # frames with mid-range velocity are filtered out. Expected keep ~61%.
    TrainConfig(
        name="pi0_merged90_rabc_finalaction_thr075",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=0.75,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_merged90_no_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # Run-2 variant: pi0 on merged90 forked with truear-trained d405 RM
    # (--sampler truear). Forked S3 prefix because 171/172 spent 3h+ in
    # capacity-PENDING — the in-place re-inject race that "non-spot makes
    # negligible" reopens when 171/172 haven't even synced yet. Forking
    # restores correctness; the fork videos are already on S3.
    TrainConfig(
        name="pi0_merged90_truearrm_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_truearrm_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=False,
            rabc_threshold=None,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Run-3 variant: pi0 on merged90 with d405-short25-RM sidecar but window
    # weight = MIN(velocity) instead of mean. Stricter — any anti-progress
    # frame in the chunk drives the weight down.
    TrainConfig(
        name="pi0_merged90_rabc_minwin",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_velocity_aggregator="min",
            rabc_use_final_action_condition=False,
            rabc_threshold=None,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Run-4 dataset: hlm + singlefold-d405 (no time filter, multi-fold pruned).
    # 7679 episodes — adds the full surviving d405 single-fold corpus to hlm.
    TrainConfig(
        name="pi0_merged_singlefold_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="hlm_plus_d405_singlefold_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=False,
            rabc_threshold=None,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # Hard Q-filter ablations — train on top-N% episodes by rorm_q, no soft weighting.
    # Direct counterpart to the multiplicative/additive RABC runs for A/B comparison.
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_topq{int(frac * 100):02d}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="tshirt_folding_d405_v010_20260420_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                top_q_frac=frac,
                val_frac=0.1,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,     
            rabc_enabled=False,
        )
        for frac in (0.10, 0.25, 0.50, 0.75)
    ],
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_topq{int(frac * 100):02d}_lora",
            model=pi0_config.Pi0Config(action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotYamRormDataConfig(
                repo_id="tshirt_folding_d405_v010_20260420_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                top_q_frac=frac,
                val_frac=0.1,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,     
            rabc_enabled=False,
            freeze_filter=pi0_config.Pi0Config(
                action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
        )
        for frac in (0.10, 0.25, 0.50, 0.75)
    ],
    # Shortest-episode filter ablations — train on shortest N% of episodes by frame count.
    # Shortest demos tend to be cleaner/more confident executions.
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_shortest_{int(frac * 100):02d}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id="tshirt_folding_d405_v010_20260420_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                top_shortest_frac=frac,
                val_frac=0.1,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,     
            rabc_enabled=False,
        )
        for frac in (0.10, 0.20, 0.50, 0.75)
    ],
    # Q-weighted RABC — multiplicative: w = v_weight * q_norm
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_mult",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="multiplicative",
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # q_threshold (linear): per-episode velocity threshold = 1 - q_norm.
    # Best episodes pass anything; worst require vel >= 1.0.
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_linear",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            q_threshold_shape="linear",
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # q_threshold (sigmoid centered at top 5% — q_norm rank=0.95).
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_sig_top5",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            q_threshold_shape="sigmoid",
            q_threshold_center=0.95,
            q_threshold_steepness=25.0,
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # q_threshold (sigmoid centered at top 10% — q_norm rank=0.90).
    # Sharp transition: top ~10% pass freely, the rest require near-1.0 vel.
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_sig_top10",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            q_threshold_shape="sigmoid",
            q_threshold_center=0.90,
            q_threshold_steepness=20.0,
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # q_threshold (sigmoid centered at top 25% — q_norm rank=0.75).
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_sig_top25",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            q_threshold_shape="sigmoid",
            q_threshold_center=0.75,
            q_threshold_steepness=15.0,
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
    ),
    # q_threshold + final_action variants: q-derived threshold replaces the
    # static threshold in the final-action keep rule. Sample is kept iff
    # vel[-1] is positive-and-accelerating OR vel[-1] > q-derived threshold.
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_linear_fa",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            rabc_use_final_action_condition=True,
            q_threshold_shape="linear",
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_sig_top5_fa",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            rabc_use_final_action_condition=True,
            q_threshold_shape="sigmoid",
            q_threshold_center=0.95,
            q_threshold_steepness=25.0,
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_thresh_sig_top10_fa",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="q_threshold",
            rabc_use_final_action_condition=True,
            q_threshold_shape="sigmoid",
            q_threshold_center=0.90,
            q_threshold_steepness=20.0,
            q_threshold_low=1.0,
            q_threshold_high=0.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
    ),
    # No-clip RABC — disable both clip bounds so v_weight passes through unmodified.
    TrainConfig(
        name="pi0_yam_tshirt_rabc_no_clip",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_clip_min=float("-inf"),
            rabc_clip_max=float("inf"),
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # use_final_action_condition: keep samples by final-velocity rule, threshold=0.80.
    TrainConfig(
        name="pi0_yam_tshirt_rabc_use_final_action_cond",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=0.50,
            rabc_clip_max=1.0,
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # Combined: top-10% Q-filter + multiplicative Q-weighted RABC.
    TrainConfig(
        name="pi0_yam_tshirt_topq10_rabc_q_mult",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            top_q_frac=0.10,
            rabc_mode="multiplicative",
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # Q-weighted RABC — additive: w = 0.5 * (v_weight + q_norm)
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_add",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="additive",
            val_frac=0.1,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,     
        rabc_enabled=True,
    ),
    # LoRA variants of the new ablation configs (topq, shortest, q_mult, q_add).
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_shortest_{int(frac * 100):02d}_lora",
            model=pi0_config.Pi0Config(action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotYamRormDataConfig(
                repo_id="tshirt_folding_d405_v010_20260420_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                top_shortest_frac=frac,
                val_frac=0.1,
            ),
            batch_size=8,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=20_000,
            keep_period=40_000,
            rabc_enabled=False,
            freeze_filter=pi0_config.Pi0Config(
                action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
        )
        for frac in (0.10, 0.20, 0.50, 0.75)
    ],
    TrainConfig(
        name="pi0_yam_tshirt_rabc_false",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="velocity_only",
            val_frac=0.1,
        ),
        batch_size=8,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_rabc_only",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="velocity_only",
            val_frac=0.1,
        ),
        batch_size=8,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_mult_lora",
        model=pi0_config.Pi0Config(action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="multiplicative",
            val_frac=0.1,
        ),
        batch_size=8,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
        freeze_filter=pi0_config.Pi0Config(
            action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_yam_tshirt_rabc_q_add_lora",
        model=pi0_config.Pi0Config(action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotYamRormDataConfig(
            repo_id="tshirt_folding_d405_v010_20260420_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="additive",
            val_frac=0.1,
        ),
        batch_size=8,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=40_000,
        rabc_enabled=True,
        freeze_filter=pi0_config.Pi0Config(
            action_horizon=30, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # ── Online Reward Model RABC (David Chen's HybridRM) ─────────────────
    TrainConfig(
        name="pi0_yam_tshirt_online_rm_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamDataConfig(
            repo_id="Qianzhong-Chen/tshirt_folding_10h_hlm_yam_white_0810",
            default_prompt="fold the tshirt",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        online_rm_enabled=True,
        online_rm_weight_method="float",
    ),
    TrainConfig(
        name="pi0_yam_tshirt_online_rm_rabc_binary",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamDataConfig(
            repo_id="Qianzhong-Chen/tshirt_folding_10h_hlm_yam_white_0810",
            default_prompt="fold the tshirt",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        online_rm_enabled=True,
        online_rm_weight_method="binary",
    ),
    # ── SARM vanilla BC (no weighting, baseline comparison) ─────────────
    TrainConfig(
        name="pi0_yam_tshirt_sarm_bc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamDataConfig(
            repo_id="sarm_dense_and_sparse_only_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=90_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    # ── SARM cached RABC sparse head ────────────────────────────────────
    TrainConfig(
        name="pi0_yam_tshirt_sarm_rabc_sparse",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_only_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="sarm_progress_delta",
            sarm_kappa=0.01,
            sarm_progress_key="sarm_sparse_progress",
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=90_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
        rabc_normalize_weights=True,
    ),
    # ── SARM cached RABC sparse head — centered+d405 under90s dataset ───
    TrainConfig(
        name="pi0_yam_tshirt_sarm_rabc_sparse_under90s",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_centered_with_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="sarm_progress_delta",
            sarm_kappa=0.01,
            sarm_progress_key="sarm_sparse_progress",
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=20_000,
        keep_period=20_000,
        rabc_enabled=True,
        rabc_normalize_weights=True,
    ),
    # ── DemInf-curation BASELINES on the SARM centered_with_d405 datasets ─
    # Plain BC (no RABC); deminf is the sole curation signal. Episode list is
    # the top-X% of episodes by training-sample count, ranked by deminf
    # `ep_idx` log-density score (higher = closer to demo distribution).
    # The keep-fractions (45.6% on under60s, 33.7% on under90s) match the
    # effective training-sample budgets of the WARP-BC twins, so the head-to-
    # head answers "is episode-level deminf curation as good as chunk-level
    # WARP-BC at matched data budget?". See assets/deminf_baselines/*.json
    # for the exact episode lists + selection metadata.
    TrainConfig(
        name="pi0_sarm_under60s_no_rabc_deminf_top456",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_centered_with_d405_under60s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_reject_zero_weighted=False,
            deminf_keep_episodes_path="assets/deminf_baselines/sarm_dense_and_sparse_centered_with_d405_under60s_gop10_top456.json",
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    TrainConfig(
        name="pi0_sarm_under90s_no_rabc_deminf_top337",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_centered_with_d405_under90s_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_reject_zero_weighted=False,
            deminf_keep_episodes_path="assets/deminf_baselines/sarm_dense_and_sparse_centered_with_d405_under90s_gop10_top337.json",
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=False,
    ),
    # ── SARM cached RABC (pre-computed dense progress predictions) ──────
    TrainConfig(
        name="pi0_yam_tshirt_sarm_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sarm_dense_and_sparse_only_gop10",
            default_prompt="Folding tshirt pile and stacking",
            base_config=DataConfig(prompt_from_task=True),
            rabc_mode="sarm_progress_delta",
            sarm_kappa=0.01,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=60_000,
        save_interval=30_000,
        keep_period=30_000,
        rabc_enabled=True,
        rabc_normalize_weights=True,
    ),
    # ── SARM sparse — under60s + full variants of the centered_with_d405 set
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_sarm_rabc_sparse_{tag}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"sarm_dense_and_sparse_centered_with_d405_{tag}_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_mode="sarm_progress_delta",
                sarm_kappa=0.01,
                sarm_progress_key="sarm_sparse_progress",
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=20_000,
            keep_period=20_000,
            rabc_enabled=True,
            rabc_normalize_weights=True,
        )
        for tag in ("full", "under60s")
    ],
    # ── d405-short25 RM (repromo_signed_magnitude) finalaction thr=1.0 NOMAX
    #    on the same 3 centered_with_d405 datasets — A/B vs the SARM method.
    *[
        TrainConfig(
            name=f"pi0_yam_tshirt_d405short25_finalaction_thr100_nomax_{tag}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"sarm_dense_and_sparse_centered_with_d405_{tag}_gop10",
                default_prompt="Folding tshirt pile and stacking",
                base_config=DataConfig(prompt_from_task=True),
                rabc_mode="velocity_only",
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
                rabc_clip_max=float("inf"),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://xdof-internal-research/model_ckpts/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999/params"),
            num_train_steps=60_000,
            save_interval=20_000,
            keep_period=20_000,
            rabc_enabled=True,
        )
        for tag in ("full", "under90s", "under60s")
    ],
    # Per-task pi0 finetunes on the 5 sim datasets with RABC final-action
    # gating (keep iff repromo_velocity[-1] > threshold). repromo_velocity /
    # repromo_quality columns must be written into each LeRobot dataset
    # offline by the corresponding best_model_*_no_abs.pt repromo checkpoint
    # before training. HF_LEROBOT_HOME must point at /home/karimelrafi/datasets.
    *[
        TrainConfig(
            name=f"pi0_sim_{short}_rabc_finalaction_thr100",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            save_interval=10_000,
            keep_period=10_000,
            rabc_enabled=True,
        )
        for short, repo_id, prompt in (
            ("hang_mug",       "sim_hang_the_mug_on_the_mug_rack_gop10",       "Hang the mug on the mug rack"),
            ("load_plates",    "sim_load_the_plates_into_the_dish_rack_gop10", "Load the plates into the dish rack"),
            ("put_bottles",    "sim_put_the_plastic_bottles_in_the_bin_gop10", "Put the plastic bottles in the bin"),
            ("sweep_paper",    "sim_sweep_away_paper_scraps_from_the_table",   "Sweep away paper scraps from the table"),
            ("throw_bottles",  "sim_throw_plastic_bottles_in_bin_gop10",       "Throw the plastic bottles in the bin"),
        )
    ],
    # Vanilla-BC counterparts to the 5 sim rabc_finalaction_thr100 configs above.
    # rabc_enabled=False bypasses both the loss reweighting and the subset filter
    # (data_loader.py forces reject_zero_weighted_samples=False), so every sample
    # trains at weight=1.0. Same dataset / init / step budget / save schedule as
    # the rabc variants — only the gate is removed, for a clean ablation.
    *[
        TrainConfig(
            name=f"pi0_sim_{short}_no_rabc",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            save_interval=10_000,
            keep_period=10_000,
            rabc_enabled=False,
        )
        for short, repo_id, prompt in (
            ("hang_mug",       "sim_hang_the_mug_on_the_mug_rack_gop10",       "Hang the mug on the mug rack"),
            ("load_plates",    "sim_load_the_plates_into_the_dish_rack_gop10", "Load the plates into the dish rack"),
            ("put_bottles",    "sim_put_the_plastic_bottles_in_the_bin_gop10", "Put the plastic bottles in the bin"),
            ("sweep_paper",    "sim_sweep_away_paper_scraps_from_the_table",   "Sweep away paper scraps from the table"),
            ("throw_bottles",  "sim_throw_plastic_bottles_in_bin_gop10",       "Throw the plastic bottles in the bin"),
        )
    ],
    # ── Vanilla BC (no_rabc) for the canonical 30hz sim datasets
    #    (s3://xdof-internal-research/repromo/datasets/sim_<task>_30hz_gop10).
    #    Counterparts to the sim RABC runs (same pi0_base init / 30k steps /
    #    save schedule; only the reward gate removed). DISTINCT from the
    #    pi0_sim_<task>_no_rabc configs above, which point at the STALE 15hz
    #    `_gop10` datasets — these use the 30hz canonical data so they're a
    #    clean ablation against the 30hz RABC runs.
    *[
        TrainConfig(
            name=f"pi0_sim_{short}_no_rabc_30hz",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            # Extended 30k -> 60k (resumed from the 29999 ckpts). decay_steps must
            # match the 60k horizon (default CosineDecaySchedule decay_steps=30k is
            # fixed, not tied to num_train_steps) so the cosine spans the full run
            # rather than sitting at the 2.5e-6 floor for steps 30k-60k. Resuming the
            # 30k-decayed ckpts gives a warm-restart LR bump at step 30k (intended).
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
            num_train_steps=60_000,
            save_interval=10_000,
            keep_period=10_000,
            rabc_enabled=False,
        )
        for short, repo_id, prompt in (
            ("load_plates",   "sim_load_the_plates_into_the_dish_rack_30hz_gop10",    "Load the plates into the dish rack"),
            ("put_bottles",   "sim_put_the_plastic_bottles_in_the_bin_30hz_gop10",    "Put the plastic bottles in the bin"),
            ("throw_bottles", "sim_throw_plastic_bottles_in_bin_30hz_gop10",          "Throw the plastic bottles in the bin"),
            ("turn_mug",      "sim_turn_the_mug_right_side_up_30hz_gop10",            "Turn the mug right side up"),
            ("sweep_paper",   "sim_sweep_away_paper_scraps_from_the_table_30hz_gop10","Sweep away paper scraps from the table"),
        )
    ],
    # ── Per-task RABC trains on the canonical 30hz sim datasets, gating on the
    #    inline `repromo_signed_magnitude` column injected by the new sss45
    #    WARP-RM checkpoints (collaborator-uploaded 2026-06-25 to
    #    s3://xdof-internal-research/repromo/datasets/mjgl_sim_30hz/<slug>/ and
    #    promoted to s3://xdof-internal-research/lerobot/<slug>/).
    #    Recipe: final-action gate at thr=1.0 (keep iff vel[-1] > 1.0), default
    #    rabc_clip_max=1.0 → binary keep semantics (matches the original
    #    pi0_sim_*_rabc_finalaction_thr100 runs, not the box/bottles/tshirt
    #    `warpbc_sss{n}` continuous-reweight variant). top_shortest_frac unset.
    #    Step budget 60k + cosine decay_steps=60k mirrors pi0_sim_*_no_rabc_30hz
    #    so the RABC-vs-vanilla-BC comparison is apples-to-apples.
    *[
        TrainConfig(
            name=f"pi0_sim_{short}_rabc_30hz",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
            num_train_steps=60_000,
            save_interval=10_000,
            keep_period=10_000,
            rabc_enabled=True,
        )
        for short, repo_id, prompt in (
            ("hang_mug",      "sim_hang_the_mug_on_the_mug_rack_30hz_gop10",          "Hang the mug on the mug rack"),
            ("load_plates",   "sim_load_the_plates_into_the_dish_rack_30hz_gop10",    "Load the plates into the dish rack"),
            ("put_bottles",   "sim_put_the_plastic_bottles_in_the_bin_30hz_gop10",    "Put the plastic bottles in the bin"),
            ("sweep_paper",   "sim_sweep_away_paper_scraps_from_the_table_30hz_gop10","Sweep away paper scraps from the table"),
            ("throw_bottles", "sim_throw_plastic_bottles_in_bin_30hz_gop10",          "Throw the plastic bottles in the bin"),
            ("turn_mug",      "sim_turn_the_mug_right_side_up_30hz_gop10",            "Turn the mug right side up"),
        )
    ],
    # ── put_bottles MJWARP RABC matrix (mirrors the DiT-XL mjwarp Table-A arms:
    #    perobj RM, overall RM, + no_rabc baseline). Trains on the mjwarp
    #    re-rendered 30hz dataset (2438 eps, 14-dim state/actions) with the
    #    per-frame `warp_rm_signed_magnitude` column injected from the DiT
    #    velocity_rm_{perobj,overall}.bin sidecars (an exact copy of the RM's
    #    per-frame signal — verified bit-for-bit against abc/score_to_sidecar).
    #    Two videos-on-S3 dataset copies:
    #      sim_put_bottles_mjwarp_rmperobj  <- velocity_rm_perobj.bin
    #      sim_put_bottles_mjwarp_rmoverall <- velocity_rm_overall.bin
    #    Recipe = pi0_sim_put_bottles_rabc_30hz (final-action gate thr=1.0,
    #    default clip_max=1.0 binary-keep, ah=30, pi0_base init, 60k cosine,
    #    bs=32). Baseline reuses the perobj copy with rabc_enabled=False (the
    #    reward column is ignored when RABC is off — RM-agnostic vanilla BC).
    TrainConfig(
        name="pi0_put_bottles_mjwarp_rabc_perobj",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sim_put_bottles_mjwarp_rmperobj",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_put_bottles_mjwarp_rabc_overall",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sim_put_bottles_mjwarp_rmoverall",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_put_bottles_mjwarp_no_rabc",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="sim_put_bottles_mjwarp_rmperobj",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
        ),
        batch_size=32,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        rabc_enabled=False,
    ),
    # ── NOMAX siblings of pi0_sim_<task>_rabc_30hz. Identical recipe except
    #    rabc_clip_max=inf — chunks above thr=1.0 keep their raw vel[-1]
    #    weight magnitude rather than saturating at 1.0. clip_min left at
    #    default 0.0 (dead under threshold-gating; see transforms.py:332,
    #    `np.clip(final_vel, None, self.clip_max)` ignores clip_min).
    *[
        TrainConfig(
            name=f"pi0_sim_{short}_rabc_30hz_nomax",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo_id,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
                rabc_clip_max=float("inf"),
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
            num_train_steps=60_000,
            save_interval=10_000,
            keep_period=10_000,
            rabc_enabled=True,
        )
        for short, repo_id, prompt in (
            ("hang_mug",      "sim_hang_the_mug_on_the_mug_rack_30hz_gop10",          "Hang the mug on the mug rack"),
            ("load_plates",   "sim_load_the_plates_into_the_dish_rack_30hz_gop10",    "Load the plates into the dish rack"),
            ("put_bottles",   "sim_put_the_plastic_bottles_in_the_bin_30hz_gop10",    "Put the plastic bottles in the bin"),
            ("sweep_paper",   "sim_sweep_away_paper_scraps_from_the_table_30hz_gop10","Sweep away paper scraps from the table"),
            ("throw_bottles", "sim_throw_plastic_bottles_in_bin_30hz_gop10",          "Throw the plastic bottles in the bin"),
            ("turn_mug",      "sim_turn_the_mug_right_side_up_30hz_gop10",            "Turn the mug right side up"),
        )
    ],
    # ── WARP-BC (reward-aligned BC) + vanilla BC: 06_10 d405 deliveries
    #    (box, bottles) + tshirt folding. 9 WARP-BC (3 tasks × {sss15,sss30,sss45}
    #    RM strides) + 3 vanilla BC = 12 runs.
    #    WARP-BC recipe (mirrors pi0_*_rabc_finalaction_thr100_nomax):
    #      τ=1.0, clip_max=inf (continuous reweight everything above τ),
    #      finalaction (terminal v_end), velocity_only, shortest 50% split.
    #      Reads the per-stride scored copy's `warp_rm_signed_magnitude` column.
    #    Vanilla BC: same shortest-50% split, rabc disabled (base dataset; no
    #      velocity column needed — ComputeRABCWeights no-ops when absent).
    #    PREREQUISITE: the per-stride scored copies <repo>_sss{15,30,45} are
    #    created by warprm2/scripts/score_and_inject_warpbc.py (RM dense-inference
    #    → inject velocity column into a videos-symlinked copy). Vanilla BC reads
    #    the base <repo> directly.
    *[
        TrainConfig(
            name=f"pi0_{short}_warpbc_sss{n}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"{repo}_sss{n}",
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
                rabc_clip_max=float("inf"),
                top_shortest_frac=0.5,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader(base_ckpt),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for short, (repo, prompt, base_ckpt) in _WARPBC_TASKS.items()
        for n in (15, 30, 45)
    ],
    # ── Multi-cam (3-camera concat RM) WARP-BC: IDENTICAL recipe to
    #    pi0_{short}_warpbc_sss{n} above, EXCEPT the velocity column is produced
    #    by the 3-camera (top + left/right wrist) concat reward model and scored
    #    into the copy <repo>_mc3_sss{n}. Only name + repo_id differ — same
    #    τ=1.0, clip_max=inf, finalaction, top_shortest_frac=0.5, action_horizon,
    #    base init (incl tshirt's special pi0_yam_tshirt ckpt), steps — so the
    #    single-cam-vs-multi-cam comparison is clean and the downstream
    #    real-robot eval is the only differing signal (RM val was ~tied).
    #    PREREQUISITE: <repo>_mc3_sss{n} produced by
    #    warprm2/scripts/launch_mc3_scoring_sky.sh (concat-RM dense inference,
    #    cache-hit over shortest-60%, inject warp_rm_signed_magnitude).
    *[
        TrainConfig(
            name=f"pi0_{short}_warpbc_mc3_sss{n}",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=f"{repo}_mc3_sss{n}",
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                rabc_use_final_action_condition=True,
                rabc_threshold=1.00,
                rabc_clip_max=float("inf"),
                top_shortest_frac=0.5,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader(base_ckpt),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=True,
        )
        for short, (repo, prompt, base_ckpt) in _WARPBC_TASKS.items()
        for n in (15, 30, 45)
    ],
    *[
        TrainConfig(
            name=f"pi0_{short}_bc",
            model=pi0_config.Pi0Config(action_horizon=30),
            data=LeRobotYamRormDataConfig(
                repo_id=repo,
                default_prompt=prompt,
                base_config=DataConfig(prompt_from_task=True),
                top_shortest_frac=0.5,
            ),
            batch_size=32,
            num_workers=8,
            weight_loader=weight_loaders.CheckpointWeightLoader(base_ckpt),
            num_train_steps=60_000,
            save_interval=30_000,
            keep_period=30_000,
            rabc_enabled=False,
        )
        for short, (repo, prompt, base_ckpt) in _WARPBC_TASKS.items()
    ],
    # ── icrrt E12 curation 3-arm study (real bottles, bs128 speedup recipe) ──
    # Same 2747-episode pool for all arms (no top_shortest filter), 60k steps
    # at batch 128 (fsdp 2, per the docs/speedup recipe — OPENPI_REMAT_POLICY
    # and the 0.93 XLA pool ride in via the sky launcher env). Arm (b) reads
    # E12's velocity injected as warp_rm_signed_magnitude in <repo>_e12rabc
    # (final-action gate reproduces the e12_zeroshot keeps exactly,
    # keep_frac 0.28306); arm (c) reads a seeded random gate (2.0 at
    # p=0.28306 = E12's keep rate) in <repo>_rndmatch with clip_max=1.0 so
    # every kept chunk weighs exactly 1.
    TrainConfig(
        name="pi0_bottles_e12study_vanilla_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_sss45",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        rabc_enabled=False,
    ),
    TrainConfig(
        name="pi0_bottles_e12study_e12rabc_thr100_nomax_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_e12rabc",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        rabc_enabled=True,
    ),
    # Sidecar twin of the e12rabc arm: same gate (final-action thr 1.0 nomax)
    # but the velocity comes from the e12 scorer's frame_signals.parquet at
    # sample time — trains on the UNMODIFIED base sss45 dataset. Gate decisions
    # are identical to pi0_bottles_e12study_e12rabc_thr100_nomax_bs128 (the
    # baked column was generated from this exact parquet).
    TrainConfig(
        name="pi0_bottles_e12study_e12rabc_sidecar_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotVelocitySidecarDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_sss45",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            velocity_sidecar_path="s3://xdof-internal-research/icrrt/curation/bottles_d405_v021_full/e12_zeroshot/frame_signals.parquet",
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_bottles_e12study_rndmatch_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotYamRormDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_rndmatch",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=1.0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        rabc_enabled=True,
    ),
    # ── e12e_allwrist_ctx8 sidecar pair (new RM, same _sss45 base pool) ──
    # Same final-action gate (thr 1.0) as the e12rabc sidecar arm but scored by
    # the e12e_allwrist_ctx8 model. The two arms differ ONLY in kept-chunk
    # weighting: nomax = weight is the raw velocity (magnitude-weighted);
    # max1 = clip_max=1.0 so every kept chunk weighs exactly 1 (binary keep).
    # velocity_sidecar_path is the LOCAL staged parquet — the s3:// read path
    # does a single `aws s3 cp` that the box proxy keeps breaking.
    TrainConfig(
        name="pi0_bottles_e12e_allwrist_ctx8_sidecar_thr100_nomax_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotVelocitySidecarDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_sss45",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            velocity_sidecar_path="/mnt/data/karim/sidecars/bottles_d405_v021_full/e12e_allwrist_ctx8/frame_signals.parquet",
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=float("inf"),
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        rabc_enabled=True,
    ),
    TrainConfig(
        name="pi0_bottles_e12e_allwrist_ctx8_sidecar_thr100_max1_bs128",
        model=pi0_config.Pi0Config(action_horizon=30),
        data=LeRobotVelocitySidecarDataConfig(
            repo_id="put_the_plastic_bottles_in_the_bin_d405_v021_sss45",
            default_prompt="Put the plastic bottles in the bin",
            base_config=DataConfig(prompt_from_task=True),
            velocity_sidecar_path="/mnt/data/karim/sidecars/bottles_d405_v021_full/e12e_allwrist_ctx8/frame_signals.parquet",
            rabc_use_final_action_condition=True,
            rabc_threshold=1.00,
            rabc_clip_max=1.0,
        ),
        batch_size=128,
        fsdp_devices=2,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=15_000),
        num_train_steps=15_000,
        save_interval=5_000,
        keep_period=5_000,
        rabc_enabled=True,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
