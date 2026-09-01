import os
import time
import dataclasses
import functools
import logging
import math
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    wandb_id_file = ckpt_dir / "wandb_id.txt"
    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # No wandb_id.txt yet (fresh run) or it never made it to S3 from the
        # prior crashed/cancelled run — start a new wandb run either way.
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        wandb_id_file.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        if (config.rabc_enabled or config.online_rm_enabled) and observation.sample_weights is not None:
            per_sample_loss = jnp.mean(chunked_loss, axis=-1)  # [B]
            weighted_loss = per_sample_loss * observation.sample_weights  # [B]
            if config.rabc_normalize_weights:
                return jnp.sum(weighted_loss) / (jnp.sum(observation.sample_weights) + 1e-6)
            return jnp.mean(weighted_loss)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if (config.rabc_enabled or config.online_rm_enabled) and observation.sample_weights is not None:
        sample_weights = observation.sample_weights
        info.update(
            {
                "sample_weight_sum": jnp.sum(sample_weights),
                "sample_weight_sq_sum": jnp.sum(jnp.square(sample_weights)),
                "sample_weight_zero_count": jnp.sum(sample_weights == 0),
                "sample_weight_count": sample_weights.size,
            }
        )
    return new_state, info


@at.typecheck
def val_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Validation loss on one batch — train_step's loss without gradients or updates."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return {"val_loss": jnp.mean(chunked_loss)}


def build_val_batches(
    config: _config.TrainConfig,
) -> list[tuple[_model.Observation, _model.Actions]] | None:
    """Pre-build a fixed, deterministic set of val batches (host memory).

    Two val sources: abc-layout datasets have a val/ split on disk
    (export_abc_layout_job.py reserves 8 episodes); LeRobot datasets get one via
    the data config's val_episodes (val_frac/val_seed on LeRobotYamRormDataConfig).
    Indices are evenly spaced over the split so the same frames are evaluated
    every pass and every run. Returns None (with a warning) when no val data is
    usable — training proceeds without validation.
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: PLC0415

    from openpi.training.abc_layout_dataset import AbcLayoutDataset  # noqa: PLC0415

    data_config = config.data.create(config.assets_dirs, config.model)
    if getattr(data_config, "abc_layout", False):
        val_root = HF_LEROBOT_HOME / data_config.repo_id / "val"
        if not val_root.exists():
            logging.warning("val_interval is set but %s does not exist — skipping validation", val_root)
            return None
        station_types = getattr(data_config, "abc_station_types", None)
        try:
            dataset = AbcLayoutDataset(val_root, action_horizon=config.model.action_horizon, station_types=station_types)
        except ValueError:
            # e.g. the v2/v3 exports' val episodes are all D405 while this config trains
            # ZED-only: fall back to the unfiltered val split and label the caveat loudly.
            dataset = AbcLayoutDataset(val_root, action_horizon=config.model.action_horizon, station_types=None)
            logging.warning(
                "val split has no episodes matching station filter %s — using the full val split; "
                "val_loss is out-of-distribution transfer for this config",
                station_types,
            )
        dataset = _data_loader.transform_dataset(dataset, data_config)
        val_src = str(val_root)
    elif data_config.val_episodes:
        # LeRobot-backed split: rebuild the torch dataset over only the held-out
        # episodes, same transform stack. Zero-weight rejection is train-only
        # sample selection — never filter the val set.
        val_config = dataclasses.replace(
            data_config,
            episodes=tuple(data_config.val_episodes),
            reject_zero_weighted_samples=False,
        )
        dataset = _data_loader.create_torch_dataset(val_config, config.model.action_horizon, config.model)
        dataset = _data_loader.transform_dataset(dataset, val_config)
        val_src = f"{data_config.repo_id} ({len(data_config.val_episodes)} val episodes)"
    else:
        logging.warning(
            "val_interval is set but %s has neither an abc-layout val/ split nor val_episodes — skipping validation",
            config.name,
        )
        return None
    n_needed = config.num_val_batches * config.batch_size
    indices = np.linspace(0, len(dataset) - 1, n_needed).astype(int)
    batches = []
    for b in range(config.num_val_batches):
        items = [dataset[int(i)] for i in indices[b * config.batch_size : (b + 1) * config.batch_size]]
        collated = jax.tree.map(lambda *xs: np.stack(xs), *items)
        batches.append((_model.Observation.from_dict(collated), collated["actions"]))
    logging.info("validation: %d batches of %d prepared from %s", len(batches), config.batch_size, val_src)
    return batches


def compute_weighted_fac_float(
    raw_reward: np.ndarray,
    reward: np.ndarray,
    valid_rm_mask: np.ndarray,
    rms,
    beta: float = 2.0,
    eps: float = 1e-8,
    th: float = 8e-3,
) -> tuple[np.ndarray, float, float]:
    reward = np.asarray(reward, dtype=np.float64).ravel()
    valid_rm_mask = np.asarray(valid_rm_mask, dtype=bool).ravel()
    assert reward.shape == valid_rm_mask.shape

    rms.update(np.asarray(raw_reward, dtype=np.float64).ravel())

    mu = float(max(rms.mean, 0.0))
    sigma = float(rms.std)
    denom = max(4.0 * sigma, eps)
    lo = mu - beta * sigma

    fac = np.clip((reward - lo) / denom, 0.0, 1.0)
    fac = np.where(reward > +th, 1.0, fac)
    fac = np.where(reward < -th, 0.0, fac)
    fac = fac * valid_rm_mask.astype(np.float64)

    return fac.astype(np.float32), fac.mean(), rms.mean


def compute_weighted_fac_binary(
    raw_reward: np.ndarray,
    reward: np.ndarray,
    valid_rm_mask: np.ndarray,
    rms,
) -> np.ndarray:
    reward = np.asarray(reward, dtype=np.float64).ravel()
    valid_rm_mask = np.asarray(valid_rm_mask, dtype=bool).ravel()
    assert reward.shape == valid_rm_mask.shape

    thr = 5.0e-3
    fac = (reward > thr).astype(np.float32)
    fac = fac * valid_rm_mask.astype(np.float32)

    return fac


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Online reward model initialization
    rm = None
    rm_stats = None
    rm_data_iter = None
    if config.online_rm_enabled:
        from openpi.reward_model.rm_utils import (
            HybridRM, RMConfig, RunningMeanStd, comply_rm_lerobot_batch_multi_stage,
        )
        rm_cfg = RMConfig()
        rm = HybridRM(rm_cfg)
        rm_stats = RunningMeanStd()
        rm_data_loader = _data_loader.create_rm_data_loader(config, sharding=data_sharding, shuffle=True)
        rm_data_iter = iter(rm_data_loader)
        logging.info("[INIT] Online reward model initialized")

        # Compute weights for the first batch
        rm_batch = next(rm_data_iter)
        valid_rm_mask = np.ones(batch[0].state.shape[0], dtype=bool)
        rm_batch_curr = comply_rm_lerobot_batch_multi_stage(rm_batch['rm'])
        rm_batch_next = comply_rm_lerobot_batch_multi_stage(rm_batch['rm_next'])
        raw_reward, reward, mean_conf = rm.eval_reward(rm_batch_curr, rm_batch_next)

        if config.online_rm_weight_method == 'binary':
            bc_weight = compute_weighted_fac_binary(raw_reward, reward, valid_rm_mask, rm_stats)
        else:
            bc_weight, _, _ = compute_weighted_fac_float(raw_reward, reward, valid_rm_mask, rm_stats)

        observation, actions = batch
        bc_weight_jax = jax.device_put(
            jnp.array(bc_weight),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
        )
        observation = dataclasses.replace(observation, sample_weights=bc_weight_jax)
        batch = (observation, actions)
        logging.info(f"First batch RM weights: {bc_weight}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    val_batches = build_val_batches(config) if config.val_interval > 0 else None
    pval_step = None
    val_rng = jax.random.key(config.seed + 1)
    if val_batches:
        pval_step = jax.jit(
            functools.partial(val_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    running_sample_weight_sum = 0.0
    running_sample_weight_sq_sum = 0.0
    running_sample_weight_zero_count = 0.0
    running_sample_weight_count = 0.0
    # Online RM tracking variables (initialized for first log interval)
    reward = np.array([0.0])
    mean_conf = 0.0
    bc_weight = np.array([0.0])
    mean_weight = 0.0
    rms_mean = 0.0
    _prof_n = int(os.environ.get('OPENPI_PROFILE_STEPS', '0'))
    _prof_warmup = int(os.environ.get('OPENPI_PROFILE_WARMUP', '20'))
    _prof_t0 = None
    _prof_i = 0
    for step in pbar:
        if os.environ.get('OPENPI_PROFILE_LOADER'):
            info = {}  # loader-only: no compute, measures max pull rate
        else:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if _prof_n:
            _prof_i += 1
            if _prof_i == _prof_warmup:
                jax.block_until_ready(train_state)
                _prof_t0 = time.time()
                _tdir = os.environ.get('OPENPI_PROFILE_TRACE')
                if _tdir:
                    jax.profiler.start_trace(_tdir)
            elif os.environ.get('OPENPI_PROFILE_TRACE') and _prof_i == _prof_warmup + 8:
                jax.block_until_ready(train_state)
                jax.profiler.stop_trace()
                print('[PROFILE] trace written', flush=True)
            elif _prof_t0 is not None and _prof_i >= _prof_warmup + _prof_n:
                jax.block_until_ready(train_state)
                _dt = time.time() - _prof_t0
                _bs = int(config.batch_size)
                print(f'[PROFILE] steps={_prof_n} wall={_dt:.2f}s '
                      f'it/s={_prof_n/_dt:.4f} samples/s={_prof_n*_bs/_dt:.2f} '
                      f'synthetic={bool(os.environ.get("OPENPI_PROFILE_SYNTHETIC"))}', flush=True)
                break
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Sample reweighting statistics logging
            if "sample_weight_count" in stacked_infos:
                window_sample_weight_sum = float(np.sum(np.asarray(stacked_infos["sample_weight_sum"])))
                window_sample_weight_sq_sum = float(np.sum(np.asarray(stacked_infos["sample_weight_sq_sum"])))
                window_sample_weight_zero_count = float(np.sum(np.asarray(stacked_infos["sample_weight_zero_count"])))
                window_sample_weight_count = float(np.sum(np.asarray(stacked_infos["sample_weight_count"])))

                running_sample_weight_sum += window_sample_weight_sum
                running_sample_weight_sq_sum += window_sample_weight_sq_sum
                running_sample_weight_zero_count += window_sample_weight_zero_count
                running_sample_weight_count += window_sample_weight_count

                window_sample_weight_mean = window_sample_weight_sum / window_sample_weight_count
                window_sample_weight_var = max(
                    window_sample_weight_sq_sum / window_sample_weight_count - window_sample_weight_mean**2, 0.0
                )
                running_sample_weight_mean = running_sample_weight_sum / running_sample_weight_count
                running_sample_weight_var = max(
                    running_sample_weight_sq_sum / running_sample_weight_count - running_sample_weight_mean**2, 0.0
                )

                reduced_info.update(
                    {
                        "sample_weight_mean": window_sample_weight_mean,
                        "sample_weight_std": math.sqrt(window_sample_weight_var),
                        "sample_weight_zero_frac": window_sample_weight_zero_count / window_sample_weight_count,
                        "sample_weight_mean_running": running_sample_weight_mean,
                        "sample_weight_std_running": math.sqrt(running_sample_weight_var),
                        "sample_weight_zero_frac_running": (
                            running_sample_weight_zero_count / running_sample_weight_count
                        ),
                    }
                )

                for key in (
                    "sample_weight_sum",
                    "sample_weight_sq_sum",
                    "sample_weight_zero_count",
                    "sample_weight_count",
                ):
                    reduced_info.pop(key, None)

            # Online RM extra metrics
            if config.online_rm_enabled and rm is not None:
                reduced_info["online_rm/step_reward"] = float(np.mean(reward))
                reduced_info["online_rm/mean_confidence"] = float(mean_conf)
                reduced_info["online_rm/num_used_actions"] = float(np.sum(bc_weight > 0))
                if config.online_rm_weight_method == 'float':
                    reduced_info["online_rm/mean_weight"] = float(mean_weight)
                    reduced_info["online_rm/rms_mean"] = float(rms_mean)

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if pval_step is not None and step % config.val_interval == 0:
            with sharding.set_mesh(mesh):
                v_losses = []
                for vb_i, vb in enumerate(val_batches):
                    vb_dev = jax.device_put(vb, data_sharding)
                    v_info = pval_step(jax.random.fold_in(val_rng, vb_i), train_state, vb_dev)
                    v_losses.append(float(jax.device_get(v_info["val_loss"])))
            val_loss = float(np.mean(v_losses))
            pbar.write(f"Step {step}: val_loss={val_loss:.4f} ({len(val_batches)} batches)")
            wandb.log({"val_loss": val_loss}, step=step)
        if os.environ.get('OPENPI_PROFILE_SYNTHETIC'):
            pass  # reuse the cached batch: measures the pure-compute ceiling
        else:
            batch = next(data_iter)

        # Online RM: compute per-sample weights and inject into observation
        if config.online_rm_enabled and rm is not None:
            from openpi.reward_model.rm_utils import comply_rm_lerobot_batch_multi_stage
            rm_batch = next(rm_data_iter)
            valid_rm_mask = np.ones(batch[0].state.shape[0], dtype=bool)
            rm_batch_curr = comply_rm_lerobot_batch_multi_stage(rm_batch['rm'])
            rm_batch_next = comply_rm_lerobot_batch_multi_stage(rm_batch['rm_next'])
            raw_reward, reward, mean_conf = rm.eval_reward(rm_batch_curr, rm_batch_next)

            if config.online_rm_weight_method == 'binary':
                bc_weight = compute_weighted_fac_binary(raw_reward, reward, valid_rm_mask, rm_stats)
            else:
                bc_weight, mean_weight, rms_mean = compute_weighted_fac_float(
                    raw_reward, reward, valid_rm_mask, rm_stats
                )

            observation, actions = batch
            bc_weight_jax = jax.device_put(
                jnp.array(bc_weight),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)),
            )
            observation = dataclasses.replace(observation, sample_weights=bc_weight_jax)
            batch = (observation, actions)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step, config.s3_checkpoint_path)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
