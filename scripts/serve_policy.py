#!/usr/bin/env python3
"""Serve any TrainConfig-defined policy from a local checkpoint over the OpenPI websocket protocol.

The served metadata carries ``policy_name`` (``<config>/<run>/<step>`` by default) so
robots_realtime can resolve ``save_root: recordings/{policy_name}``.

Examples:
    # pi0.5 siemens industrial packing, step 14999
    uv run scripts/serve_policy.py \
        --config pi05_siemens_industrial_packing_bs128 \
        --checkpoint-dir /nfs_us_2/siemens/policy_ckpts/pi05_siemens_industrial_packing_bs128/<run>/14999 \
        --default-prompt "pack the items into the box" --port 8012

    # Load-and-infer sanity check on a synthetic YAM observation (no server, exits afterwards)
    uv run scripts/serve_policy.py --config ... --checkpoint-dir ... --smoke-test
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time

import numpy as np
import tyro

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.policies import yam_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

logger = logging.getLogger(__name__)

YAM_ACTION_DIM = 14


@dataclasses.dataclass
class Args:
    """Local checkpoint serving arguments."""

    # TrainConfig name (see openpi/training/config.py), e.g. pi05_siemens_industrial_packing_bs128.
    config: str
    # Checkpoint step directory containing params/ and assets/. A run directory (containing numbered
    # step directories) is also accepted; the highest step is used.
    checkpoint_dir: pathlib.Path
    port: int = 8012
    host: str = "0.0.0.0"
    # Injected as the language instruction when the client does not send one. For datasets converted
    # with prompt_from_task this should be the task_name the episodes were recorded with.
    default_prompt: str | None = None
    # Override model.action_horizon. Only use this if you know the checkpoint was trained with a
    # different horizon than the config declares -- a mismatch degrades the sampled chunk.
    action_horizon: int | None = None
    # Reported to clients; defaults to <config>/<run>/<step>.
    policy_name: str | None = None
    # Wrap the policy in a PolicyRecorder (writes every obs/action pair to policy_records/).
    record: bool = False
    # Load the checkpoint, run two inferences on a synthetic YAM observation, print timings, exit.
    smoke_test: bool = False


def _resolve_checkpoint_dir(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {path}")
    if (path / "params").is_dir():
        return path
    # Run directory: pick the highest numeric step.
    steps = sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if steps and (steps[-1] / "params").is_dir():
        logger.info("checkpoint_dir is a run directory; using latest step %s", steps[-1].name)
        return steps[-1]
    raise ValueError(f"{path} contains neither params/ nor numbered step directories with params/")


def _check_assets(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path) -> str:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError(f"config {train_config.name} has no asset_id; cannot locate norm stats")
    norm_stats = checkpoint_dir / "assets" / asset_id / "norm_stats.json"
    if not norm_stats.is_file():
        assets_dir = checkpoint_dir / "assets"
        available = sorted(p.name for p in assets_dir.iterdir()) if assets_dir.is_dir() else []
        raise FileNotFoundError(
            f"norm stats not found at {norm_stats}. The config's repo_id/asset_id ({asset_id}) must match the "
            f"asset folder saved with the checkpoint; available: {available}"
        )
    return asset_id


def _smoke_test(policy: policy_lib.Policy, train_config: _config.TrainConfig, prompt: str | None) -> None:
    example = yam_policy.make_yam_example()
    if prompt is not None:
        example["prompt"] = prompt
    logger.info("smoke test: first inference (includes JIT compile) ...")
    t0 = time.monotonic()
    out = policy.infer(example)
    compile_s = time.monotonic() - t0
    t0 = time.monotonic()
    out = policy.infer(example)
    steady_s = time.monotonic() - t0
    actions = np.asarray(out["actions"])
    expected = (train_config.model.action_horizon, YAM_ACTION_DIM)
    if actions.shape != expected:
        raise AssertionError(f"actions shape {actions.shape} != expected {expected}")
    if not np.all(np.isfinite(actions)):
        raise AssertionError("actions contain NaN/inf")
    logger.info(
        "smoke test OK: actions %s, first call %.1fs (compile), steady-state %.0f ms, "
        "joint range [%.2f, %.2f], gripper range [%.2f, %.2f], server_timing=%s",
        actions.shape,
        compile_s,
        steady_s * 1000,
        actions[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].min(),
        actions[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max(),
        actions[:, [6, 13]].min(),
        actions[:, [6, 13]].max(),
        out.get("policy_timing"),
    )


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    if args.action_horizon is not None:
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, action_horizon=args.action_horizon)
        )
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    asset_id = _check_assets(train_config, checkpoint_dir)
    logger.info(
        "config=%s checkpoint=%s asset_id=%s action_horizon=%s default_prompt=%r",
        args.config,
        checkpoint_dir,
        asset_id,
        train_config.model.action_horizon,
        args.default_prompt,
    )

    policy = policy_config.create_trained_policy(train_config, checkpoint_dir, default_prompt=args.default_prompt)

    if args.smoke_test:
        _smoke_test(policy, train_config, args.default_prompt)
        return

    if args.record:
        policy = policy_lib.PolicyRecorder(policy, "policy_records")

    policy_name = args.policy_name or f"{args.config}/{checkpoint_dir.parent.name}/{checkpoint_dir.name}"
    metadata = dict(policy.metadata or {})
    metadata.update(
        policy_name=policy_name,
        config=args.config,
        checkpoint_dir=str(checkpoint_dir),
        action_horizon=train_config.model.action_horizon,
        default_prompt=args.default_prompt,
    )
    logger.info("serving %s on ws://%s:%d", policy_name, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
