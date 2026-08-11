#!/usr/bin/env python3
"""Serve either public WARP-RM paper simulation policy from a local download."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

import tyro

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config


POLICY_CONFIGS = {
    "vanilla": "pi0_warp_rm_sim_bottles_vanilla",
    "warp_rabc_sss15": "pi0_warp_rm_sim_bottles_rabc_sss15",
    "hang_mug": "pi0_sim_hang_mug",
}
DEFAULT_PROMPT = "Put the plastic bottles in the bin"


@dataclasses.dataclass
class Args:
    """Public-only local serving arguments."""

    checkpoint_dir: Path
    policy: Literal["vanilla", "warp_rabc_sss15", "hang_mug"] = "vanilla"
    port: int = 8000
    default_prompt: str = DEFAULT_PROMPT
    record: bool = False


def main(args: Args) -> None:
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not (checkpoint_dir / "params").is_dir() or not (checkpoint_dir / "assets").is_dir():
        raise ValueError("checkpoint_dir must contain both params/ and assets/")

    policy = policy_config.create_trained_policy(
        config.get_config(POLICY_CONFIGS[args.policy]),
        checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    if args.record:
        policy = policy_lib.PolicyRecorder(policy, "policy_records")
    metadata = dict(policy.metadata)
    metadata.setdefault("policy_name", args.policy)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
