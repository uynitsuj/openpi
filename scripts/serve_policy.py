import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    YAM = "yam"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class NamedCheckpoint:
    """Load a preset checkpoint by short name (see NAMED_CHECKPOINTS)."""

    name: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8012
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | NamedCheckpoint | Default = dataclasses.field(default_factory=Default)


# Named presets for frequently-served checkpoints. Select via `--policy named --policy.name <key>`.
NAMED_CHECKPOINTS: dict[str, Checkpoint] = {
    "yam_no_rabc_39k": Checkpoint(
        config="pi0_yam_tshirt_no_rabc",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_no_rabc/sky_yam_tshirt_rorm_weighted_20260415_000110/39999",
    ),
    "yam_rabc_1k": Checkpoint(
        config="pi0_yam_tshirt_rabc",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc/sky_pi0_yam_tshirt_rabc_yam_tshirt_rorm_weighted_20260415_090157/1000",
    ),
    "yam_rabc_11k_a": Checkpoint(
        config="pi0_yam_tshirt_rabc",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc/sky_pi0_yam_tshirt_rabc_yam_tshirt_rorm_weighted_20260415_111935/11000",
    ),
    "yam_rabc_11k_b": Checkpoint(
        config="pi0_yam_tshirt_rabc",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc/sky_pi0_yam_tshirt_rabc_yam_tshirt_rorm_weighted_20260415_115838/11000",
    ),
    "yam_rabc_30k": Checkpoint(
        config="pi0_yam_tshirt_rabc",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc/sky_pi0_yam_tshirt_rabc_yam_tshirt_rorm_weighted_20260415_174132/30000",
    ),
    "yam_rabc_thresh1_clip6_39k": Checkpoint(
        config="pi0_yam_tshirt_rabc_thresh_1_00_clip_max_6_0",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc_thresh_1_00_clip_max_6_0/sky_pi0_yam_tshirt_rabc_thresh_1_00_clip_max_6_0_yam_tshirt_rorm_weighted_20260415_183732/39999",
    ),
    "yam_rabc_thresh0_25_clip6_39k": Checkpoint(
        config="pi0_yam_tshirt_rabc_thresh_0_25_clip_max_6_0",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc_thresh_0_25_clip_max_6_0/sky_pi0_yam_tshirt_rabc_thresh_0_25_clip_max_6_0_yam_tshirt_rorm_weighted_20260415_184347/39999",
    ),
    "yam_rabc_thresh0_50_clip6_39k": Checkpoint(
        config="pi0_yam_tshirt_rabc_thresh_0_50_clip_max_6_0",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc_thresh_0_50_clip_max_6_0/sky_pi0_yam_tshirt_rabc_thresh_0_50_clip_max_6_0_yam_tshirt_rorm_weighted_20260415_184251/39999",
    ),
    "yam_rabc_thresh0_75_clip6_30k": Checkpoint(
        config="pi0_yam_tshirt_rabc_thresh_0_75_clip_max_6_0",
        dir="/home/justinyu/checkpoints/pi0_yam_tshirt_rabc_thresh_0_75_clip_max_6_0/sky_pi0_yam_tshirt_rabc_thresh_0_75_clip_max_6_0_yam_tshirt_rorm_weighted_20260415_184228/30000",
    ),
}


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.YAM: NAMED_CHECKPOINTS["yam_no_rabc_39k"],
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case NamedCheckpoint():
            if args.policy.name not in NAMED_CHECKPOINTS:
                raise ValueError(
                    f"Unknown named checkpoint: {args.policy.name!r}. "
                    f"Available: {sorted(NAMED_CHECKPOINTS)}"
                )
            ckpt = NAMED_CHECKPOINTS[args.policy.name]
            return _policy_config.create_trained_policy(
                _config.get_config(ckpt.config), ckpt.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
