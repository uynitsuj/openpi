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
    XDOF = "xdof"
    XDOF_FAST = "xdof_fast"
    XDOF_FAST_XMI_RBY = "xdof_fast_xmi_rby"
    XDOF_XMI_RBY = "xdof_xmi_rby"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


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
    port: int = 8111
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.XDOF: Checkpoint(
        config="pi0_yam_low_mem_finetune",
        dir="/home/justinyu/checkpoints/pi0_yam_low_mem_finetune/unload_dishes_from_dishrack_abs_joint_use_action/29999",
    ),
    EnvMode.XDOF_FAST: Checkpoint(
        config="pi0fast_yam_low_mem_finetune",
        dir="/home/justinyu/checkpoints/pi0fast_yam_low_mem_finetune/unload_dishes_from_dishrack/20000",
    ),
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="gs://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    EnvMode.XDOF_FAST_XMI_RBY: Checkpoint(
        config="pi0_fast_xmi_rby_low_mem_finetune",
        # dir="/home/justinyu/checkpoints/pi0_fast_xmi_rby_low_mem_finetune/pi0fast_coffee_cup_20250709_1500/29999",
        dir="s3://xdof-internal-research/model_ckpts/pi0_fast_xmi_rby_low_mem_finetune/sky_pi0fast_soup_can_in_domain_29D_intergripper_relative_20250801/36000",
    ),
    EnvMode.XDOF_XMI_RBY: Checkpoint(
        # config="pi0_xmi_rby_low_mem_finetune",
        config="pi0_xmi_rby",
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_soup_can_in_domain_xmi_data_center_cropped_20250818_20250818_123408/44999" # SOTA tabletop 29D Evaled 36/40
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_soup_can_in_domain_xmi_data_center_cropped_20250828_20250831_203302/13000" # SOTA tabletop 20D Evaled 29/40

        
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_shelf_soup_in_domain_xmi_data_20250822_20250822_161415/32000" # SOTA shelf old data
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_shelf_soup_in_domain_xmi_data_20250902_20250902_162910/34000" # SOTA shelf 29D Evaled 35/40
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_shelf_soup_in_domain_xmi_data_20250902_20250902_161443/10000" # SOTA shelf 20D Evaling

        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_fruit_memory_xmi_data_20250905_20250905_153550/61000" # Fruit Picking Memory
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_fruit_tabletop_memory_xmi_data_20250908_20250908_204351/11000" # Fruit Picking Memory

        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_fruit_plate_memory_xmi_data_20250909_20250909_201239/13000" # Fruit Picking Memory v2

        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_shopping_basket_memory_xmi_data_20250909_20250909_124612/18000" # Item Picking Single Tstep State
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_shopping_basket_memory_xmi_data_20250909_20250909_151858/14000" # Item Picking Keyframe SelectMemory Horizon 2

        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_memory_xmi_data_20250910_20250910_221954/28000" # Item Picking Single Tstep State
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_memory_xmi_data_20250910_20250910_231735/34000" # Item Picking Keyframe Select Memory Horizon 2

        dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_memory_xmi_data_20250911_20250911_161526/36000" # Item Picking Keyframe Select Memory Horizon 2 v2
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_sort_item_memory_xmi_data_20250911_w_negative_trajs_20250911_222815/12000" # Item Picking Keyframe Select Memory Horizon 2 v2 w negative trajs

        # dir="/home/justinyu/checkpoints/pi0_xmi_rby/sky_dishrack_unload_20250823_20250823_233046/16000" # SOTA dishrack (?)
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby_low_mem_finetune/whismy_shelf_soup_can_in_domain_29D_intergripper_relative_20250825/24000" # shelf last-mile finetune
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby_low_mem_finetune/ruin_mean_flow_soup_can_in_domain_29D_intergripper_relative_20250827/35000" # Mean Flow Tabletop Soup Can
        # dir="/home/justinyu/checkpoints/pi0_xmi_rby_low_mem_finetune/ruin_tabletop_soup_can_20D_20250825/34000" # 20D Tabletop Soup Can

    ),
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
