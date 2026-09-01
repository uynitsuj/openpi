"""Backfill the siemens-industrial-packing wandb project with all packing runs.

Sources:
  - v1 (cloud jobs 6/7): metric history lost with cluster teardown -> config + summary only.
  - v2: full loss curves parsed from zedonly_v2_local.log / combined_v2_local.log.
  - v3: parsed from v3_pipeline.log (sections split on arm markers); an arm is only
    backfilled once its DONE_ARM line exists — rerun this script after each arm finishes.

Idempotent: skips any run whose name already exists in the project.
Usage: WANDB_ENTITY=<optional> .venv/bin/python wandb_backfill.py
"""

import os
import re
from pathlib import Path

import wandb

PROJECT = "siemens-industrial-packing"
ENTITY = os.environ.get("WANDB_ENTITY", "karim-el-refai-ucb")
OPENPI = Path("/home/karim/openpi")
STEP_RE = re.compile(r"Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+), param_norm=([\d.eE+-]+)")
VAL_RE = re.compile(r"Step (\d+): val_loss=([\d.eE+-]+)")

BASE_CFG = {
    "model": "pi0.5 (Pi0Config pi05=True)",
    "action_horizon": 30,
    "batch_size": 128,
    "num_train_steps": 15000,
    "lr_schedule": "cosine decay 15k",
    "save_interval": 5000,
    "weight_init": "gs://openpi-assets/checkpoints/pi05_base/params",
    "job_id": "01a01dc5-bef9-7233-b409-4db2d832ac91",
    "s3_ckpt_root": "s3://xdof-internal-research/siemens/policy_ckpts",
}

CROP_NOTE = ("ZED top cropped to D405-ref FOV 78.7x63.2deg, bottom-anchored, center +86.7px/fx729 "
             "right of principal point (per-station: sz_48 1194x896@(435,304), sz_43 1226x920@(463,280), "
             "sz_50 1202x900@(453,300), sz_04 1224x918@(442,282))")

RUNS = [
    dict(
        name="v1_lerobot_20260825",
        log=None,
        config=dict(BASE_CFG, version="v1", train_config="pi05_siemens_industrial_packing_bs128",
                    exp_name="siemens_packing_pi05_lerobot_20260825",
                    dataset="industrial_packing_yam (LeRobot v3.0, 716 eps / 939326 frames)",
                    loader="LeRobotDataset", actions="observed state", joint_order="yam flip",
                    hardware="cloud p5.48xlarge 8x80GB (sky job 7)"),
        summary={"final_loss": 0.0022, "final_loss_step": 14500, "sec_per_it": 1.7,
                 "wall_total": "7h43m", "ckpts": "5000/10000/14999, 44.7GB each"},
        notes="v1 LeRobot arm. Metric history lost with cloud cluster teardown; summary from run logs.",
    ),
    dict(
        name="v1_abcloader_20260825",
        log=None,
        config=dict(BASE_CFG, version="v1", train_config="pi05_siemens_packing_abcloader_bs128",
                    exp_name="siemens_packing_pi05_abcloader_20260825",
                    dataset="industrial_packing_abc224 (abc layout, 708 train + 8 val)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    hardware="cloud 8x80GB (sky job 6)"),
        summary={"sec_per_it": 1.7, "wall_job": "~7h51m", "wall_total": "9h49m incl 2h GPU queue",
                 "ckpts": "5000/10000/14999, 44.7GB each"},
        notes="v1 abc-loader arm. Throughput identical to LeRobot arm (GPU-bound). "
              "Metric history lost with cloud cluster teardown.",
    ),
    dict(
        name="v2_zedonly_20260828",
        log=OPENPI / "zedonly_v2_local.log",
        config=dict(BASE_CFG, version="v2", train_config="pi05_siemens_packing_abcloader_v2_zedonly_bs128",
                    exp_name="siemens_packing_pi05_zedonly_v2_20260826",
                    dataset="industrial_packing_abc224_v2, station filter yam_zed_0_61 "
                            "(1346 train eps / 1.62M frames)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    crop=CROP_NOTE, hardware="local SZ box 8xH100-96GB"),
        summary={"wall_min": 288, "it_per_sec": 1.2},
        notes="v2 ZED-only arm, local 8xH100. rc=0.",
    ),
    dict(
        name="v2_combined_20260828",
        log=OPENPI / "combined_v2_local.log",
        config=dict(BASE_CFG, version="v2", train_config="pi05_siemens_packing_abcloader_v2_bs128",
                    exp_name="siemens_packing_pi05_combined_v2_20260826_local",
                    dataset="industrial_packing_abc224_v2 (1427 train eps: cropped ZED + D405)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    crop=CROP_NOTE, hardware="local SZ box 8xH100-96GB"),
        summary={"wall_min": 313, "it_per_sec": 1.2},
        notes="v2 combined arm, local 8xH100. rc=0. Cloud job 12 cancelled after 6h40m GPU famine.",
    ),
    dict(
        name="v3_zedonly_20260829",
        log=("v3", "zedonly"),
        config=dict(BASE_CFG, version="v3", train_config="pi05_siemens_packing_abcloader_v3_zedonly_bs128",
                    exp_name="siemens_packing_pi05_zedonly_v3_20260829",
                    dataset="industrial_packing_abc224_v3, station filter yam_zed_0_61 "
                            "(+3 sz_04 eps vs v2)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    crop=CROP_NOTE, hardware="local SZ box 8xH100-96GB"),
        summary={},
        notes="v3 ZED-only arm (adds sz_04 station).",
    ),
    dict(
        name="v3cc_combined_20260830",
        log=OPENPI / "v3cc_local.log",
        config=dict(BASE_CFG, version="v3cc", train_config="pi05_siemens_packing_abcloader_v3cc_bs128",
                    exp_name="siemens_packing_pi05_combined_v3cc_20260829",
                    dataset="industrial_packing_abc224_v3cc (same 1460 train + same 8 val eps as v3; "
                            "center-crop resize instead of letterbox pad)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    resize_mode="center_crop",
                    crop=CROP_NOTE + "; then largest centered square (wrists -37.5% hFOV, top window ~-25%)",
                    val="live val pass: val_interval=1000, 8x128 frames from the D405 val split, fixed rng",
                    hardware="local SZ box 8xH100-96GB"),
        summary={},
        notes="Center-crop ablation of v3 combined; first run with train.py's live val pass. "
              "Serving THIS checkpoint requires center_crop preprocessing (bair-style), "
              "unlike every pad-lineage run in this project.",
    ),
    dict(
        name="v3_combined_20260829",
        log=("v3", "combined"),
        config=dict(BASE_CFG, version="v3", train_config="pi05_siemens_packing_abcloader_v3_bs128",
                    exp_name="siemens_packing_pi05_combined_v3_20260829",
                    dataset="industrial_packing_abc224_v3 (1460 train eps: +30 D405 sz_44, +3 ZED sz_04)",
                    loader="AbcLayoutDataset", actions="commanded", joint_order="raw",
                    crop=CROP_NOTE, hardware="local SZ box 8xH100-96GB"),
        summary={},
        notes="v3 combined arm (119 D405 eps vs 89 in v2).",
    ),
]


def parse_steps(text: str):
    return [(int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            for m in STEP_RE.finditer(text)]


def v3_section(arm: str) -> str | None:
    """Return the log section for a v3 arm, only if that arm reached DONE_ARM."""
    text = (OPENPI / "v3_pipeline.log").read_text(errors="replace")
    z_start = text.find("2/3 ZED-only v3 arm")
    c_start = text.find("3/3 combined v3 arm")
    if arm == "zedonly":
        if z_start < 0:
            return None
        section = text[z_start:c_start if c_start > 0 else len(text)]
    else:
        if c_start < 0:
            return None
        section = text[c_start:]
    return section if "DONE_ARM" in section else None


def existing_run_names() -> set[str]:
    try:
        api = wandb.Api(timeout=30)
        ent = ENTITY or api.default_entity
        return {r.name for r in api.runs(f"{ent}/{PROJECT}")}
    except Exception:
        return set()  # project doesn't exist yet


def main():
    done = existing_run_names()
    for spec in RUNS:
        if spec["name"] in done:
            print(f"skip (exists): {spec['name']}")
            continue
        steps, vals = [], []
        wall_min = None
        text = None
        if isinstance(spec["log"], tuple):
            text = v3_section(spec["log"][1])
            if text is None:
                print(f"skip (not finished): {spec['name']}")
                continue
            m = re.search(r"DONE_ARM rc=(\d+) wall=(\d+)min", text)
            if m:
                spec["summary"]["train_rc"] = int(m.group(1))
                wall_min = int(m.group(2))
        elif spec["log"] is not None:
            text = Path(spec["log"]).read_text(errors="replace")
            m = re.search(r"DONE rc=(\d+) wall=(\d+)min", text)
            if m:
                spec["summary"]["train_rc"] = int(m.group(1))
                wall_min = int(m.group(2))
        if text is not None:
            steps = parse_steps(text)
            vals = [(int(s), float(v)) for s, v in VAL_RE.findall(text)]

        run = wandb.init(project=PROJECT, entity=ENTITY, name=spec["name"],
                         config=spec["config"], notes=spec["notes"],
                         tags=[spec["config"]["version"], "pi0.5", "backfill"],
                         reinit=True)
        # Merge train and val points so wandb sees monotonically increasing steps.
        merged: dict[int, dict] = {}
        for step, gn, loss, pn in steps:
            merged.setdefault(step, {}).update(
                {"train/loss": loss, "train/grad_norm": gn, "train/param_norm": pn})
        for step, v in vals:
            merged.setdefault(step, {})["val_loss"] = v
        for step in sorted(merged):
            run.log(merged[step], step=step)
        for k, v in spec["summary"].items():
            run.summary[k] = v
        if wall_min is not None:
            run.summary["wall_min"] = wall_min
        if steps:
            run.summary["final_loss"] = steps[-1][2]
            run.summary["logged_points"] = len(steps)
        if vals:
            run.summary["final_val_loss"] = vals[-1][1]
            best_step, best_val = min(vals, key=lambda x: x[1])
            run.summary["best_val_loss"] = best_val
            run.summary["best_val_step"] = best_step
        ckpt = f"{BASE_CFG['s3_ckpt_root']}/{spec['config']['train_config']}/{spec['config']['exp_name']}"
        run.summary["s3_checkpoints"] = ckpt
        run.finish()
        print(f"backfilled: {spec['name']} ({len(steps)} points)")


if __name__ == "__main__":
    main()
