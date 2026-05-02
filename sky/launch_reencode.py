#!/usr/bin/env python3
"""
SkyPilot launcher for `scripts/reencode_dense_keyframes.py` — re-encodes a
LeRobot dataset's videos with denser keyframes (default GOP=10) for ~5×
faster random-access decode during training.

CPU-only job (ffmpeg + libx264 are CPU-bound). Targets a 32+ vCPU instance,
falls back across AWS / Lambda. Total wall-clock ≈ 30-60 min for a 1356-video
dataset like hlm_tshirt_reward_select on a c5.9xlarge.

Pipeline:
  1. aws s3 sync s3://<bucket>/lerobot/<src>/ → /data/lerobot/<src>/
  2. uv run scripts/reencode_dense_keyframes.py --src ... --dst ...
  3. aws s3 sync /data/lerobot/<dst>/ → s3://<bucket>/lerobot/<dst>/

Usage:
    cd ~/openpi
    uv run sky/launch_reencode.py --src-repo-id hlm_tshirt_reward_select

    # Custom destination name + workers:
    uv run sky/launch_reencode.py \\
        --src-repo-id tshirt_folding_d405_v010_20260420 \\
        --dst-repo-id tshirt_folding_d405_v010_20260420_gop10 \\
        --workers 48 --gop 10

    # Custom bucket (e.g. external collaborator):
    REPROMO_S3_BUCKET=my-bucket uv run sky/launch_reencode.py --src-repo-id <name>
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tyro
import yaml


@dataclass
class Args:
    src_repo_id: str
    """LeRobot repo to re-encode. Last path segment under s3://<bucket>/<s3_prefix>/."""

    dst_repo_id: Optional[str] = None
    """Destination repo. Defaults to `<src_repo_id>_gop<gop>`."""

    s3_bucket: str = "xdof-internal-research"
    """S3 bucket name (no `s3://`). Override with `--s3-bucket your-bucket`."""

    s3_prefix: str = "lerobot"
    """Sub-path under s3://<bucket>/ where LeRobot repos live."""

    gop: int = 10
    """Keyframe interval. Matches openpi default."""

    crf: int = 23
    """x264 CRF. Default 23 = visually lossless-ish."""

    preset: str = "veryfast"
    """x264 preset. veryfast saves ~3x re-encode time."""

    workers: int = 32
    """Parallel ffmpeg processes on the worker. Cap at instance vCPU count."""

    cpus: str = "32+"
    """sky `cpus` selector. `32+` means at least 32 vCPUs. Ignored if
    `instance_type` is set."""

    memory: str = "32+"
    """sky `memory` selector in GiB. ffmpeg is CPU-heavy, light on RAM.
    Ignored if `instance_type` is set."""

    instance_type: Optional[str] = "c5.9xlarge"
    """Explicit AWS instance type. Default `c5.9xlarge` (36 vCPU, $1.53/hr
    in us-west-2) bypasses the sky `cpus` resolver, which can hit
    `ValueError: 'GH200:1' is not in list` on certain controller versions
    when no accelerators are requested. Set to None to fall back to the
    `cpus`/`memory` selectors."""

    disk_size_gb: int = 200
    """Root disk. Source + dest videos are roughly the same size; dataset-scale dependent."""

    providers: List[str] = field(default_factory=lambda: ["aws", "lambda"])
    """Cloud providers, ordered by preference."""

    region: str = "us-west-2"
    """AWS region. Co-locate with S3 to skip egress."""

    use_spot: bool = False
    """Spot is fine for re-encode (job is restartable)."""

    managed: bool = True
    """sky jobs launch (auto-teardown) vs sky launch (manual)."""

    dry_run: bool = False
    """Print the generated YAML and exit."""

    cluster_name: Optional[str] = None
    """Explicit job name. Defaults to `reencode-<dst>-<timestamp>`."""

    overwrite: bool = False
    """If True, ffmpeg re-encodes videos even when destination exists."""


def _build_run_script(args: Args, dst_repo_id: str) -> str:
    """Bash payload that the worker runs."""
    src_s3 = f"s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/{args.src_repo_id}"
    dst_s3 = f"s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/{dst_repo_id}"
    src_local = f"/data/lerobot/{args.src_repo_id}"
    dst_local = f"/data/lerobot/{dst_repo_id}"
    overwrite_flag = " --overwrite" if args.overwrite else ""

    # The reencode script only needs ffmpeg + python stdlib + tyro + tqdm.
    # We avoid uv here so the launcher works on bare cloud AMIs without
    # any pre-installed Python tooling.
    lines = [
        "set -euo pipefail",
        "",
        'echo "[RUN] task_id=$SKYPILOT_TASK_ID cluster_info=$SKYPILOT_CLUSTER_INFO"',
        "",
        # /data is owned by root on a fresh worker.
        'if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi',
        "$SUDO mkdir -p /data && $SUDO chown $USER:$USER /data",
        "",
        # Single apt pass to install all system deps.
        'echo "[RUN] installing ffmpeg + awscli + pip"',
        "$SUDO apt-get update -qq",
        "$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg awscli python3-pip",
        "pip3 install --user --quiet tyro tqdm",
        'export PATH="$HOME/.local/bin:$PATH"',
        "",
        f'mkdir -p {shlex.quote(src_local)} {shlex.quote(dst_local)}',
        "",
        f'echo "[RUN] sync source from {src_s3}"',
        f'aws s3 sync {shlex.quote(src_s3)} {shlex.quote(src_local)}',
        "",
        f'echo "[RUN] reencode {args.src_repo_id} → {dst_repo_id} (gop={args.gop}, workers={args.workers})"',
        f"python3 scripts/reencode_dense_keyframes.py "
        f"--src {shlex.quote(src_local)} "
        f"--dst {shlex.quote(dst_local)} "
        f"--gop {args.gop} --crf {args.crf} --preset {shlex.quote(args.preset)} "
        f"--workers {args.workers}{overwrite_flag}",
        "",
        f'echo "[RUN] upload to {dst_s3}"',
        f'aws s3 sync {shlex.quote(dst_local)} {shlex.quote(dst_s3)}',
        "",
        'N_VID=$(find /data/lerobot/' + dst_repo_id + ' -name "*.mp4" | wc -l)',
        'echo "[OK] reencode complete: $N_VID videos at ' + dst_s3 + '"',
    ]
    return "\n".join(lines) + "\n"


def _build_resources(args: Args) -> dict:
    """Sky any_of resources list — one entry per provider.

    Newer SkyPilot rejects `infra: aws` + separate `region:` — `infra` must
    be `aws/<region>`. Lambda has no region pin, so plain `infra: lambda`.

    When `instance_type` is set, use that on AWS and drop cpus/memory; the
    cpus selector triggers a controller-side accelerator catalog lookup
    that fails on certain sky 0.12.0 controllers with
    `ValueError: 'GH200:1' is not in list`.
    """
    out = []
    for p in args.providers:
        entry: dict = {"disk_size": args.disk_size_gb}
        if p == "aws":
            entry["infra"] = f"aws/{args.region}"
            entry["use_spot"] = args.use_spot
            if args.instance_type:
                entry["instance_type"] = args.instance_type
            else:
                entry["cpus"] = args.cpus
                entry["memory"] = args.memory
        else:
            entry["infra"] = p
            entry["cpus"] = args.cpus
            entry["memory"] = args.memory
        out.append(entry)
    return {"any_of": out}


def _build_sky_yaml(args: Args, dst_repo_id: str) -> dict:
    return {
        "name": args.cluster_name or f"reencode-{dst_repo_id}",
        "workdir": str(Path(__file__).resolve().parent.parent),
        "resources": _build_resources(args),
        "envs": {
            "AWS_DEFAULT_REGION": args.region,
        },
        "run": _build_run_script(args, dst_repo_id),
    }


def main(args: Args) -> None:
    dst_repo_id = args.dst_repo_id or f"{args.src_repo_id}_gop{args.gop}"
    print(f"[reencode] {args.src_repo_id}  →  {dst_repo_id}")
    print(f"[reencode] s3://{args.s3_bucket}/{args.s3_prefix}/{args.src_repo_id}/")
    print(f"           → s3://{args.s3_bucket}/{args.s3_prefix}/{dst_repo_id}/")
    print(f"[reencode] gop={args.gop} workers={args.workers} preset={args.preset} crf={args.crf}")

    sky_yaml = _build_sky_yaml(args, dst_repo_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sky_yaml, f, default_flow_style=False, sort_keys=False)
        yaml_path = Path(f.name)

    if args.dry_run:
        print()
        print(f"[dry-run] task YAML: {yaml_path}")
        print()
        print(yaml.dump(sky_yaml, default_flow_style=False, sort_keys=False))
        return

    job_name = args.cluster_name or f"reencode-{dst_repo_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    cmd = (
        ["sky", "jobs", "launch", str(yaml_path), "--yes", "--async", "-n", job_name]
        if args.managed
        else ["sky", "launch", str(yaml_path), "--yes", "--cluster", job_name]
    )
    print(f"[reencode] $ {shlex.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[ERROR] sky launch failed (rc={rc})")
        sys.exit(rc)


if __name__ == "__main__":
    main(tyro.cli(Args))
