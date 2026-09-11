# Lambda: Siemens v13 shortest-25% training

This setup runs the two 20,000-step, global-batch-128 crop ablations from the
latest `karim/industrial-packing-v3` branch. On the Lambda A100 nodes, each run
uses all eight GPUs in a 2x4 data/FSDP mesh with `fsdp_devices=4`. The configs
retain their validated `fsdp_devices=2` default for H100. On Lambda A100, FSDP4
and the stock memory-minimizing `nothing_saveable` rematerialization policy
avoid the larger A100 XLA backward-workspace peak without reducing global batch
128. A 1x8 FSDP mesh is not used because this JAX/XLA build fails its first
collective with a buffer type/count mismatch.

## Data and node assignment

| Node | Variant | OpenPI config | LeRobot repo | Episodes | Camera preprocessing |
|---|---|---|---|---:|---|
| `132.145.143.219` | CC | `pi05_siemens_simple_d405_v13short25cc_bs128` | `siemens_simple_d405_v13short25cc` | 1,976 | center crop all cameras |
| `64.181.218.97` | TC | `pi05_siemens_simple_d405_v13short25tc_bs128` | `siemens_simple_d405_v13short25tc` | 1,982 | center crop top, pad wrists |

Both datasets are 30 Hz, 224×224, driver-joint-order LeRobot v3 datasets whose
actions come from the recorded leader/action MCAP stream. Dataset inputs are at
`s3://xdof-internal-research/siemens/datasets/<repo>` and checkpoint outputs go
to `s3://xdof-internal-research/siemens/policy_ckpts/<config>/<experiment>`.

## Current run lineage (2026-09-11)

Both runs start from the inference parameters at:

```text
s3://xdof-internal-research/siemens/policy_ckpts/pi05_siemens_simple_d405_v12dj_recent_bs128/siemens_simple_d405_v12dj_recent_20k_20260910/19999/params
```

The experiment names are:

- CC: `siemens_simple_d405_v13short25cc_from_v12dj_recent_20k_20260911`
- TC: `siemens_simple_d405_v13short25tc_from_v12dj_recent_20k_20260911`

Each checkpoint is synchronously copied to the S3 checkpoint root above after
its local Orbax save completes. Step 5,000 for both runs was verified in S3 on
2026-09-11 before this documentation was committed.

## Safe two-phase launcher

Staging is CPU/network-only and cannot launch training:

```bash
cd /home/ubuntu/karim/openpi-v13short25
scripts/run_siemens_v13short25_lambda.sh stage cc  # CC node
scripts/run_siemens_v13short25_lambda.sh stage tc  # TC node
```

Training refuses to start unless both the base parameter URI and experiment
name are explicit. This is the launch gate while the base checkpoint is being
selected:

```bash
cd /home/ubuntu/karim/openpi-v13short25
BASE_PARAMS_URI='s3://bucket/path/to/checkpoint/step/params' \
EXP_NAME='siemens_v13short25cc_20k_<base-label>_20260911' \
  scripts/run_siemens_v13short25_lambda.sh train cc
```

Use `tc` and a TC-specific experiment name on the second node. The launcher
also refuses to launch over an existing GPU process or checkpoint directory.
To continue a genuinely interrupted run, repeat with `RESUME=1`.

The launcher uses `scripts/train_siemens_v13short25.py`, a narrow entry point
that retrieves the existing config with `get_config()` and passes it directly
to `train.main()`. This avoids the generic Tyro CLI's expensive construction of
every config subcommand on this large branch; it also asserts batch 128,
20,000 steps, a valid FSDP width, and the expected dataset before training.

W&B defaults to offline mode because fresh Lambda nodes do not carry a W&B API
key. Offline run files are kept under `/home/ubuntu/karim/wandb` and can be
synced later. Set `WANDB_MODE=online` only after authenticating the node.

Recommended detached launch after the base is chosen:

```bash
tmux new-session -d -s v13short25_cc \
  "cd /home/ubuntu/karim/openpi-v13short25 && BASE_PARAMS_URI='...' EXP_NAME='...' scripts/run_siemens_v13short25_lambda.sh train cc 2>&1 | tee /home/ubuntu/karim/v13short25_cc.log"
```

Check status with:

```bash
nvidia-smi
tmux list-sessions
tail -n 60 /home/ubuntu/karim/v13short25_cc.log
```

The configs save at steps 5,000, 10,000, 15,000, and 19,999. The final step is
19,999 because OpenPI numbers a 20,000-step run from zero.
