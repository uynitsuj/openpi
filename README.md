# OpenPI support for WARP-RM paper simulation

This branch contains the public OpenPI serving code and the two narrow configs
needed for the WARP-RM paper's bottle-in-bin simulation policies. It does not
contain policy weights, datasets, training state, internal checkpoint paths,
or a preset that reaches private storage.

The two policy parameter trees are published at
[`uynitsuj/paper-sim-policy-checkpoints`](https://huggingface.co/uynitsuj/paper-sim-policy-checkpoints).
Their use is subject to the applicable upstream Pi0/OpenPI terms; the artifact
does not assert an independent license grant over upstream-derived weights.

## Serve a public policy

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

hf download uynitsuj/paper-sim-policy-checkpoints \
  --local-dir paper-sim-policy-checkpoints

# Baseline arm
uv run scripts/serve_paper_sim_policy.py \
  --checkpoint-dir paper-sim-policy-checkpoints/vanilla --port 8000

# WARP-RM reward-aligned arm
uv run scripts/serve_paper_sim_policy.py \
  --policy warp_rabc_sss15 \
  --checkpoint-dir paper-sim-policy-checkpoints/warp_rabc_sss15 --port 8000
```

The public ABC MuJoCo-Warp evaluator connects through its `--policy-backend
pi0 --pi0-host --pi0-port` options. For the deterministic paper-table audit,
download the canonical trace artifact and run `score_bottles.py --self-test`;
see the ABC `release-candidate` README. Fresh rollouts are a separate
stochastic check and should be evaluated with the paired n=128 protocol.

## License

Code in this branch is Apache-2.0. Third-party and upstream terms continue to
apply to their respective components and model materials.
