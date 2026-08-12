# pi0 training-throughput profile — protective-crimson-cow

Objective (Justin, 2026-08-11): reduce pi0 training wall-clock as much as possible.
Config under test: `pi0_put_bottles_mjwarp_no_rabc` (vanilla BC, sim bottles).
Box: 8xA100-SXM4-80GB, 200 cores, 925 GB RAM, dataset 21 GB on /scratch NVMe.
Recipe baseline: bs=32, 30k steps, --fsdp-devices 2 -> 5.1 h wall.

All numbers: 100-step windows (fewer at large batch) after 20 warmup steps,
`jax.block_until_ready` fenced, `[PROFILE]` line printed by the patched train.py.

## Sweep 1 — where does the time go?

| run | it/s | samples/s | note |
|---|---|---|---|
| bs32 baseline (real loader) | 1.6385 | 52.4 | the recipe |
| bs32 synthetic (reuse batch) | 1.6437 | 52.6 | loader cost = 0.3% |
| loader only (no compute) | 6.7345 | **215.5** | loader max rate |
| bs64 | 1.0519 | 67.3 | |
| bs128 | 0.6016 | 77.0 | |
| bs256 | 0.3377 | 86.5 | |
| bs128, fsdp=1 | OOM | — | full replication does not fit |

**Verdict 1: the dataloader is FREE at the current operating point** (0.3%).
JAX async dispatch already overlaps fetch with compute; 8 workers on 200 cores
sustain 4.1x the compute rate. A custom data format buys ~nothing until compute
exceeds ~215 samples/s. The stacked-frames/GOP repo is GATED on that number,
not dead: it also matters for CPU-poor hosts.

**Verdict 2: batch scaling is sublinear.** Fixed-plus-variable fit of step time:
~0.28 s fixed per step + ~10.5 ms per sample -> asymptote ~95 samples/s.

## Sweep 2 — remat and EMA

| run | samples/s | maxmem | note |
|---|---|---|---|
| bs64 dots_with_no_batch_dims_saveable | 71.1 (+5.6%) | 63.1 G | only survivor |
| bs64 everything_saveable | OOM (+76 G asked) | — | |
| bs128 dots | OOM (+37 G) | — | may fit at higher mem fraction |
| bs128/bs256 everything | OOM (146 G / 266 G) | — | |
| bs64 EMA off | 68.3 (+1.5%) | — | EMA is cheap |

**Verdict 3: `nothing_saveable` remat is LOAD-BEARING, not waste.** Without
aggressive recompute the activations of a ~3B model over ~850 tokens do not fit.
Relaxing remat is worth at most ~6%. EMA ~1.5%. Neither is the lever.

**Found:** XLA allocator pool defaults to 75% = 60 GB of 80. Every run above ran
inside a 60 GB pool; ~14 GB/GPU is being left on the table
(`XLA_PYTHON_CLIENT_MEM_FRACTION`).

**Found:** GPU util during the stepping phase (tail-windowed sampler) is only
53-56% at bs64. Nearly half the step is gaps, not math. Cause unknown ->
profiler trace (sweep 3, running).

## Measurement discipline

- Sweep-1 util numbers were diluted by setup time; the sampler is now windowed
  to the stepping phase. Do not quote sweep-1 util.
- `py_compile` does not catch missing imports; verify names via AST after
  patching (train.py imported neither `os` nor `time`).
- Env-gated hooks on the cow working copy: OPENPI_PROFILE_STEPS / _WARMUP /
  _SYNTHETIC / _LOADER / _TRACE, OPENPI_REMAT_POLICY (gemma.py + siglip.py).
  Originals saved as *.orig. All default-off; the recipe path is byte-identical
  when env vars are unset.

## CONVERGENCE VERDICT — bs128 recipe vs vanilla, same host (cow), n=128 (08-12 15:40 PDT)

speed_bs128_v1: bs128/dots/0.93, 7500 steps = 960k samples (recipe-equal), decay
rescaled, peak lr unchanged. Baseline: released vanilla ckpt evaluated on the SAME
host and seeds. Both arms fresh rollouts.
```

===== tight-anypart  n=128  [PAPER] =====
  bottles placed:  514/768 -> 509/768
  bottles/scene:   4.016 -> 3.977   diff -0.039  t(127)=-0.25  p=8.0e-01
  time/bottle:     10.6 -> 11.0 s   per-scene paired t=1.46 p=1.4e-01
  throughput:      245 -> 243 /hr   per-scene paired t=-0.27 p=7.9e-01
  >= 4 placed:     66.4% -> 68.0%
  >= 5 placed:     42.2% -> 36.7%
  >= 6 placed:     10.2% -> 11.7%
```

## CONVERGENCE VERDICT — 2026-08-12 15:40 PDT: NULL, the claim SHIPS

vancow (released vanilla ckpt, fresh rollouts) vs bs128conv (fast-recipe ckpt),
same host (cow), same 128 canonical seeds, paired, paper rule:

    vancow 4.016 -> bs128conv 3.977   diff -0.039  t(127)=-0.25  p=0.80
    throughput 245 -> 243/hr (p=0.79)   >=6: 10.2% vs 11.7%

Pre-registered rule: |diff| <= 0.2 and n.s. -> ship with a +/-0.3 bound (n=128
paired SE ~0.15). Outcome is dead-center null; secondaries agree; vancow
absolute matches campaign draws (3.885-3.977).

SHIPPED CLAIM: pi0 fine-tune 5h12m -> 3h10m (1.66x) at equal policy quality
(bounded +/-0.3 bottles/scene), via bs 32->128 + dots remat + 0.93 pool +
decay rescaled to 7500 + peak lr unchanged. Seed caveat: one recipe draw vs one
released draw; the lr2x/seed-1 arm (training) adds a second recipe draw, and the
batched runner makes tighter n cheap.
