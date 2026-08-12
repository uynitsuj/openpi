# pi0 training speedup — end-to-end explanation

*What was implemented, what was measured, and why the winning configuration works.
Written for a reader who knows forward/backward passes but not this model or stack.*

Objective (2026-08-11): reduce pi0 training wall-clock as much as possible.
Test config: `pi0_put_bottles_mjwarp_no_rabc` (vanilla behavior cloning on sim bottles).
Hardware: `protective-crimson-cow` — 8×A100-80GB, 200 CPU cores, 925 GB RAM, NVMe scratch.

**Headline: 52.4 → 86.9 samples/s (1.66×), so the recipe's 5.1 h becomes 3.1 h —
pending one convergence validation that is training now.**

---

## 1. The model and the training step, in five sentences

pi0 is a vision-language-action policy of roughly 3B parameters: a **SigLIP** vision
encoder (~400M) turns each camera image into 256 tokens, a **Gemma** language model
(~2.6B) consumes those tokens plus a text prompt, and a small **action expert** (~300M)
attached to Gemma emits a chunk of robot actions via flow matching. One training sample
is ~850 tokens through the big transformer. The loss is behavior cloning: match the
demonstrated action chunk. Training runs in JAX: the whole step (forward, backward,
optimizer) is compiled by XLA into one graph and launched onto the GPUs. The 8 GPUs
form a mesh — the batch is split across all 8, and the parameters + optimizer state
are split ("sharded") across pairs (`fsdp-devices 2`), because the fp32 parameters,
two Adam moment tensors, and an EMA copy total ~53 GB and do not fit comfortably
replicated on one card.

The recipe: global batch 32 (= **4 samples per GPU**), 30,000 steps, ~5.1 hours.

## 2. What was implemented

All changes are **environment-variable-gated and default-off**: with no env vars set,
the code path is byte-identical to stock. Originals are saved as `*.orig`.

### 2.1 Profiling hooks in `scripts/train.py`

| env var | effect |
|---|---|
| `OPENPI_PROFILE_STEPS=N` | time exactly N steps after `OPENPI_PROFILE_WARMUP` (default 20) warmup steps, print one `[PROFILE] ... it/s samples/s` line, exit |
| `OPENPI_PROFILE_SYNTHETIC=1` | never fetch a new batch — reuse the first one forever |
| `OPENPI_PROFILE_LOADER=1` | never run the training step — only pull batches |
| `OPENPI_PROFILE_TRACE=dir` | capture a `jax.profiler` trace of 8 post-warmup steps |

The timer is fenced with `jax.block_until_ready` on both ends. JAX dispatches work
asynchronously — without the fence you time how fast Python *enqueues* steps, not how
fast the GPU *runs* them. Warmup matters because the first steps include XLA
compilation (minutes).

Why these three modes: they decompose the loop.

```
baseline   = compute + any loader time not hidden by overlap
synthetic  = pure compute            (loader removed)
loader     = pure loader             (compute removed)
```

`baseline − synthetic` = what the dataloader actually costs.
`loader` alone = the ceiling the loader could ever sustain.

### 2.2 Remat policy override in `gemma.py` / `siglip.py`

The stock code hardcodes `jax.checkpoint_policies.nothing_saveable` on every
transformer block. One-line change: read the policy name from `OPENPI_REMAT_POLICY`,
defaulting to the original. (Section 4 explains what this does.)

### 2.3 Measurement discipline (learned the hard way)

- `python -m py_compile` only checks syntax. `train.py` imported neither `os` nor
  `time`; the patch would have crashed at step 1. Every patch is now followed by an
  **AST walk asserting the names it uses are actually imported**. This trap fired
  three separate times across three files.
- GPU utilization sampled with `nvidia-smi -l 5` across a whole run is diluted by
  minutes of setup at 0%. The sampler now reports the **tail 40% window** only.
- Throughput at different batch sizes is compared in **samples/s**, never it/s.

### 2.4 Trace analysis

A parser for the perfetto JSON that JAX's profiler emits: merges overlapping op
intervals per GPU stream (naive summing double-counts nested events), reports busy
fraction per stream and top ops by total time.

## 3. Finding 1 — the dataloader costs nothing here

```
baseline  (real loader):   52.43 samples/s
synthetic (no loader):     52.60 samples/s      → loader cost: 0.3%
loader alone:             215.50 samples/s      → 4.1× headroom
```

Why it is free on this box: (1) the training loop fetches the *next* batch while the
GPU still runs the *current* step — JAX's async dispatch gives pipelining without any
explicit code; (2) 8 decode workers on 200 cores; (3) the 21 GB dataset fits in page
cache after one epoch, so video "seeks" hit RAM.

Consequence: any custom data format (stacked frames, GOP tricks) buys ~0.3% here.
That work is **gated** behind compute exceeding ~215 samples/s, or behind hardware
where the three conditions above fail (small RAM, few cores, S3 streaming).

## 4. Finding 2 — where memory actually goes, and what "remat" is

The backward pass needs the forward's intermediate activations. Two ways to have them:
**store** during forward (memory) or **recompute** during backward (a second forward's
worth of FLOPs). JAX calls the recompute option *rematerialization* and lets a
**policy** decide, per intermediate value, which side it falls on:

| policy | keeps | memory | extra compute |
|---|---|---|---|
| `everything_saveable` | all activations | enormous | none |
| `dots_with_no_batch_dims_saveable` | matmul outputs only | moderate | small |
| `nothing_saveable` *(stock)* | ~nothing | minimal | ≈ a full extra forward |

"Dots" are matmul results (`dot_general` in XLA). Saving exactly those is the classic
sweet spot: matmuls are the expensive ops (recomputing one doubles its cost) and their
outputs are what weight gradients need anyway; the cheap glue between them (layernorms,
GELUs, softmax) is re-derived on the fly.

Measured consequences (the OOM column is the memory a normal
store-everything stack would have needed):

```
bs64   everything_saveable:  OOM  (+76 GB over the pool)
bs128  everything_saveable:  OOM  (+146 GB)
bs256  everything_saveable:  OOM  (+266 GB)
bs256  nothing_saveable:     fits in 60 GB
```

This answers "how is batch 128 even possible?": the stock `nothing_saveable` makes
activation memory almost flat in batch size. Large batch was always possible — it was
the *default pool size* that hid it (next section).

## 5. Finding 3 — the invisible 60 GB pool

Every run, including the OOMs, showed max GPU memory ≈ 62 GB. That is not the card;
it is `XLA_PYTHON_CLIENT_MEM_FRACTION` defaulting to **0.75**, so XLA's allocator owns
60 GB of each 80 GB A100. Raising it to 0.93 (74 GB) recovered 14 GB per card — enough
to fit the middle remat policy at batch 128, which plain 60 GB could not.

## 6. Finding 4 — batch scaling and the step-time decomposition

```
global batch:    32     64     128    256
samples/s:      52.4   67.3   77.0   86.5      (stock remat, 60 GB pool)
```

Fitting step time = fixed + variable·batch: **~0.28 s fixed per step** (45% of a bs32
step!) plus ~10.5 ms per sample → an asymptote near ~95 samples/s. The recipe's 4
samples/GPU mostly measures the fixed cost; large batch amortizes it.

The winning combination stacks the three levers:

```
bs128 + dots remat + 0.93 pool  =  86.9 samples/s  =  1.66× recipe
(bs192 with the same settings OOMs — this is the edge of the card)
```

## 7. The nulls (measured, so they never get re-tried blind)

| lever | result |
|---|---|
| CUDA-graph command buffers (`--xla_gpu_enable_command_buffer=...`) | **0.0%** at bs64 and bs128 |
| EMA off (skips a 3B-param update per step) | +1.5% |
| `fsdp-devices 1` (full replication) | OOM at bs128 |
| relaxing remat *without* the bigger pool | OOM |

The command-buffer null matters because the profiler trace shows **~11,000 kernel
launches per step** with the GPU compute stream only **61.7% busy** — the natural
hypothesis was launch-gap overhead, and batching launches should have fixed it. It
did not, so the ~38% idle is scheduling/collective structure, not launch cost. On the
trace, NCCL collectives (FSDP all-gathers + gradient all-reduces) take ~8.3% of stream
time, GEMMs ~20-25%, and a long tail of small fused kernels the rest.

## 8. Open lead found in the training log

XLA prints, from inside the per-layer scan loop:

```
[spmd] Involuntary full rematerialization ... sharding {devices=[8,1,1]}
to {devices=[1,1,2,4]} ... "enrich the sharding annotations"
```

Two sharding annotations disagree, forcing an extra tensor rematerialization every
layer, every step. This is a sharding-annotation fix in openpi (not kernel work) and
is the best current suspect for part of the 38% idle. Not yet attempted.

## 9. The convergence validation (running now)

1.66× more samples/s is a **throughput** claim. The **training-time** claim requires
that batch 128 reaches the same policy quality on the same data budget. The check:

- 7,500 steps × 128 = 960,000 samples = exactly the recipe's 30,000 × 32.
- Cosine lr decay rescaled to 7,500 steps (otherwise the schedule never finishes and
  the comparison is rigged against the new recipe).
- Peak lr **unchanged** — the conservative choice for Adam at 4× batch; an lr-scaled
  arm is the pre-planned follow-up if this lands short.
- Verdict comes from **closed-loop evaluation** (bottles placed per scene), not the
  loss curve: on this task three seeds finished within 0.0009 loss of each other and
  spanned 0.275 bottles/scene closed-loop. Loss certifies optimization health only.

What "4× fewer optimizer steps" trades: each Adam update sees a 4× less noisy
gradient, but there are only 7,500 of them. Sometimes that is free; sometimes it costs
a bit of final quality; occasionally it needs a higher peak lr. That is the open
question, and it is empirical.

## 10. Status and what ships where

| item | state |
|---|---|
| bs128 convergence run | training on cow; checkpoint ~00:30, closed-loop eval after |
| fork branch (`uynitsuj/openpi`) | assembled after the verdict: config-plumbed remat policy, pool fraction, profiling hooks, this log — measured wins only |
| sharding-annotation fix (§8) | next investigation |
| custom data-format repo | parked behind the measured 215 samples/s gate; design sketch exists for the RAM-poor / core-poor / S3 cases |

All raw numbers: `PROFILE_LOG.md` (this directory, mirrored on cow at
`/scratch/warprm/PROFILE_LOG.md`).
