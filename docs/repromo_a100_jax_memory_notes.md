# Repromo RABC A100/JAX Memory Notes

Date: 2026-06-18

## Machine Summary

This machine has 8x NVIDIA A100-SXM4-80GB GPUs.

- Driver: 595.71.05
- CUDA reported by `nvidia-smi`: 13.2
- GPU memory per card: 81920 MiB
- MIG: disabled
- `openpi` Python: 3.11.15
- `jax`: 0.5.3
- `jaxlib`: 0.5.3

No JAX/XLA memory environment overrides were set in the shell:

```bash
CUDA_VISIBLE_DEVICES=<unset>
XLA_PYTHON_CLIENT_MEM_FRACTION=<unset>
XLA_PYTHON_CLIENT_PREALLOCATE=<unset>
XLA_PYTHON_CLIENT_ALLOCATOR=<unset>
JAX_PLATFORM_NAME=<unset>
XLA_FLAGS=<unset>
JAX_PLATFORMS=<unset>
```

## Launcher Behavior

`scripts/launch_repromo_rabc_runs.sh` launches one independent training process per GPU by setting `CUDA_VISIBLE_DEVICES` for each subprocess.

Dry-run output showed:

```bash
hang_mug      -> CUDA_VISIBLE_DEVICES=0
load_plates   -> CUDA_VISIBLE_DEVICES=1
put_bottles   -> CUDA_VISIBLE_DEVICES=2
sweep_paper   -> CUDA_VISIBLE_DEVICES=3
throw_bottles -> CUDA_VISIBLE_DEVICES=4
turn_mug      -> CUDA_VISIBLE_DEVICES=5
```

JAX reports each subprocess device as `CudaDevice(id=0)` because each process only sees one GPU. This is logical device 0 within that process, not necessarily physical GPU 0.

Probe:

```bash
CUDA_VISIBLE_DEVICES=3 uv run python - <<'PY'
import os
import jax
print(os.environ["CUDA_VISIBLE_DEVICES"])
print(jax.devices())
PY
```

Output:

```text
3
[CudaDevice(id=0)]
```

`nvidia-smi --query-compute-apps` confirmed the batch-size-16 run was spread across physical GPUs 0-5 with separate Python PIDs.

## Observed Failure At Batch Size 32

The predefined `pi0_sim_*_rabc_finalaction_thr100_nomax` configs use:

```python
batch_size=32
num_train_steps=60_000
save_interval=30_000
keep_period=30_000
```

When launched as one independent process per GPU with batch size 32, all six RABC runs failed on the first train step after restoring `pi0_base`.

Common log signature:

```text
Finished restoring checkpoint ... pi0_base/params
Allocator (GPU_0_bfc) ran out of memory trying to allocate 17.90GiB
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 19216000656 bytes
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED
```

Example log directory:

```text
logs/repromo_rabc_20260618_090817
```

This was not the earlier Repromo data-column issue. The injected datasets had already been verified to contain:

- `repromo_progress`
- `repromo_signed_magnitude`
- `repromo_quality`

## JAX Preallocation Observation

Plain JAX initialization/processes on this machine reserve approximately 61430 MiB per A100, about 75% of the 81920 MiB card.

During the active batch-size-16 relaunch, `nvidia-smi` showed:

```text
GPU 0: 61437 MiB / 81920 MiB
GPU 1: 61437 MiB / 81920 MiB
GPU 2: 61437 MiB / 81920 MiB
GPU 3: 61437 MiB / 81920 MiB
GPU 4: 61437 MiB / 81920 MiB
GPU 5: 61437 MiB / 81920 MiB
```

This is consistent with default JAX/XLA GPU memory preallocation. The batch-size-32 failure appears to happen because the train step needs an additional large temporary allocation inside/around this memory pool.

## Root Cause (Resolved)

The `61437 MiB` reservation is not arbitrary: `0.75 * 81920 = 61440`, i.e. JAX's
**default** `XLA_PYTHON_CLIENT_MEM_FRACTION = 0.75`. With preallocation on (also
default), JAX carves out a ~61 GiB BFC pool and will not grow past it, leaving
the remaining ~20 GiB of the 80 GB card unused. At batch size 32 the first train
step needs another ~17.9 GiB on top of params + optimizer + activations, which
the capped pool cannot satisfy -> `RESOURCE_EXHAUSTED`, even though `nvidia-smi`
shows the card only ~75% full.

This is **not** a hardware difference between the two A100-80GB clusters, and
**not** an FSDP difference (configs default `fsdp_devices=1`, and the launcher
pins one GPU per process, so nothing is sharded either way). It is purely the
launch path:

- The working cluster launches via SkyPilot. `sky_utils.generate_sky_config`
  defaults `xla_mem_fraction=0.95` and exports it before `train.py`, giving a
  ~78 GiB pool that fits batch size 32.
- This machine launched via `scripts/launch_repromo_rabc_runs.sh`, which did not
  set `XLA_PYTHON_CLIENT_MEM_FRACTION` at all, so JAX fell back to 0.75.

**Fix applied:** `scripts/launch_repromo_rabc_runs.sh` now exports
`XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"`,
matching the SkyPilot path. Batch size 32 no longer needs the manual env prefix.
An explicit env override is still respected if you need to lower the fraction
when co-scheduling other processes on a card.

## Historical Workaround (superseded by the fix above)

Before the root cause was understood, the relaunch dropped to batch size 16:

```bash
CHECKPOINT_BASE_DIR=/mnt/data/openpi_checkpoints \
scripts/launch_repromo_rabc_runs.sh \
  --overwrite \
  --batch-size 16 \
  --save-interval 30000 \
  --keep-period 60000
```

Those six batch-size-16 processes were alive and mapped across physical GPUs
0-5. Lowering the batch size is no longer necessary now that the launcher sets
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

## Batch-Size-32 Launch

With the fix in place, the default batch size 32 runs directly (no env prefix
needed):

```bash
CHECKPOINT_BASE_DIR=/mnt/data/openpi_checkpoints \
scripts/launch_repromo_rabc_runs.sh \
  --overwrite \
  --save-interval 30000 \
  --keep-period 60000
```

The launcher uses `0.95` of each card. Lower it via an explicit env override if
other processes share a GPU; here each run gets a dedicated GPU, so it is safe.

## Comparison Commands For Other Machine

Run these in the other `openpi` checkout:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader,nounits
env | rg 'XLA|JAX|CUDA|TF_FORCE_GPU_ALLOW_GROWTH|MEM_FRACTION|PREALLOCATE'
uv run python - <<'PY'
import os, sys
import jax, jaxlib
print('python', sys.version.replace('\n', ' '))
print('jax', jax.__version__)
print('jaxlib', jaxlib.__version__)
print('devices', jax.devices())
for key in [
    'CUDA_VISIBLE_DEVICES',
    'XLA_PYTHON_CLIENT_MEM_FRACTION',
    'XLA_PYTHON_CLIENT_PREALLOCATE',
    'XLA_PYTHON_CLIENT_ALLOCATOR',
    'JAX_PLATFORM_NAME',
    'XLA_FLAGS',
    'JAX_PLATFORMS',
]:
    print(f'{key}={os.environ.get(key, "<unset>")}')
PY
```

To verify device remapping:

```bash
for gpu in 0 1 2; do
  echo "=== CUDA_VISIBLE_DEVICES=$gpu ==="
  CUDA_VISIBLE_DEVICES=$gpu uv run python - <<'PY'
import os
import jax
print('env CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('jax devices=', jax.devices())
PY
done
```
