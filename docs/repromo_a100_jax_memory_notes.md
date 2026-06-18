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

## Current Workaround

The batch-size-16 relaunch was started with:

```bash
CHECKPOINT_BASE_DIR=/mnt/data/openpi_checkpoints \
scripts/launch_repromo_rabc_runs.sh \
  --overwrite \
  --batch-size 16 \
  --save-interval 30000 \
  --keep-period 60000
```

At the time of these notes, those six batch-size-16 processes were alive and mapped across physical GPUs 0-5.

## Hypothesis For Difference From Other A100 Machine

If batch size 32 works on another A100-80GB machine, likely differences to compare:

- `XLA_PYTHON_CLIENT_MEM_FRACTION`
- `XLA_PYTHON_CLIENT_PREALLOCATE`
- `XLA_PYTHON_CLIENT_ALLOCATOR`
- `XLA_FLAGS`
- `jax` / `jaxlib` versions
- NVIDIA driver and CUDA runtime reported by `nvidia-smi`
- Whether the other launch uses one process per GPU or one distributed run split across multiple GPUs

A plausible batch-size-32 retry on this machine would be:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CHECKPOINT_BASE_DIR=/mnt/data/openpi_checkpoints \
scripts/launch_repromo_rabc_runs.sh \
  --overwrite \
  --save-interval 30000 \
  --keep-period 60000
```

This should be tested carefully because it leaves less unreserved memory for other GPU users/processes.

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
