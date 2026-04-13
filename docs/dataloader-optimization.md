# Dataloader Optimization for Video-Based LeRobot Training

Notes on getting high GPU utilization for Pi0/Pi0.5 training on large video datasets.

## Problem

Default LeRobot video loading was ~5.2s per batch (bs=32), starving the GPU. Root causes:
1. Sparse H.264 keyframes (1 I-frame per ~8 seconds / 236 frames)
2. pyav backend (slow sequential decode from keyframe)
3. `num_workers=0` (all decode in main process)

## Solution: Three optimizations, ~150x combined speedup

### 1. Dense Keyframes (5x speedup)

H.264 random access requires decoding from the nearest I-frame. With sparse GOP (default libx264), seeking to an arbitrary frame means decoding up to 236 intermediate frames.

Re-encode with dense keyframes:
```bash
python scripts/data/reencode_dense_keyframes.py \
    --data-dir /path/to/lerobot_dataset \
    --gop 10 --workers 32
```

This sets `-g 10 -bf 0` (keyframe every 10 frames, no B-frames). Worst-case seek drops from 236 to 10 frame decodes. File size is unchanged or slightly smaller.

**Before:** 57ms/camera (sparse GOP, pyav)
**After:** 12ms/camera (dense GOP, pyav)

### 2. torchcodec Backend (7.5x over pyav)

torchcodec uses FFmpeg's native seeking which is much faster than pyav's sequential decode. LeRobot auto-selects it when available.

**The anaconda problem:** If your Python comes from anaconda, torchcodec fails to load because anaconda ships old `libstdc++.so.6` (GLIBCXX_3.4.26) but system FFmpeg 6 needs GLIBCXX_3.4.32. The error looks like:
```
libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by libavfilter.so.9)
```

**Fix:** Rebuild the venv with a non-anaconda Python:
```bash
# Comment out conda init in ~/.bashrc first, then:
uv python install 3.11
rm -rf .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync --python ~/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
```

If you can't remove anaconda, use `LD_PRELOAD` as a workaround:
```bash
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0" \
uv run python scripts/train.py ...
```

**torchcodec vs pyav (same sparse GOP video):**
| Backend | Mean (1 camera, 10 frames) |
|---|---|
| pyav | 46ms |
| torchcodec | 6ms |

### 3. num_workers=8 (8x parallelism)

Default `num_workers=0` decodes all video in the main process. Set `num_workers=8` in the train config to parallelize across DataLoader workers.

### Combined Results

| Setup | Per-camera latency | Batch time (bs=32, 3 cams) |
|---|---|---|
| Baseline (sparse GOP, pyav, 0 workers) | 57ms | ~5.5s |
| + Dense keyframes | 12ms | ~1.2s |
| + num_workers=8 | 12ms / 8 | ~0.14s |
| + torchcodec | 6ms / 8 | ~0.07s |
| + torchcodec + dense GOP | ~3ms / 8 | ~0.04s |

## Dataset Preparation Checklist

1. **Convert to LeRobot format** with RORM weights:
   ```bash
   python RORM/scripts/data/convert_to_lerobot.py \
       --data-dirs /path/to/episodes \
       --checkpoint /path/to/rorm_checkpoint.pt
   ```

2. **Fix timestamps** — must be `frame_index / fps`, not epoch timestamps. The convert script handles this now.

3. **Re-encode with dense keyframes:**
   ```bash
   python RORM/scripts/data/reencode_dense_keyframes.py \
       --data-dir /path/to/lerobot_dataset --gop 10 --workers 32
   ```

4. **Symlink into HF cache** so LeRobot finds it by repo name:
   ```bash
   ln -sfn /path/to/lerobot_dataset ~/.cache/huggingface/lerobot/my_dataset_name
   ```

5. **Compute norm stats:**
   ```bash
   uv run python scripts/compute_norm_stats.py --config-name my_config --max-frames 50000
   ```

6. **Set tolerance** in `data_loader.py` if videos have slight FPS mismatch:
   ```python
   tolerance_s=0.04  # 40ms, covers ~1 frame at 30fps
   ```

## Training Launch

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CUDA_VISIBLE_DEVICES=1 \
uv run python scripts/train.py pi0_yam_tshirt_no_rabc \
    --exp-name my_experiment --batch-size 8
```

Notes:
- Pi0 (3B params) with action_horizon=30 needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- batch_size=8 fits on A100-80GB ONE GPU; batch_size=16+ OOMs. With more free GPUs (a node with 8 GPUs) we can get larger batch sizes
- First run downloads pi0_base checkpoint from S3 (~3GB, one-time)
- First run builds HF datasets arrow cache (~10 min for 17M frames, one-time)
