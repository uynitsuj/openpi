"""Monkey-patches for the bundled lerobot package.

Imported for side effects from openpi.training.data_loader so the patches
land before any LeRobotDataset / LeRobotDatasetMetadata is constructed.

Patch 1 — numeric file ordering in ``load_nested_dataset``
=========================================================
``lerobot.datasets.utils.load_nested_dataset`` does
``paths = sorted(pq_dir.glob("*/*.parquet"))`` which is *byte-level* lex sort.
For 3-digit zero-padded filenames (``file-000.parquet`` … ``file-999.parquet``)
plus 4-digit names (``file-1000.parquet`` …), the lex ordering puts the
4-digit files between ``file-100.parquet`` and ``file-101.parquet`` because
``.`` (0x2E) sorts before ``0`` (0x30). The HF dataset built from this
sequence has position ≠ episode_index for every row past ep 100 in datasets
with ≥1000 files, which corrupts ``meta.episodes[ep_idx]`` indexing and
ultimately makes ``_query_videos`` look up the wrong video file (out-of-bounds
``frame_index``). Affects sim_tasks_warp datasets (1141 files each); does not
affect tshirt / SARM (≤999 files). See memory ``lerobot-lex-sort-bug``.

The fix sorts by ``(parent_dir, int(file_index))`` so position == episode_index
whenever filenames follow the ``file-N(.parquet)`` convention. Falls back to
lex sort for non-matching names so behavior is unchanged on legacy datasets.
"""
from __future__ import annotations

import re

from lerobot.datasets import utils as _le_utils


def _numeric_sort_key(p):
    """Key for ``sorted`` that puts ``file-N.parquet`` in numeric order.

    Tuple: (parent path, parsed int, original filename). The parent component
    keeps chunk-000, chunk-001, … grouped together; the parsed int orders within
    a chunk; the original filename is a final tie-break for any non-conforming
    names.
    """
    m = re.search(r"file-(\d+)", p.name)
    return (str(p.parent), int(m.group(1)) if m else -1, p.name)


_original_load_nested_dataset = _le_utils.load_nested_dataset


def _load_nested_dataset_numeric_sort(pq_dir, features=None, episodes=None):
    """Replacement: identical to the original except files sort numerically."""
    import pyarrow.dataset as pa_ds
    from datasets import Dataset

    # Mirror upstream behavior, but with numeric sort.
    from pathlib import Path
    pq_dir = Path(pq_dir)
    paths = sorted(pq_dir.glob("*/*.parquet"), key=_numeric_sort_key)
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")

    # Upstream uses SuppressProgressBars from lerobot.datasets.utils; reuse it
    # so output stays identical.
    with _le_utils.SuppressProgressBars():
        filters = pa_ds.field("episode_index").isin(episodes) if episodes is not None else None
        return Dataset.from_parquet([str(p) for p in paths], filters=filters, features=features)


_le_utils.load_nested_dataset = _load_nested_dataset_numeric_sort

# ``lerobot_dataset`` does ``from .utils import load_nested_dataset`` at module
# scope, so reassigning the source binding above doesn't reach that consumer
# (its local name still points at the original function). Patch the local
# binding too. ``load_episodes`` in ``_le_utils`` calls the (now-patched)
# module-level name via ``_le_utils.load_nested_dataset``, so that side is
# already covered. If lerobot adds more consumers in future versions, patch
# them here as well.
from lerobot.datasets import lerobot_dataset as _le_dataset  # noqa: E402

_le_dataset.load_nested_dataset = _load_nested_dataset_numeric_sort

# Re-export so importers don't have to fish through lerobot.
load_nested_dataset = _load_nested_dataset_numeric_sort
