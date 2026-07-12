"""Per-metric intra-shard resume checkpoints.

The executor's completion markers are shard-granular: a crash (host OOM, dead
CUDA driver, kill) at 90% of a big shard used to force a full recompute of the
whole shard, hours of GPU time on the million-clip sets. With a checkpoint
attached, every metric appends one JSONL line per finished clip to
``<checkpoint_folder>/<slug>/<rank>.jsonl``; on restart it reads the file back,
fills the cached values into the segments' metadata and computes only the rest.

JSONL, not CSV, so numeric types survive the round-trip — downstream metrics
consume upstream columns as numbers (StyleMetric reads ``pitch_mean``), and a
string would silently break them. Sentinel rows (``NaN``/``-1`` for clips the
guard gave up on) are checkpointed too: a corrupt clip stays corrupt, retrying
it on every resume would just burn time.

Lane-safe by construction: lanes already require disjoint columns per metric,
each metric owns its own file, and all appends happen on the metric's main
thread. Rank-safe: one file per rank, and ranks never share a process
concurrently.

Staleness: rows are keyed by clip id and the slug only encodes the metric class
and its first output column — changing a metric's *parameters* (model, backend,
thresholds) does not invalidate them. Delete the ``checkpoints/`` folder when
re-running with a changed metric config.
"""

from __future__ import annotations

import json

from loguru import logger

from audiogear.io import DataFolderLike, get_datafolder


class MetricCheckpoint:
    def __init__(self, folder: DataFolderLike, slug: str, rank: int):
        self.data_folder = get_datafolder(folder)
        self.path = f"{slug}/{rank:05d}.jsonl"
        self._handle = None

    def load(self) -> dict[str, dict]:
        """Read back ``{clip id -> cached column values}``.

        A line torn by a crash mid-write (or any other corruption) is skipped —
        that clip is simply recomputed."""
        if not self.data_folder.isfile(self.path):
            return {}
        cache: dict[str, dict] = {}
        skipped = 0
        with self.data_folder.open(self.path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    cache[str(row.pop("id"))] = row
                except (ValueError, KeyError, TypeError, AttributeError):
                    skipped += 1
        if skipped:
            logger.warning(f"checkpoint {self.path}: skipped {skipped} corrupt line(s)")
        return cache

    def append(self, segment_id, values: dict) -> None:
        if self._handle is None:
            self._handle = self.data_folder.open(self.path, "at")
        # default=str: numpy scalars and other exotica degrade to strings rather
        # than fail; allow_nan (default) keeps the guard's NaN sentinels intact.
        self._handle.write(json.dumps({"id": segment_id, **values}, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()  # a crash may lose at most the line being written

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
