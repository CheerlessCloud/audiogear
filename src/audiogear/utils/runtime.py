"""Runtime helpers shared by metric blocks: CPU-thread sizing and CUDA-OOM
recovery.

Two concerns live here so every metric handles them the same way:

1. **CPU parallelism** — ``resolve_num_threads`` turns the ``-1``/``0``/``None``
   sentinel into "all available cores", so CPU-bound DSP metrics fan out across
   the whole machine by default (the user's "use all cores" request).

2. **CUDA OOM** — GPU metrics process one clip at a time; a pathologically long
   clip can blow up VRAM. ``is_oom_error`` / ``free_cuda`` let the base metric
   catch that *specific* failure, clear the cache, and retry with a degraded
   strategy (chunked GPU → CPU) instead of killing the whole shard.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import torch


class NotRecoverable(Exception):
    """Raised by a recovery hook that a metric does not implement / cannot apply.

    Signals the base ladder to drop to the next strategy (and ultimately a
    sentinel value) rather than propagating as a real error.
    """


# --- Process-global model cache -------------------------------------------
# Heavy models are cached at *module* (process) scope, NOT on the metric
# instance. The executor re-creates metric objects for every task (deepcopy for
# workers==1, re-pickle for workers>1), so instance-stored models reload on each
# of ``tasks`` shards — minutes of overhead × tasks (EXPERIENCE ❌5). Keying the
# cache here means a model loads once per *worker process* and is reused across
# every shard that process handles. It also keeps metric instances cheap to
# deepcopy/pickle (they hold only config, never a CUDA model).
#
# GPU stability falls out for free: the first task in a process loads the model
# onto whatever GPU it was pinned to and the cached model stays there, so each
# worker process sticks to one GPU regardless of later rank shuffling.
_MODEL_CACHE: dict = {}


def cached_model(key, factory):
    """Return a process-global singleton model for ``key``, building it once.

    ``key`` must be a hashable identity (e.g. ``(ClassName, model_id, device)``)
    and ``factory`` a zero-arg callable that builds+moves the model. Loads happen
    after the worker's GPU is pinned, so ``device='cuda'`` lands on the right GPU.
    """
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = factory()
    return _MODEL_CACHE[key]


def length_buckets(durations: list[float], max_batch_seconds: float, max_batch_size: int) -> list[list[int]]:
    """Group clip indices into length-homogeneous, VRAM-bounded batches.

    Indices are sorted by duration so each batch holds similar-length clips
    (minimal zero-padding). A batch is flushed when adding the next clip would
    push ``len(batch) * max(duration in batch)`` past ``max_batch_seconds`` (a
    proxy for activation VRAM ≈ batch_size × padded_length) or exceed the hard
    ``max_batch_size`` cap. Returns lists of *original* indices.

    A single clip longer than the budget still forms its own (size-1) batch —
    the caller's OOM ladder handles it from there.
    """
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0.0
    for i in order:
        d = max(0.0, durations[i])
        new_max = max(cur_max, d)
        too_big = (len(cur) + 1) * new_max > max_batch_seconds
        too_many = len(cur) + 1 > max_batch_size
        if cur and (too_big or too_many):
            batches.append(cur)
            cur, cur_max = [], 0.0
            new_max = d
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def normalize_device(device) -> str:
    """Collapse a device spec to the bare ``"cuda"``/``"cpu"`` string that
    faster-whisper / gigaam / pyannote expect (they don't take ``"cuda:1"`` —
    the executor already pins one GPU per worker via CUDA_VISIBLE_DEVICES)."""
    return "cuda" if str(device).startswith("cuda") else "cpu"


def resolve_num_threads(num_threads: int | None) -> int:
    """Interpret a thread count, treating ``None``/``<= 0`` as "all cores".

    ``os.cpu_count()`` can return ``None`` in exotic environments; fall back to 1.
    """
    if num_threads is None or num_threads <= 0:
        return os.cpu_count() or 1
    return num_threads


# Substrings that mark a CUDA allocator failure across torch / CTranslate2 /
# cuDNN / cuBLAS. Matched case-insensitively against the exception message so we
# only intercept *memory* failures, never genuine model/logic errors.
_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "failed to allocate",
)


def is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory failure (any framework)."""
    # torch's dedicated subclass is the cleanest signal when present.
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # torch missing / too old to expose the class
        pass
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


def free_cuda() -> None:
    """Best-effort release of cached CUDA memory after an OOM.

    ``empty_cache`` returns the allocator's unused blocks to the driver and
    ``ipc_collect`` reaps cross-process handles; together they give the retry a
    clean slate. No-op if torch/CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - defensive
        pass


def iter_windows(
    waveform: "torch.Tensor", sample_rate: int, seconds: float
) -> Iterator["torch.Tensor"]:
    """Yield consecutive ``seconds``-long, non-overlapping windows of a clip.

    ``waveform`` is ``(channels, samples)`` (torchaudio convention). Used by the
    OOM ladder: a clip too long to fit whole is decoded window-by-window and the
    per-window metric values are aggregated. The trailing remainder is yielded
    as-is (it is shorter than a full window but still valid for these models).
    """
    win = max(1, int(seconds * sample_rate))
    total = waveform.shape[-1]
    if total <= win:
        yield waveform
        return
    for start in range(0, total, win):
        chunk = waveform[..., start : start + win]
        if chunk.shape[-1] == 0:
            continue
        yield chunk
