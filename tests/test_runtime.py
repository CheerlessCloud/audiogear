"""Unit tests for the runtime helpers used by every metric block."""

import os

import pytest

from audiogear.utils.runtime import (
    is_oom_error,
    length_buckets,
    normalize_device,
    resolve_num_threads,
)


# --- resolve_num_threads --------------------------------------------------
@pytest.mark.parametrize("sentinel", [None, 0, -1])
def test_resolve_num_threads_sentinel_means_all_cores(sentinel):
    assert resolve_num_threads(sentinel) == (os.cpu_count() or 1)


def test_resolve_num_threads_explicit():
    assert resolve_num_threads(4) == 4


# --- normalize_device -----------------------------------------------------
@pytest.mark.parametrize(
    "spec,expected",
    [("cuda", "cuda"), ("cuda:1", "cuda"), ("cpu", "cpu"), ("CPU?", "cpu")],
)
def test_normalize_device(spec, expected):
    assert normalize_device(spec) == expected


# --- is_oom_error ---------------------------------------------------------
@pytest.mark.parametrize(
    "msg",
    [
        "CUDA out of memory. Tried to allocate ...",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cudnn_status_alloc_failed",
        "Failed to allocate memory",
    ],
)
def test_is_oom_error_detects_allocator_failures(msg):
    assert is_oom_error(RuntimeError(msg)) is True


def test_is_oom_error_ignores_real_errors():
    assert is_oom_error(ValueError("bad shape")) is False
    assert is_oom_error(KeyError("missing")) is False


# --- length_buckets -------------------------------------------------------
def test_length_buckets_covers_every_index_exactly_once():
    durations = [5.0, 1.0, 3.0, 2.0, 8.0, 0.5]
    buckets = length_buckets(durations, max_batch_seconds=10.0, max_batch_size=4)
    flat = sorted(i for b in buckets for i in b)
    assert flat == list(range(len(durations)))


def test_length_buckets_respects_size_cap():
    durations = [1.0] * 10
    buckets = length_buckets(durations, max_batch_seconds=10_000, max_batch_size=3)
    assert all(len(b) <= 3 for b in buckets)


def test_length_buckets_respects_seconds_budget():
    # padded cost = len(batch) * max_duration must stay <= budget
    durations = [4.0, 4.0, 4.0]
    buckets = length_buckets(durations, max_batch_seconds=8.0, max_batch_size=100)
    for b in buckets:
        assert len(b) * max(durations[i] for i in b) <= 8.0


def test_length_buckets_oversized_clip_gets_its_own_batch():
    durations = [100.0, 1.0, 1.0]
    buckets = length_buckets(durations, max_batch_seconds=10.0, max_batch_size=8)
    big = next(b for b in buckets if 0 in b)
    assert big == [0]


def test_iter_windows_splits_long_clip():
    torch = pytest.importorskip("torch")
    from audiogear.utils.runtime import iter_windows

    wav = torch.arange(10, dtype=torch.float32).reshape(1, 10)  # 10 samples
    windows = list(iter_windows(wav, sample_rate=1, seconds=3))  # win = 3 samples
    # 10 samples / 3 -> 4 chunks (3,3,3,1) and they reconstruct the original
    assert [w.shape[-1] for w in windows] == [3, 3, 3, 1]
    assert torch.cat(windows, dim=-1).tolist() == wav.tolist()


def test_iter_windows_short_clip_is_single_window():
    torch = pytest.importorskip("torch")
    from audiogear.utils.runtime import iter_windows

    wav = torch.zeros(1, 5)
    assert len(list(iter_windows(wav, sample_rate=1, seconds=10))) == 1
