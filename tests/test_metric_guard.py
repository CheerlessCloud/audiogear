"""The centralized per-clip failure guard in ``BaseMetric``.

One corrupt clip used to kill an entire shard (a single bad mp3 took down a
549k-clip dataset; a 2 ms clip crashed SQUIM and 1.1M clips with it). These
model-free fakes poison individual clips on every scheduling path (serial,
parallel CPU, plain GPU, batched, prefetch) and assert the poisoned clip gets a
sentinel while every other clip still receives its own value. The abort
threshold (``max_consecutive_failures``) must still trip on an unbroken failure
streak — that is systematic breakage, not bad data.
"""

import math

import pytest

from audiogear.data import AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.metrics.base import BaseMetric, PrefetchGPUMetric
from audiogear.pipeline.parallel import ParallelLanes

BAD_IDS = {"3", "7"}


def _segments(n):
    return [
        AudioSegment(id=str(i), audio_file=f"{i}.wav", format="wav", duration=float(n - i), metadata={"x": i})
        for i in range(n)
    ]


def _assert_guarded(segments):
    for seg in segments:
        val = seg.metadata["val"]
        if seg.id in BAD_IDS:
            assert isinstance(val, float) and math.isnan(val), f"id={seg.id} should carry the sentinel"
        else:
            assert val == seg.metadata["x"] * 10, f"desync/loss at id={seg.id}"


class _FakeSerial(BaseMetric):
    def __init__(self, **kw):
        super().__init__(metric="val", **kw)

    def compute_metric(self, segment):
        if segment.id in BAD_IDS:
            raise ValueError("corrupt clip")
        return segment.metadata["x"] * 10


class _FakeParallelCPU(_FakeSerial):
    parallel_cpu = True

    def __init__(self):
        super().__init__()
        self.num_threads = 4


class _FakePlainGPU(_FakeSerial):
    gpu = True


class _FakeBatched(BaseMetric):
    gpu = True
    supports_batch = True

    def __init__(self):
        super().__init__(metric="val", batch_size=4, max_batch_seconds=10_000)

    def compute_metric(self, segment):
        if segment.id in BAD_IDS:
            raise ValueError("corrupt clip")
        return segment.metadata["x"] * 10

    def compute_batch(self, segments):
        # one poisoned clip fails the whole forward — like a corrupt file in a
        # padded batch — so the guard must retry the batch clip by clip
        return [self.compute_metric(s) for s in segments]


class _FakePrefetchBadDecode(PrefetchGPUMetric):
    """Failure in ``_prepare`` (decode thread) — surfaces via ``fut.result()``."""

    def __init__(self):
        super().__init__(metric="val", device="cpu")
        self.num_threads = 4

    def _prepare(self, segment):
        if segment.id in BAD_IDS:
            raise ValueError("undecodable audio")
        return segment.metadata["x"]

    def _infer(self, prepared):
        return prepared * 10


class _FakePrefetchBadInfer(PrefetchGPUMetric):
    """Failure in ``_infer`` (model forward on one clip)."""

    def __init__(self):
        super().__init__(metric="val", device="cpu")
        self.num_threads = 4

    def _prepare(self, segment):
        return segment

    def _infer(self, segment):
        if segment.id in BAD_IDS:
            raise RuntimeError("model choked on this input")
        return segment.metadata["x"] * 10

    def compute_metric(self, segment):
        return self._infer(segment)


@pytest.mark.parametrize(
    "metric_factory",
    [_FakeSerial, _FakeParallelCPU, _FakePlainGPU, _FakeBatched, _FakePrefetchBadDecode, _FakePrefetchBadInfer],
)
def test_one_bad_clip_never_kills_the_shard(metric_factory):
    segs = _segments(20)
    metric_factory().run(segs)
    _assert_guarded(segs)


def test_tuple_metric_gets_tuple_sentinel():
    class _Tuple(_FakeSerial):
        def __init__(self):
            BaseMetric.__init__(self, metric=("a", "b"))

        def compute_metric(self, segment):
            if segment.id in BAD_IDS:
                raise ValueError("corrupt clip")
            return segment.metadata["x"], segment.metadata["x"] * 10

    segs = _segments(10)
    _Tuple().run(segs)
    bad = next(s for s in segs if s.id in BAD_IDS)
    assert math.isnan(bad.metadata["a"]) and math.isnan(bad.metadata["b"])
    good = next(s for s in segs if s.id not in BAD_IDS)
    assert good.metadata["a"] == good.metadata["x"]


def test_unbroken_failure_streak_aborts_the_shard():
    class _AlwaysFails(BaseMetric):
        def __init__(self):
            super().__init__(metric="val", max_consecutive_failures=5)

        def compute_metric(self, segment):
            raise RuntimeError("model never loaded")

    with pytest.raises(RuntimeError, match="failed in an unbroken"):
        _AlwaysFails().run(_segments(20))


def test_streak_resets_on_success_and_failures_are_counted():
    class _EveryOtherFails(BaseMetric):
        def __init__(self):
            super().__init__(metric="val", max_consecutive_failures=2)

        def compute_metric(self, segment):
            if int(segment.id) % 2:
                raise ValueError("bad clip")
            return 1.0

    metric = _EveryOtherFails()
    segs = _segments(20)
    metric.run(segs)  # alternating failures never reach the streak threshold
    assert metric._failures == 10
    assert sum(math.isnan(s.metadata["val"]) for s in segs) == 10


class _NoopStep(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        for seg in data:
            seg.metadata["noop"] = True
        return data


class _BoomStep(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        raise RuntimeError("lane blew up")


def test_failed_lane_is_reported_by_name_and_others_still_finish():
    segs = _segments(5)
    lanes = ParallelLanes({"good": [_NoopStep()], "bad": [_BoomStep()]})
    with pytest.raises(RuntimeError, match="bad"):
        lanes.run(segs)
    assert all(s.metadata.get("noop") for s in segs), "healthy lane must run to completion"


def test_lanes_aggregate_declared_output_columns():
    lanes = ParallelLanes({"a": [_FakeSerial()], "b": [_FakeBatched()]})
    assert lanes.output_columns == ("val", "val")
