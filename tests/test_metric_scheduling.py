"""Result<->segment ordering across every BaseMetric execution path.

BUG-1's report suspected the prefetch path of mis-pairing results with segments.
These model-free fakes drive each scheduling path (prefetch, batched, plain GPU,
parallel CPU) and assert every segment receives the value computed from *its own*
input — i.e. the scheduling itself never desyncs. (The real corruption was in the
CSV writer, covered in test_csv_writer.py.)
"""

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric, PrefetchGPUMetric


def _segments(n):
    # metadata["x"] is the per-clip input; expected output is x * 10.
    return [AudioSegment(id=str(i), audio_file=f"{i}.wav", format="wav", duration=float(n - i), metadata={"x": i})
            for i in range(n)]


def _assert_paired(segments):
    for seg in segments:
        assert seg.metadata["val"] == seg.metadata["x"] * 10, f"desync at id={seg.id}"


class _FakePrefetch(PrefetchGPUMetric):
    def __init__(self):
        super().__init__(metric="val", device="cpu")
        self.num_threads = 4

    def _prepare(self, segment):
        return segment.metadata["x"]

    def _infer(self, prepared):
        return prepared * 10


class _FakePlainGPU(BaseMetric):
    gpu = True

    def __init__(self):
        super().__init__(metric="val")

    def compute_metric(self, segment):
        return segment.metadata["x"] * 10


class _FakeBatched(BaseMetric):
    gpu = True
    supports_batch = True

    def __init__(self):
        super().__init__(metric="val", batch_size=4, max_batch_seconds=10_000)

    def compute_metric(self, segment):  # abstract in BaseMetric; unused on the batched path
        return segment.metadata["x"] * 10

    def compute_batch(self, segments):
        return [s.metadata["x"] * 10 for s in segments]


class _FakeParallelCPU(BaseMetric):
    parallel_cpu = True

    def __init__(self):
        super().__init__(metric="val")
        self.num_threads = 4

    def compute_metric(self, segment):
        return segment.metadata["x"] * 10


def test_prefetch_path_pairs_results_with_segments():
    segs = _segments(50)
    _FakePrefetch().run(segs)
    _assert_paired(segs)


def test_plain_gpu_path_pairs_results_with_segments():
    segs = _segments(20)
    _FakePlainGPU().run(segs)
    _assert_paired(segs)


def test_batched_path_pairs_results_despite_length_reordering():
    # durations are descending while ids ascend, so length bucketing reorders the
    # clips; assignment must still follow the segment, not the processing order.
    segs = _segments(30)
    _FakeBatched().run(segs)
    _assert_paired(segs)


def test_parallel_cpu_path_pairs_results_with_segments():
    segs = _segments(40)
    _FakeParallelCPU().run(segs)
    _assert_paired(segs)
