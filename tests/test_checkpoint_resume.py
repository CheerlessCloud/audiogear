"""Intra-shard resume via per-metric JSONL checkpoints.

Completion markers are shard-granular, so a crash at 90% of a big shard used to
recompute everything. With ``checkpoint_folder`` set, a metric appends one JSONL
line per finished clip and, on restart, computes only clips absent from the
file. These tests drive run → "crash" → rerun and assert only the missing work
is redone, values (and their types) survive the round-trip, torn lines are
ignored, and ``build_pipeline`` wires the folder automatically (including into
lanes).
"""

import json
import math
import os

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric


def _segments(n):
    return [
        AudioSegment(id=str(i), audio_file=f"{i}.wav", format="wav", duration=1.0, metadata={"x": i})
        for i in range(n)
    ]


class _CountingMetric(BaseMetric):
    def __init__(self, folder, fail_ids=frozenset()):
        super().__init__(metric="val", checkpoint_folder=str(folder))
        self.calls = 0
        self.fail_ids = fail_ids

    def compute_metric(self, segment):
        self.calls += 1
        if segment.id in self.fail_ids:
            raise ValueError("corrupt clip")
        return segment.metadata["x"] * 1.5  # non-integer: types must survive


def _ckpt_path(folder):
    return os.path.join(str(folder), "_CountingMetric.val", "00000.jsonl")


def test_second_run_computes_nothing_and_restores_values(tmp_path):
    _CountingMetric(tmp_path).run(_segments(10))

    segs = _segments(10)  # fresh objects — nothing precomputed in metadata
    metric = _CountingMetric(tmp_path)
    metric.run(segs)
    assert metric.calls == 0, "everything was checkpointed; nothing to compute"
    for seg in segs:
        assert seg.metadata["val"] == seg.metadata["x"] * 1.5
        assert isinstance(seg.metadata["val"], float), "JSONL must preserve numeric types"


def test_partial_checkpoint_resumes_only_the_missing_clips(tmp_path):
    _CountingMetric(tmp_path).run(_segments(10)[:6])  # "crash" after 6 clips

    segs = _segments(10)
    metric = _CountingMetric(tmp_path)
    metric.run(segs)
    assert metric.calls == 4
    assert all(s.metadata["val"] == s.metadata["x"] * 1.5 for s in segs)


def test_sentinel_rows_are_checkpointed_and_not_retried(tmp_path):
    _CountingMetric(tmp_path, fail_ids={"3"}).run(_segments(10))

    segs = _segments(10)
    metric = _CountingMetric(tmp_path)  # would succeed on "3" if asked
    metric.run(segs)
    assert metric.calls == 0, "a known-corrupt clip must not be retried on every resume"
    assert math.isnan(segs[3].metadata["val"])


def test_torn_checkpoint_line_is_skipped_and_recomputed(tmp_path):
    _CountingMetric(tmp_path).run(_segments(10)[:5])
    with open(_ckpt_path(tmp_path), "a") as f:
        f.write('{"id": "9", "val"')  # crash mid-write

    segs = _segments(10)
    metric = _CountingMetric(tmp_path)
    metric.run(segs)
    assert metric.calls == 5, "the torn line's clip (9) plus 5..8 are recomputed"
    assert segs[9].metadata["val"] == 9 * 1.5


def test_checkpoint_lines_are_flushed_per_clip(tmp_path):
    # A hard kill must lose at most the line being written: after run() every
    # clip is on disk, one JSON object per line.
    _CountingMetric(tmp_path).run(_segments(7))
    with open(_ckpt_path(tmp_path)) as f:
        rows = [json.loads(line) for line in f]
    assert sorted(r["id"] for r in rows) == sorted(str(i) for i in range(7))


def _cfg(tmp_path, **extra):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "reader": {"_target_": "audiogear.pipeline.readers.csv.CsvReader", "data_folder": str(tmp_path)},
            "metrics": [{"_target_": "audiogear.pipeline.metrics.wada_snr.SnrMetric"}],
            "writer": {"_target_": "audiogear.pipeline.writers.csv.CsvWriter", "output_folder": str(tmp_path)},
            **extra,
        }
    )


def test_build_pipeline_attaches_checkpoints_by_default(tmp_path):
    from audiogear.build import build_pipeline

    steps = build_pipeline(_cfg(tmp_path))
    assert steps[1].checkpoint_folder == os.path.join(str(tmp_path), "checkpoints")


def test_resume_false_disables_checkpoints(tmp_path):
    from audiogear.build import build_pipeline

    steps = build_pipeline(_cfg(tmp_path, resume=False))
    assert steps[1].checkpoint_folder is None


def test_checkpoint_wiring_descends_into_lanes(tmp_path):
    from audiogear.build import build_pipeline

    cfg = _cfg(tmp_path)
    cfg.metrics = [
        {
            "_target_": "audiogear.pipeline.parallel.ParallelLanes",
            "lanes": {"cpu": [{"_target_": "audiogear.pipeline.metrics.wada_snr.SnrMetric"}]},
        }
    ]
    steps = build_pipeline(cfg)
    inner = steps[1].lanes[0][0]
    assert inner.checkpoint_folder == os.path.join(str(tmp_path), "checkpoints")
