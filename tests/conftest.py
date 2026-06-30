"""Shared fixtures/helpers for the audiogear test suite.

These tests are deliberately CPU-only and model-free: they exercise the pipeline
plumbing (CSV I/O, the metric scheduling/ordering in ``BaseMetric``, runtime
helpers) where the real bugs lived, without downloading any model weights.
"""

from audiogear.data import AudioSegment


def make_segment(seg_id, *, metadata=None, **fields):
    """Build an ``AudioSegment`` with sane defaults for writer/reader tests."""
    base = {
        "audio_file": f"clips/{seg_id}.wav",
        "path": f"clips/{seg_id}.wav",
        "format": "wav",
        "sample_rate": 44100,
        "channels": 1,
        "bit_rate": "128000",
        "duration": 1.0,
        "text": "hello world",
    }
    base.update(fields)
    return AudioSegment(id=seg_id, metadata=metadata or {}, **base)
