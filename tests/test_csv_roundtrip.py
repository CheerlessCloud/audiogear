"""End-to-end CsvWriter -> CsvReader round-trip (BUG-1 + BUG-2 together).

Writing then reading the same metadata must recover the metric values intact and
with the correct numeric types (sample_rate int, duration float).
"""

from conftest import make_segment

from audiogear.pipeline.readers.csv import CsvReader
from audiogear.pipeline.writers.csv import CsvWriter


def test_roundtrip_preserves_values_and_types(tmp_path):
    segments = [
        make_segment("a", duration=1.5, metadata={"distillmos": 4.2, "whisper_wer": 0.0}),
        make_segment("b", bit_rate=None, text="", duration=2.0, metadata={"distillmos": 3.1, "whisper_wer": 0.5}),
    ]
    writer = CsvWriter(str(tmp_path), output_filename="metadata.csv", sep="|")
    with writer:
        for seg in segments:
            writer.write(seg, rank=0)

    reader = CsvReader(str(tmp_path), delimiter="|", glob_pattern="metadata.csv")
    out = reader.run(world_size=1, rank=0)

    assert [s.id for s in out] == ["a", "b"]
    a, b = out
    # BUG-2: numeric fields come back as real numbers, not strings.
    assert a.sample_rate == 44100 and isinstance(a.sample_rate, int)
    assert a.duration == 1.5 and isinstance(a.duration, float)
    # BUG-1: metric values land under the right columns.
    assert a.metadata["distillmos"] == "4.2"
    assert b.metadata["distillmos"] == "3.1"
    assert b.metadata["whisper_wer"] == "0.5"
