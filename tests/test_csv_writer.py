"""Regression tests for BUG-1: CsvWriter must keep a stable column layout.

The original writer derived ``fieldnames`` per row and the base adapter dropped
falsy fields. A clip missing e.g. ``bit_rate`` (or with empty ``text``) therefore
wrote one fewer column than the header, shifting every later value — so on read a
neighbouring column's value landed under the wrong header (impossible MOS/WER
like ``distillmos=15984`` or ``whisper_wer=-30``). These tests pin the columns.
"""

import csv

from conftest import make_segment

from audiogear.pipeline.writers.csv import CsvWriter


def _write(tmp_path, segments, filename="out.csv", rank=0):
    writer = CsvWriter(str(tmp_path), output_filename=filename, sep="|")
    with writer:
        for seg in segments:
            writer.write(seg, rank=rank)
    return writer


def _read_raw(path, sep="|"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=sep))


def test_no_column_shift_when_rows_have_ragged_fields(tmp_path):
    # Row 1 is missing bit_rate and has empty text — exactly the raggedness that
    # used to shift the numeric columns.
    segments = [
        make_segment("a", metadata={"distillmos": 4.2, "whisper_wer": 0.0, "pyt_si_sdr": 18.0}),
        make_segment(
            "b",
            bit_rate=None,
            text="",
            metadata={"distillmos": 3.1, "whisper_wer": 0.5, "pyt_si_sdr": -30.6},
        ),
    ]
    _write(tmp_path, segments)
    rows = _read_raw(tmp_path / "out.csv")

    assert rows[0]["distillmos"] == "4.2"
    assert rows[0]["whisper_wer"] == "0.0"
    # The shifted-neighbour failure mode: si_sdr value must NOT appear under wer.
    assert rows[1]["distillmos"] == "3.1"
    assert rows[1]["whisper_wer"] == "0.5"
    assert rows[1]["pyt_si_sdr"] == "-30.6"
    # The missing field is an empty cell, not an absent column.
    assert rows[1]["bit_rate"] == ""
    assert rows[1]["text"] == ""


def test_every_row_has_the_same_column_count(tmp_path):
    segments = [
        make_segment("a", metadata={"m": 1}),
        make_segment("b", bit_rate=None, text="", metadata={"m": 2}),
        make_segment("c", duration=None, channels=None, metadata={"m": 3}),
    ]
    _write(tmp_path, segments)
    with open(tmp_path / "out.csv", newline="") as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    counts = {ln.count("|") for ln in lines}
    assert len(counts) == 1, f"ragged rows: {counts}"


def test_header_written_exactly_once_per_file(tmp_path):
    segments = [make_segment(c, metadata={"m": i}) for i, c in enumerate("abc")]
    _write(tmp_path, segments)
    with open(tmp_path / "out.csv", newline="") as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    header = lines[0]
    assert header.startswith("id|")
    assert sum(1 for ln in lines if ln == header) == 1


def test_header_written_for_each_sharded_file(tmp_path):
    # One writer instance, two output files via the $rank template. Both files
    # must carry a header (the old single-bool guard only headered the first).
    writer = CsvWriter(str(tmp_path), output_filename="out_$rank.csv", sep="|")
    with writer:
        writer.write(make_segment("a", metadata={"m": 1}), rank=0)
        writer.write(make_segment("b", metadata={"m": 2}), rank=1)

    for rank in ("00000", "00001"):
        with open(tmp_path / f"out_{rank}.csv", newline="") as f:
            first = f.readline()
        assert first.startswith("id|"), f"missing header in shard {rank}"


def test_metadata_is_flattened_into_columns(tmp_path):
    _write(tmp_path, [make_segment("a", metadata={"distillmos": 4.5, "gender_pred": "male"})])
    rows = _read_raw(tmp_path / "out.csv")
    assert rows[0]["distillmos"] == "4.5"
    assert rows[0]["gender_pred"] == "male"
    assert "metadata" not in rows[0]


def test_ensure_columns_survive_a_first_row_that_lacks_them(tmp_path):
    # The schema locks from the first row; a metric column missing there
    # (skipped clip, conditional metric) must still make the header so later
    # rows don't lose their values.
    writer = CsvWriter(str(tmp_path), output_filename="out.csv", sep="|", ensure_columns=["late_metric"])
    with writer:
        writer.write(make_segment("a", metadata={"m": 1}), rank=0)
        writer.write(make_segment("b", metadata={"m": 2, "late_metric": 0.7}), rank=0)
    rows = _read_raw(tmp_path / "out.csv")
    assert rows[0]["late_metric"] == ""
    assert rows[1]["late_metric"] == "0.7"


def test_build_pipeline_predeclares_metric_columns_to_the_writer(tmp_path):
    from omegaconf import OmegaConf

    from audiogear.build import build_pipeline

    cfg = OmegaConf.create(
        {
            "reader": {"_target_": "audiogear.pipeline.readers.csv.CsvReader", "data_folder": str(tmp_path)},
            "metrics": [{"_target_": "audiogear.pipeline.metrics.wada_snr.SnrMetric"}],
            "writer": {"_target_": "audiogear.pipeline.writers.csv.CsvWriter", "output_folder": str(tmp_path)},
        }
    )
    steps = build_pipeline(cfg)
    assert steps[-1].ensure_columns == ["wada_snr"]


def test_unexpected_late_column_is_dropped_not_shifted(tmp_path, caplog):
    # If a later row sprouts an extra key, it must be dropped (alignment kept),
    # not silently widen that row and corrupt the file.
    segments = [
        make_segment("a", metadata={"m": 1}),
        make_segment("b", metadata={"m": 2, "surprise": 99}),
    ]
    _write(tmp_path, segments)
    rows = _read_raw(tmp_path / "out.csv")
    assert "surprise" not in rows[0]
    assert rows[1]["m"] == "2"
    assert len(rows[0]) == len(rows[1])
