"""BUG-2: the default reader adapter must cast numeric CSV fields.

CSV values arrive as strings; ``sample_rate``/``channels``/``duration`` must be
cast so downstream (``librosa.load(sr=...)``, duration filters) get real numbers
instead of crashing on ``'<=' not supported between 'str' and 'int'``.
"""

import pytest

from audiogear.pipeline.readers.csv import CsvReader


@pytest.fixture
def adapter(tmp_path):
    return CsvReader(str(tmp_path))._default_adapter


def test_numeric_fields_are_cast(adapter):
    out = adapter(
        {"id": "x", "audio_file": "a.wav", "sample_rate": "44100", "channels": "2", "duration": "1.5"},
        "f",
        0,
    )
    assert out["sample_rate"] == 44100 and isinstance(out["sample_rate"], int)
    assert out["channels"] == 2 and isinstance(out["channels"], int)
    assert out["duration"] == 1.5 and isinstance(out["duration"], float)


def test_blank_and_invalid_numbers_become_none(adapter):
    out = adapter(
        {"id": "x", "audio_file": "a.wav", "sample_rate": "", "channels": "abc", "duration": None},
        "f",
        0,
    )
    assert out["sample_rate"] is None
    assert out["channels"] is None
    assert out["duration"] is None


def test_unknown_columns_go_into_metadata(adapter):
    out = adapter({"id": "x", "audio_file": "a.wav", "distillmos": "4.2", "gender": "male"}, "f", 0)
    assert out["metadata"]["distillmos"] == "4.2"
    assert out["metadata"]["gender"] == "male"


def test_id_falls_back_to_path_and_index(adapter):
    out = adapter({"audio_file": "a.wav"}, "some/file.csv", 7)
    assert out["id"] == "some/file.csv/7"
