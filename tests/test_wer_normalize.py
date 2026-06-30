"""Tests for the language-agnostic text normalization used by WER/CER.

``normalize_text`` is a pure function and importable without the heavy ASR deps.
"""

from audiogear.pipeline.metrics.wer import normalize_text


def test_lowercases_and_collapses_whitespace():
    assert normalize_text("Hello   WORLD\n") == "hello world"


def test_strips_punctuation_including_unicode_quotes_and_dashes():
    assert normalize_text("Привет, «мир» — как дела?") == "привет мир как дела"


def test_removes_bracketed_and_angle_segments():
    assert normalize_text("hello (laughs) <noise> world") == "hello world"


def test_empty_and_none_safe():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
