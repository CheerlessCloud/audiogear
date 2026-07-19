"""transfer_punctuation: reference words + ASR-heard punctuation.

Pure-text tests (no model): the contract is that words always come from the
reference and only trailing ``.,!?`` move over from the hypothesis, with the
alignment surviving ASR word errors and refusing to touch unreliable rows.
"""

import pytest

from audiogear.pipeline.metrics.punctuation import split_trailing_punct, transfer_punctuation


def test_split_strips_only_trailing_marks():
    words, puncts = split_trailing_punct("Привет, мир! Как дела?")
    assert words == ["Привет", "мир", "Как", "дела"]
    assert puncts == [",", "!", "", "?"]


def test_transfer_takes_question_mark_from_hypothesis():
    out, frac = transfer_punctuation(
        "ты придешь сегодня вечером домой",
        "Ты придёшь сегодня вечером домой?",
    )
    assert out == "ты придешь сегодня вечером домой?"
    assert frac == 1.0


def test_transfer_survives_asr_word_errors():
    # one substituted word ("вечером"->"ветром") must not derail the rest
    out, frac = transfer_punctuation(
        "ты придешь сегодня вечером домой",
        "Ты придёшь сегодня, ветром домой?",
    )
    assert out == "ты придешь сегодня, вечером домой?"
    assert 0.6 <= frac < 1.0


def test_transfer_refuses_low_match():
    out, frac = transfer_punctuation("совсем другие слова здесь", "Ты придёшь домой?")
    assert out is None
    assert frac < 0.6


def test_transfer_terminal_fallback_dot():
    # hypothesis without a terminal mark -> reference still ends with '.'
    out, _ = transfer_punctuation("мама мыла раму", "мама мыла раму")
    assert out == "мама мыла раму."


def test_transfer_empty_inputs():
    assert transfer_punctuation("", "Привет.") == (None, 0.0)
    assert transfer_punctuation("привет", "") == (None, 0.0)


def test_gigaam_metric_declares_three_columns():
    pytest.importorskip("gigaam")
    pytest.importorskip("jiwer")
    from audiogear.data import AudioSegment
    from audiogear.pipeline.metrics.gigaam_v3 import GigaAMv3

    m = GigaAMv3(device="cpu")
    assert m.output_columns == ("gigaam3_text", "gigaam3_cer", "text_punctuated")
    assert m._failed_value() == ("", -1.0, "")

    seg = AudioSegment(id="1", audio_file="x.wav", format="wav", text="ты придешь домой")
    hyp, cer, punct = m._score(seg, "Ты придёшь домой?")
    assert hyp == "Ты придёшь домой?"
    assert cer < 0.15
    assert punct == "ты придешь домой?"

    # punct_column=null keeps the legacy 2-column contract
    m2 = GigaAMv3(device="cpu", punct_column=None)
    assert m2.output_columns == ("gigaam3_text", "gigaam3_cer")
    assert m2._failed_value() == ("", -1.0)
    assert m2._score(seg, "Ты придёшь домой?") == ("Ты придёшь домой?", cer)
