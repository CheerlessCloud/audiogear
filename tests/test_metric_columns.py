"""BUG-3: prediction metrics must write to ``*_pred`` columns, never overwrite
the curated input labels (``gender`` / ``emotion``)."""

import pytest


def test_gender_metric_writes_pred_column():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from audiogear.pipeline.metrics.gender import GenderMetric

    assert GenderMetric(device="cpu").metric == "gender_pred"


def test_emotion_metric_writes_pred_columns():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from audiogear.pipeline.metrics.emotion import EmotionMetric

    assert EmotionMetric(device="cpu").metric == ("emotion_pred", "emotion_score")


def test_whisper_wer_columns():
    pytest.importorskip("faster_whisper")
    pytest.importorskip("jiwer")
    from audiogear.pipeline.metrics.wer import WhisperWer

    assert WhisperWer(device="cpu").metric == ("whisper_wer", "whisper_cer")
