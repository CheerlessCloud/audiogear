import sys
from types import SimpleNamespace

from audiogear.pipeline.transcribers.selection import TranscriptionCandidate, select_candidate


def test_pair_distance_averages_both_cer_directions(monkeypatch):
    distances = {
        ("alpha", "beta"): 0.2,
        ("beta", "alpha"): 0.6,
    }
    monkeypatch.setitem(sys.modules, "jiwer", SimpleNamespace(cer=lambda first, second: distances[(first, second)]))

    result = select_candidate(
        [
            TranscriptionCandidate("first", "ok", "alpha"),
            TranscriptionCandidate("second", "ok", "beta"),
        ],
        ["first", "second"],
        prefer_punctuated=False,
    )

    assert result.pairwise_distances == {("first", "second"): 0.4}
    assert result.mean_distances == {"first": 0.4, "second": 0.4}
    assert result.medoid_candidate_id == "first"
    assert result.agreement == 0.6


def test_order_breaks_medoid_and_punctuation_ties(monkeypatch):
    monkeypatch.setitem(sys.modules, "jiwer", SimpleNamespace(cer=lambda first, second: 1.0))
    candidates = {
        "plain": {"status": "ok", "text": "plain"},
        "punctuated_first": {"status": "ok", "text": "First!"},
        "punctuated_second": {"status": "ok", "text": "Second?"},
    }

    result = select_candidate(
        candidates,
        ["plain", "punctuated_first", "punctuated_second"],
        prefer_punctuated=True,
    )

    assert result.medoid_candidate_id == "plain"
    assert result.punctuation_candidate_ids == ("punctuated_first", "punctuated_second")
    assert result.selected_candidate_id == "punctuated_first"
    assert result.selected_text == "First!"
    assert result.prefer_punctuated_changed_selection is True


def test_single_candidate_has_no_pure_agreement():
    result = select_candidate(
        [
            TranscriptionCandidate("error", "error", "ignored"),
            TranscriptionCandidate("blank", "ok", "  "),
            TranscriptionCandidate("only", "ok", "text"),
        ],
        ["error", "blank", "only"],
    )

    assert result.eligible_candidate_ids == ("only",)
    assert result.status == "single_candidate"
    assert result.agreement is None
    assert result.medoid_candidate_id == "only"
    assert result.selected_candidate_id == "only"


def test_no_candidate_has_complete_empty_result():
    result = select_candidate(
        {
            "failed": {"status": "error", "text": "not eligible"},
            "silent": {"status": "no_speech", "text": ""},
        },
        ["failed", "silent"],
    )

    assert result.eligible_candidate_ids == ()
    assert result.pairwise_distances == {}
    assert result.mean_distances == {}
    assert result.medoid_candidate_id is None
    assert result.selected_candidate_id is None
    assert result.selected_text == ""
    assert result.agreement is None
    assert result.status == "no_candidate"
    assert result.punctuation_candidate_ids == ()
    assert result.prefer_punctuated_changed_selection is False
