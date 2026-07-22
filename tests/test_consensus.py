import sys
from types import SimpleNamespace

import pytest
from conftest import make_segment

from audiogear.pipeline.transcribers.consensus import ConsensusTranscriber


class _Backend:
    def __init__(self, name, responses, identity=None):
        self.name = name
        self.responses = iter(responses)
        self.identity = identity or {"type": type(self).__name__, "name": self.name}
        self.calls = 0

    @property
    def checkpoint_identity(self):
        return self.identity

    def transcribe(self, audio_file):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def fake_jiwer(monkeypatch):
    def cer(reference, hypothesis):
        if reference == hypothesis:
            return 0.0
        return 1.0

    monkeypatch.setitem(sys.modules, "jiwer", SimpleNamespace(cer=cer))


def test_declares_every_dynamic_output_column():
    transcriber = ConsensusTranscriber(
        [_Backend("qwen3", []), _Backend("whisper", [])],
        min_agreement=0.5,
    )

    assert transcriber.output_columns == (
        "asr_text_qwen3",
        "asr_text_whisper",
        "asr_chosen_backend",
        "asr_agreement",
        "asr_low_confidence",
    )


def test_rejects_duplicate_backend_names():
    with pytest.raises(ValueError, match="qwen3"):
        ConsensusTranscriber([_Backend("qwen3", []), _Backend("qwen3", [])])


def test_one_nonempty_backend_keeps_valid_single_backend_semantics():
    segment = make_segment("one", text="")
    transcriber = ConsensusTranscriber([_Backend("qwen3", ["сделать заказ."])])

    transcriber.run([segment])

    assert segment.text == "сделать заказ."
    assert segment.metadata["asr_chosen_backend"] == "qwen3"
    assert segment.metadata["asr_agreement"] == 1.0


def test_empty_and_failed_hypotheses_are_excluded_from_agreement():
    segment = make_segment("mixed", text="")
    transcriber = ConsensusTranscriber(
        [
            _Backend("empty", ["  "]),
            _Backend("failed", [RuntimeError("decode failed")]),
            _Backend("qwen3", ["готово"]),
        ],
        min_agreement=0.5,
    )

    transcriber.run([segment])

    assert segment.metadata["asr_text_empty"] == "  "
    assert segment.metadata["asr_text_failed"] == ""
    assert segment.metadata["asr_chosen_backend"] == "qwen3"
    assert segment.metadata["asr_agreement"] == 1.0
    assert segment.metadata["asr_low_confidence"] is False


def test_all_failures_write_consistent_blank_columns_and_zero_agreement():
    segment = make_segment("failed", text="")
    transcriber = ConsensusTranscriber(
        [
            _Backend("missing", [ImportError("not installed")]),
            _Backend("broken", [RuntimeError("decode failed")]),
        ],
        min_agreement=0.5,
    )

    transcriber.run([segment])

    assert segment.metadata["asr_text_missing"] == ""
    assert segment.metadata["asr_text_broken"] == ""
    assert segment.metadata["asr_chosen_backend"] == ""
    assert segment.metadata["asr_agreement"] == 0.0
    assert segment.metadata["asr_low_confidence"] is True
    assert segment.text == ""


def test_import_failure_is_disabled_after_one_attempt():
    backend = _Backend("missing", [ImportError("not installed")])
    segments = [make_segment("a", text=""), make_segment("b", text="")]

    ConsensusTranscriber([backend]).run(segments)

    assert backend.calls == 1
    assert all(segment.metadata["asr_text_missing"] == "" for segment in segments)
    assert all(segment.metadata["asr_chosen_backend"] == "" for segment in segments)


def test_only_missing_preserves_existing_reference_text():
    backend = _Backend("qwen3", ["новый текст"])
    segment = make_segment("reference", text="золотой текст")

    ConsensusTranscriber([backend], only_missing=True).run([segment])

    assert backend.calls == 0
    assert segment.text == "золотой текст"


def test_valid_empty_transcript_resets_systematic_failure_streak():
    backend = _Backend(
        "qwen3",
        [
            RuntimeError("failed"),
            "",
            RuntimeError("failed"),
            RuntimeError("failed"),
        ],
    )
    transcriber = ConsensusTranscriber([backend], max_consecutive_failures=2)

    with pytest.raises(RuntimeError, match="no successful backend call"):
        transcriber.run([make_segment(str(index), text="") for index in range(4)])

    assert backend.calls == 4


def test_all_backend_error_rows_are_not_checkpointed(tmp_path):
    audio_files = []
    for index in range(2):
        audio_file = tmp_path / f"{index}.wav"
        audio_file.write_bytes(f"audio-{index}".encode())
        audio_files.append(audio_file)
    segments = [
        make_segment(str(index), audio_file=str(audio_file), path=str(audio_file), text="")
        for index, audio_file in enumerate(audio_files)
    ]
    transcriber = ConsensusTranscriber(
        [_Backend("qwen3", [RuntimeError("failed"), RuntimeError("failed")])],
        max_consecutive_failures=2,
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )

    with pytest.raises(RuntimeError, match="no successful backend call"):
        transcriber.run(segments)

    assert list((tmp_path / "checkpoints").rglob("*.jsonl")) == []


def test_checkpoint_resumes_missing_rows_and_restores_overwritten_text(tmp_path):
    segments = []
    for index in range(3):
        audio_file = tmp_path / f"{index}.wav"
        audio_file.write_bytes(f"audio-{index}".encode())
        segments.append(
            make_segment(str(index), audio_file=str(audio_file), path=str(audio_file), text="")
        )

    first_backend = _Backend("qwen3", ["первый", "второй"])
    first = ConsensusTranscriber(
        [first_backend],
        overwrite_text=True,
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    first.run(segments[:2])

    resumed_segments = [
        make_segment(
            segment.id,
            audio_file=segment.audio_file,
            path=segment.path,
            text="",
        )
        for segment in segments
    ]
    second_backend = _Backend("qwen3", ["третий"])
    second = ConsensusTranscriber(
        [second_backend],
        overwrite_text=True,
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    second.run(resumed_segments)

    assert second_backend.calls == 1
    assert [segment.text for segment in resumed_segments] == ["первый", "второй", "третий"]


def test_transcriber_checkpoint_recomputes_changed_audio_content(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"first audio")
    first = ConsensusTranscriber(
        [_Backend("qwen3", ["первый"])],
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    first.run([make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")])

    audio_file.write_bytes(b"changed audio")
    backend = _Backend("qwen3", ["второй"])
    second = ConsensusTranscriber(
        [backend],
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")
    second.run([segment])

    assert backend.calls == 1
    assert segment.metadata["asr_text_qwen3"] == "второй"


def test_torn_transcriber_checkpoint_line_is_recomputed(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"audio")
    segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")
    first = ConsensusTranscriber(
        [_Backend("qwen3", ["готово"])],
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    first.run([segment])
    checkpoint_file = next((tmp_path / "checkpoints").rglob("*.jsonl"))
    checkpoint_file.write_text('{"id":"clip"', encoding="utf-8")

    backend = _Backend("qwen3", ["повтор"])
    resumed = ConsensusTranscriber(
        [backend],
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    resumed.run([make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")])

    assert backend.calls == 1


def test_changed_backend_identity_cannot_reuse_transcriber_checkpoint(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"audio")
    segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")
    first_backend = _Backend("qwen3", ["первая модель"], identity={"model": "first"})
    first = ConsensusTranscriber([first_backend], checkpoint_folder=str(tmp_path / "checkpoints"))
    first.run([segment])

    second_backend = _Backend("qwen3", ["вторая модель"], identity={"model": "second"})
    second = ConsensusTranscriber([second_backend], checkpoint_folder=str(tmp_path / "checkpoints"))
    fresh = make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="")
    second.run([fresh])

    assert second_backend.calls == 1
    assert fresh.metadata["asr_text_qwen3"] == "вторая модель"


def test_punctuated_hypothesis_remains_preferred(fake_jiwer):
    segment = make_segment("punctuation", text="")
    transcriber = ConsensusTranscriber(
        [
            _Backend("plain", ["сделать заказ"]),
            _Backend("punctuated", ["Сделать заказ."]),
        ]
    )

    transcriber.run([segment])

    assert segment.metadata["asr_chosen_backend"] == "punctuated"
    assert segment.text == "Сделать заказ."
    assert segment.metadata["asr_agreement"] == 1.0
