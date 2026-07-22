import json

import pytest
from conftest import make_segment

from audiogear.pipeline.transcribers.candidate import CandidateTranscriber


class _Backend:
    name = "candidate"

    def __init__(self, responses, identity=None, load_error=None):
        self.responses = iter(responses)
        self.identity = identity or {"type": "fake", "model": "exact", "revision": "full"}
        self.load_error = load_error
        self.load_calls = 0
        self.transcribe_calls = 0

    @property
    def checkpoint_identity(self):
        return self.identity

    @property
    def model(self):
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return object()

    def transcribe(self, audio_file):
        self.transcribe_calls += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def _segment(tmp_path, segment_id, content):
    audio_file = tmp_path / f"{segment_id}.wav"
    audio_file.write_bytes(content)
    return make_segment(segment_id, audio_file=str(audio_file), path=str(audio_file), text="")


def test_candidate_status_invariants_and_custom_columns(tmp_path):
    backend = _Backend(["  hello  ", "  ", RuntimeError("private decoder detail")])
    segments = [_segment(tmp_path, str(index), f"audio-{index}".encode()) for index in range(3)]
    transcriber = CandidateTranscriber(
        backend,
        candidate_id_column="candidate",
        status_column="status",
        text_column="hypothesis",
        error_code_column="error",
    )

    result = transcriber.run(segments)

    assert result == segments
    assert [segment.metadata for segment in segments] == [
        {"candidate": "candidate", "status": "ok", "hypothesis": "hello", "error": ""},
        {"candidate": "candidate", "status": "no_speech", "hypothesis": "", "error": ""},
        {"candidate": "candidate", "status": "error", "hypothesis": "", "error": "inference_error"},
    ]


def test_candidate_uses_fixed_non_sensitive_error_taxonomy(tmp_path):
    backend = _Backend(
        [
            RuntimeError("CUDA out of memory while allocating private tensor"),
            ModuleNotFoundError("private_dependency"),
            ValueError("private decoder state"),
            object(),
        ]
    )
    segments = [_segment(tmp_path, str(index), f"audio-{index}".encode()) for index in range(4)]

    CandidateTranscriber(backend).run(segments)

    assert [segment.metadata["asr_error_code"] for segment in segments] == [
        "out_of_memory",
        "dependency_error",
        "inference_error",
        "invalid_result",
    ]
    assert all(segment.metadata["asr_text"] == "" for segment in segments)


def test_candidate_preloads_before_iterating_rows(tmp_path):
    backend = _Backend([], load_error=ModuleNotFoundError("missing package"))
    segment = _segment(tmp_path, "clip", b"audio")

    with pytest.raises(ModuleNotFoundError, match="missing package"):
        CandidateTranscriber(backend).run([segment])

    assert backend.load_calls == 1
    assert backend.transcribe_calls == 0
    assert segment.metadata == {}


def test_candidate_resumes_successes_but_retries_errors(tmp_path):
    checkpoint_folder = tmp_path / "checkpoints"
    ok_segment = _segment(tmp_path, "ok", b"successful audio")
    error_segment = _segment(tmp_path, "error", b"failing audio")
    first_backend = _Backend(["saved", RuntimeError("secret failure")])
    CandidateTranscriber(first_backend, checkpoint_folder=str(checkpoint_folder)).run([ok_segment, error_segment])

    resumed_ok = _segment(tmp_path, "ok", b"successful audio")
    resumed_error = _segment(tmp_path, "error", b"failing audio")
    second_backend = _Backend(["recovered"])
    CandidateTranscriber(second_backend, checkpoint_folder=str(checkpoint_folder)).run([resumed_ok, resumed_error])

    assert second_backend.transcribe_calls == 1
    assert resumed_ok.metadata["asr_text"] == "saved"
    assert resumed_error.metadata["asr_status"] == "ok"
    assert resumed_error.metadata["asr_text"] == "recovered"
    checkpoint_text = "\n".join(path.read_text() for path in checkpoint_folder.rglob("*.jsonl"))
    assert "secret failure" not in checkpoint_text
    rows = [json.loads(line) for line in checkpoint_text.splitlines()]
    assert all(len(row["_audiogear_input_fingerprint"]) == 64 for row in rows)


def test_candidate_checkpoint_changes_with_complete_backend_identity(tmp_path):
    audio_file = _segment(tmp_path, "clip", b"audio")
    first_backend = _Backend(["first"], identity={"model": "one", "revision": "a" * 40})
    CandidateTranscriber(first_backend, checkpoint_folder=str(tmp_path / "checkpoints")).run([audio_file])

    fresh_segment = _segment(tmp_path, "clip", b"audio")
    second_backend = _Backend(["second"], identity={"model": "one", "revision": "b" * 40})
    CandidateTranscriber(second_backend, checkpoint_folder=str(tmp_path / "checkpoints")).run([fresh_segment])

    assert second_backend.transcribe_calls == 1
    assert fresh_segment.metadata["asr_text"] == "second"
