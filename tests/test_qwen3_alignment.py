import json
import sys
import wave
from types import SimpleNamespace

import pytest
from conftest import make_segment

from audiogear.pipeline.metrics.qwen3_alignment import Qwen3ForcedAlignmentMetric
from audiogear.pipeline.qwen3_snapshot import resolve_qwen_model_path
from audiogear.utils import runtime


class _Aligner:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def align(self, **options):
        self.calls.append(options)
        return self.results


def _word(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def _result(*items):
    return SimpleNamespace(items=list(items))


def _write_wav(path, frame_count=16000, sample_rate=16000, sample_value=0):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframesraw(sample_value.to_bytes(2, byteorder="little", signed=True) * frame_count)


def _wav_segment(tmp_path, segment_id, *, frame_count=16000, sample_rate=16000, **fields):
    audio_file = tmp_path / f"{segment_id}.wav"
    _write_wav(audio_file, frame_count=frame_count, sample_rate=sample_rate)
    return make_segment(segment_id, audio_file=str(audio_file), path=str(audio_file), **fields)


@pytest.fixture(autouse=True)
def clear_model_cache():
    runtime._MODEL_CACHE.clear()
    resolve_qwen_model_path.cache_clear()
    yield
    runtime._MODEL_CACHE.clear()
    resolve_qwen_model_path.cache_clear()


def test_batch_preserves_order_around_empty_reference_text(tmp_path):
    segments = [
        _wav_segment(tmp_path, "a", text="сделать заказ"),
        _wav_segment(tmp_path, "empty", text="  "),
        _wav_segment(tmp_path, "c", text="с доставкой"),
    ]
    aligner = _Aligner(
        [
            _result(_word("сделать", 0.0, 0.4), _word("заказ", 0.4, 0.9)),
            _result(_word("с", 0.1, 0.2), _word("доставкой", 0.2, 0.8)),
        ]
    )
    metric = Qwen3ForcedAlignmentMetric(batch_size=3)
    metric._model_on = lambda device: aligner

    results = metric.compute_batch(segments)

    assert aligner.calls == [
        {
            "audio": [segments[0].audio_file, segments[2].audio_file],
            "text": ["сделать заказ", "с доставкой"],
            "language": ["Russian", "Russian"],
        }
    ]
    assert results[1] == ("[]", "empty_text")
    assert json.loads(results[0][0]) == [
        {"text": "сделать", "start": 0.0, "end": 0.4},
        {"text": "заказ", "start": 0.4, "end": 0.9},
    ]
    assert results[0][1] == "ok"
    assert results[2][1] == "ok"


def test_all_empty_batch_never_loads_or_invokes_model():
    metric = Qwen3ForcedAlignmentMetric(batch_size=2)

    def fail_if_loaded(device):
        raise AssertionError("model must not load")

    metric._model_on = fail_if_loaded
    results = metric.compute_batch([make_segment("a", text=""), make_segment("b", text=None)])

    assert results == [("[]", "empty_text"), ("[]", "empty_text")]


def test_json_is_compact_utf8_and_preserves_interpolated_off_grid_timestamps():
    metric = Qwen3ForcedAlignmentMetric()
    serialized = metric._serialize_result(_result(_word('ёлка, | "дом"', 2.98, 3.48)))

    assert serialized == '[{"text":"ёлка, | \\"дом\\"","start":2.98,"end":3.48}]'
    assert json.loads(serialized)[0]["text"] == 'ёлка, | "дом"'


@pytest.mark.parametrize(
    "items,match",
    [
        ([_word("bad", float("nan"), 1.0)], "invalid span"),
        ([_word("bad", -0.1, 1.0)], "invalid span"),
        ([_word("bad", -0.0001, 1.0)], "invalid span"),
        ([_word("bad", 1.0, 0.5)], "invalid span"),
        ([_word("one", 0.0, 1.0), _word("two", 0.9, 1.2)], "not monotonic"),
        ([_word("one", 0.0, 1.0001), _word("two", 1.0, 1.2)], "not monotonic"),
    ],
)
def test_invalid_timestamps_are_rejected(items, match):
    with pytest.raises(ValueError, match=match):
        Qwen3ForcedAlignmentMetric._serialize_result(_result(*items))


@pytest.mark.parametrize(
    "result",
    [
        _result(),
        _result(_word("", 0.0, 0.5)),
        _result(_word("  ", 0.0, 0.5)),
        _result(_word("bad", float("inf"), 2.0)),
    ],
)
def test_malformed_result_becomes_structured_error_sentinel(result):
    segment = make_segment("bad", text="текст")
    metric = Qwen3ForcedAlignmentMetric()
    metric._model_on = lambda device: _Aligner([result])

    metric.run([segment])

    assert segment.metadata["qwen3_alignment"] == "[]"
    assert segment.metadata["qwen3_alignment_status"] == "error"


def test_official_result_cardinality_is_validated():
    metric = Qwen3ForcedAlignmentMetric()
    metric._model_on = lambda device: _Aligner([])

    with pytest.raises(ValueError, match="returned 0 results"):
        metric.compute_metric(make_segment("a", text="текст"))


def test_cpu_fallback_uses_separate_cpu_model(tmp_path):
    segment = _wav_segment(tmp_path, "cpu", text="текст")
    aligner = _Aligner([_result(_word("текст", 0.0, 0.5))])
    devices = []
    metric = Qwen3ForcedAlignmentMetric(device="cuda")

    def model_on(device):
        devices.append(device)
        return aligner

    metric._model_on = model_on

    alignment, status = metric.compute_metric_cpu(segment)

    assert devices == ["cpu"]
    assert status == "ok"
    assert json.loads(alignment)[0]["text"] == "текст"


def test_endpoint_exactly_at_tolerance_boundary_is_ok(tmp_path):
    segment = _wav_segment(tmp_path, "boundary", frame_count=16000, sample_rate=16000, text="текст")
    metric = Qwen3ForcedAlignmentMetric()
    metric._model_on = lambda device: _Aligner([_result(_word("текст", 0.0, 1.08))])

    alignment, status = metric.compute_metric(segment)

    assert status == "ok"
    assert json.loads(alignment)[0]["end"] == 1.08


def test_endpoint_one_frame_beyond_tolerance_is_out_of_bounds_and_preserved(tmp_path):
    endpoint = 17281 / 16000
    segment = _wav_segment(tmp_path, "beyond", frame_count=16000, sample_rate=16000, text="текст")
    metric = Qwen3ForcedAlignmentMetric()
    metric._model_on = lambda device: _Aligner([_result(_word("текст", 0.0, endpoint))])

    alignment, status = metric.compute_metric(segment)

    assert status == "out_of_bounds"
    assert json.loads(alignment) == [{"text": "текст", "start": 0.0, "end": endpoint}]


@pytest.mark.parametrize("metadata_duration", [None, 100.0])
def test_wav_header_wins_over_absent_or_stale_segment_metadata(tmp_path, metadata_duration):
    segment = _wav_segment(
        tmp_path,
        f"header-{metadata_duration}",
        frame_count=16000,
        sample_rate=16000,
        text="текст",
        duration=metadata_duration,
    )
    segment.sample_rate = 8000
    metric = Qwen3ForcedAlignmentMetric()
    metric._model_on = lambda device: _Aligner([_result(_word("текст", 0.0, 1.081))])

    _, status = metric.compute_metric(segment)

    assert status == "out_of_bounds"


def test_checkpoint_identity_covers_model_revision_language_dtype_and_columns():
    baseline = Qwen3ForcedAlignmentMetric()
    variants = [
        Qwen3ForcedAlignmentMetric(model_name_or_path="local/model"),
        Qwen3ForcedAlignmentMetric(revision="snapshot"),
        Qwen3ForcedAlignmentMetric(language="English"),
        Qwen3ForcedAlignmentMetric(dtype="float16"),
        Qwen3ForcedAlignmentMetric(alignment_column="words"),
        Qwen3ForcedAlignmentMetric(device="cpu"),
        Qwen3ForcedAlignmentMetric(batch_size=2),
        Qwen3ForcedAlignmentMetric(max_batch_seconds=100.0),
    ]

    assert all(metric.checkpoint_identity != baseline.checkpoint_identity for metric in variants)
    cpu_identity = json.loads(Qwen3ForcedAlignmentMetric(device="cpu").checkpoint_identity)
    assert cpu_identity["device"] == "cpu"
    assert cpu_identity["effective_dtype"] == "float32"
    assert cpu_identity["timestamp_segment_ms"] == 80
    assert cpu_identity["endpoint_tolerance_ms"] == 80
    assert cpu_identity["status_mapping_version"] == "2.0.0"


def test_checkpoint_identity_changes_with_timestamp_segment_tolerance_and_status_mapping_version(monkeypatch):
    baseline = Qwen3ForcedAlignmentMetric().checkpoint_identity

    monkeypatch.setattr(Qwen3ForcedAlignmentMetric, "timestampSegmentMs", 81)
    changed_timestamp_segment = Qwen3ForcedAlignmentMetric().checkpoint_identity

    monkeypatch.setattr(Qwen3ForcedAlignmentMetric, "timestampSegmentMs", 80)
    monkeypatch.setattr(Qwen3ForcedAlignmentMetric, "endpointToleranceMs", 81)
    changed_tolerance = Qwen3ForcedAlignmentMetric().checkpoint_identity

    monkeypatch.setattr(Qwen3ForcedAlignmentMetric, "endpointToleranceMs", 80)
    monkeypatch.setattr(Qwen3ForcedAlignmentMetric, "statusMappingVersion", "2.0.1")
    changed_mapping = Qwen3ForcedAlignmentMetric().checkpoint_identity

    assert changed_timestamp_segment != baseline
    assert changed_tolerance != baseline
    assert changed_mapping != baseline


def test_checkpoint_is_bound_to_exact_reference_and_full_audio_content(tmp_path):
    audio_file = tmp_path / "clip.wav"
    _write_wav(audio_file, sample_value=1)
    aligner = _Aligner([_result(_word("текст", 0.0, 0.5))])

    def segment(text):
        return make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text=text)

    first = Qwen3ForcedAlignmentMetric(checkpoint_folder=str(tmp_path / "checkpoints"))
    first._model_on = lambda device: aligner
    first.run([segment("точный текст")])

    resumed = segment("точный текст")
    second = Qwen3ForcedAlignmentMetric(checkpoint_folder=str(tmp_path / "checkpoints"))
    second._model_on = lambda device: aligner
    second.run([resumed])
    assert len(aligner.calls) == 1
    assert "_audiogear_input_fingerprint" not in resumed.metadata

    changed_text = Qwen3ForcedAlignmentMetric(checkpoint_folder=str(tmp_path / "checkpoints"))
    changed_text._model_on = lambda device: aligner
    changed_text.run([segment("исправленный текст")])
    assert len(aligner.calls) == 2

    _write_wav(audio_file, sample_value=2)
    changed_audio = Qwen3ForcedAlignmentMetric(checkpoint_folder=str(tmp_path / "checkpoints"))
    changed_audio._model_on = lambda device: aligner
    changed_audio.run([segment("исправленный текст")])
    assert len(aligner.calls) == 3


def test_out_of_bounds_result_resumes_without_inference(tmp_path):
    checkpoint_folder = str(tmp_path / "checkpoints")
    segment = _wav_segment(tmp_path, "clip", text="текст")
    aligner = _Aligner([_result(_word("текст", 0.0, 1.16))])
    first = Qwen3ForcedAlignmentMetric(checkpoint_folder=checkpoint_folder)
    first._model_on = lambda device: aligner

    first.run([segment])

    assert segment.metadata["qwen3_alignment_status"] == "out_of_bounds"

    resumed = make_segment(
        "clip",
        audio_file=segment.audio_file,
        path=segment.path,
        text="текст",
    )
    second = Qwen3ForcedAlignmentMetric(checkpoint_folder=checkpoint_folder)
    second._model_on = lambda device: aligner
    second.run([resumed])

    assert len(aligner.calls) == 1
    assert resumed.metadata["qwen3_alignment_status"] == "out_of_bounds"
    assert json.loads(resumed.metadata["qwen3_alignment"])[0]["end"] == 1.16


def test_error_sentinels_are_recomputed_after_systematic_failure_is_fixed(tmp_path):
    audio_file = tmp_path / "clip.wav"
    _write_wav(audio_file)

    def segments():
        return [
            make_segment(
                segment_id,
                audio_file=str(audio_file),
                path=str(audio_file),
                text="текст",
            )
            for segment_id in ("first", "second")
        ]

    class BrokenAligner:
        def align(self, **options):
            raise RuntimeError("broken model")

    checkpoint_folder = str(tmp_path / "checkpoints")
    broken = Qwen3ForcedAlignmentMetric(
        checkpoint_folder=checkpoint_folder,
        max_consecutive_failures=2,
    )
    broken._model_on = lambda device: BrokenAligner()

    with pytest.raises(RuntimeError, match="2 clips failed"):
        broken.run(segments())

    working_aligner = _Aligner([_result(_word("текст", 0.0, 0.5))])
    repaired = Qwen3ForcedAlignmentMetric(
        checkpoint_folder=checkpoint_folder,
        max_consecutive_failures=2,
    )
    repaired._model_on = lambda device: working_aligner
    repaired_segments = segments()
    repaired.run(repaired_segments)

    assert len(working_aligner.calls) == 2
    assert all(segment.metadata["qwen3_alignment_status"] == "ok" for segment in repaired_segments)


def test_cpu_checkpoint_is_not_reused_on_cuda(tmp_path):
    audio_file = tmp_path / "clip.wav"
    _write_wav(audio_file)
    segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="текст")
    aligner = _Aligner([_result(_word("текст", 0.0, 0.5))])

    cpu_metric = Qwen3ForcedAlignmentMetric(
        device="cpu",
        checkpoint_folder=str(tmp_path / "checkpoints"),
    )
    cpu_metric._model_on = lambda device: aligner
    cpu_metric.run([segment])

    cuda_metric = Qwen3ForcedAlignmentMetric(checkpoint_folder=str(tmp_path / "checkpoints"))
    cuda_metric._model_on = lambda device: aligner
    cuda_metric.run([make_segment("clip", audio_file=str(audio_file), path=str(audio_file), text="текст")])

    assert len(aligner.calls) == 2


def test_optional_package_is_loaded_lazily_with_cpu_float32(monkeypatch):
    load_calls = []

    class FakeForcedAligner:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **options):
            load_calls.append((model_name_or_path, options))
            return cls()

    snapshot_calls = []

    def snapshot_download(**options):
        snapshot_calls.append(options)
        return "/cache/resolved-aligner"

    fake_torch = SimpleNamespace(float32="torch.float32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=FakeForcedAligner))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    metric = Qwen3ForcedAlignmentMetric(revision="snapshot", device="cpu")

    assert load_calls == []
    metric._model_on("cpu")

    assert load_calls[0][0] == "/cache/resolved-aligner"
    assert str(load_calls[0][1]["dtype"]) == "torch.float32"
    assert load_calls[0][1]["device_map"] == "cpu"
    assert "revision" not in load_calls[0][1]
    assert snapshot_calls == [{"repo_id": "Qwen/Qwen3-ForcedAligner-0.6B", "revision": "snapshot"}]
