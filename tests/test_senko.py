import builtins
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
from conftest import make_segment

from audiogear.pipeline.metrics.senko import SenkoDiarizationMetric
from audiogear.utils import runtime


def _speech_result():
    return {
        "merged_segments": [
            {"speaker": "SPEAKER_01", "start": 0.0001, "end": 1.23456},
            {"speaker": "SPEAKER_02", "start": 1.3, "end": 2.0},
        ],
        "raw_segments": [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 0.7},
            {"speaker": "SPEAKER_02", "start": 0.7, "end": 2.0},
        ],
        "merged_speakers_detected": 2,
        "timing_stats": {"vad_time": 0.12, "total_time": 0.34},
        "speaker_centroids": {"SPEAKER_01": np.zeros(192)},
        "speaker_color_sets": {"0": {"SPEAKER_01": "#ffffff"}},
        "vad": [(0.0, 2.0)],
    }


def _write_audio(path, samples, sample_rate=16000, subtype=None):
    soundfile.write(path, np.asarray(samples), sample_rate, subtype=subtype)


@pytest.fixture(autouse=True)
def clear_model_cache():
    runtime._MODEL_CACHE.clear()
    yield
    runtime._MODEL_CACHE.clear()


def test_optional_package_is_lazy_cached_and_official_arguments_are_forwarded(monkeypatch):
    constructor_calls = []
    diarize_calls = []

    class FakeDiarizer:
        def __init__(self, **options):
            constructor_calls.append(options)

        def diarize_samples(self, samples, **options):
            diarize_calls.append((samples, options))
            return None

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=FakeDiarizer))
    first = SenkoDiarizationMetric(device="cuda:7", mer_cos=0.8)
    second = SenkoDiarizationMetric(device="cuda")

    assert constructor_calls == []
    first_result = first._infer((np.zeros(160, dtype=np.float32), "one.wav"))
    first._infer((np.zeros(80, dtype=np.float32), "two.wav"))
    second._infer((np.zeros(40, dtype=np.float32), "three.wav"))

    assert first_result == ("[]", "[]", 0, "{}", "no_speech")
    assert constructor_calls == [
        {
            "device": "cuda",
            "vad": "silero",
            "clustering": "cpu",
            "warmup": False,
            "quiet": True,
            "mer_cos": 0.8,
        },
        {
            "device": "cuda",
            "vad": "silero",
            "clustering": "cpu",
            "warmup": False,
            "quiet": True,
            "mer_cos": None,
        },
    ]
    assert diarize_calls[0][1] == {
        "sample_rate": 16000,
        "accurate": False,
        "generate_colors": False,
        "source_name": "one.wav",
    }
    assert diarize_calls[0][0].dtype == np.float32


def test_cache_identity_covers_constructor_and_inference_configuration(monkeypatch):
    constructor_calls = []

    class FakeDiarizer:
        def __init__(self, **options):
            constructor_calls.append(options)

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=FakeDiarizer))
    metrics = [
        SenkoDiarizationMetric(),
        SenkoDiarizationMetric(vad="auto"),
        SenkoDiarizationMetric(clustering="auto"),
        SenkoDiarizationMetric(warmup=True),
        SenkoDiarizationMetric(accurate=True),
        SenkoDiarizationMetric(mer_cos=0.85),
    ]

    for metric in metrics:
        metric._runtime_on("cuda")

    assert len(constructor_calls) == len(metrics)


def test_decode_downmixes_and_resamples_stereo_to_contiguous_float32(tmp_path):
    source_sample_rate = 48000
    seconds = 0.1
    time_axis = np.arange(int(source_sample_rate * seconds)) / source_sample_rate
    stereo = np.column_stack(
        [
            0.8 * np.sin(2 * np.pi * 440 * time_axis),
            0.4 * np.sin(2 * np.pi * 220 * time_axis),
        ]
    ).astype(np.float32)
    audio_file = tmp_path / "stereo.wav"
    _write_audio(audio_file, stereo, source_sample_rate, subtype="FLOAT")

    samples, source_name = SenkoDiarizationMetric()._prepare(
        make_segment("stereo", audio_file=str(audio_file), path=str(audio_file))
    )

    assert source_name == str(audio_file)
    assert samples.shape == (1600,)
    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    assert np.all(np.isfinite(samples))
    assert float(np.max(np.abs(samples))) <= 1.0


def test_resampler_overshoot_is_clipped_but_valid_source_is_accepted(tmp_path):
    source = np.tile(np.array([1.0, -1.0], dtype=np.float32), 240)
    audio_file = tmp_path / "overshoot.wav"
    _write_audio(audio_file, source, 48000, subtype="FLOAT")

    samples, _ = SenkoDiarizationMetric()._prepare(
        make_segment("overshoot", audio_file=str(audio_file), path=str(audio_file))
    )

    assert float(samples.max()) <= 1.0
    assert float(samples.min()) >= -1.0


@pytest.mark.parametrize("bad_sample", [float("nan"), float("inf"), 1.01, -1.01])
def test_invalid_source_samples_are_rejected_before_inference(tmp_path, bad_sample):
    audio_file = tmp_path / "invalid.wav"
    _write_audio(audio_file, np.array([0.0, bad_sample, 0.0], dtype=np.float32), subtype="FLOAT")
    metric = SenkoDiarizationMetric()

    with pytest.raises(ValueError, match="finite|normalized"):
        metric._prepare(make_segment("invalid", audio_file=str(audio_file), path=str(audio_file)))


def test_decode_never_imports_or_calls_torchaudio_after_senko_load(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    _write_audio(audio_file, np.zeros(160, dtype=np.float32))
    metric = SenkoDiarizationMetric()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            raise AssertionError("Senko decode must not import torchaudio")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=lambda path: pytest.fail("torchaudio.load called")))

    samples, _ = metric._prepare(make_segment("safe", audio_file=str(audio_file), path=str(audio_file)))

    assert samples.shape == (160,)


def test_raw_merged_and_timing_are_mapped_without_extra_upstream_data():
    mapped = SenkoDiarizationMetric()._map_result(_speech_result())

    assert json.loads(mapped[0]) == [
        {"speaker": "SPEAKER_01", "start": 0.0, "end": 1.235},
        {"speaker": "SPEAKER_02", "start": 1.3, "end": 2.0},
    ]
    assert json.loads(mapped[1]) == [
        {"speaker": "SPEAKER_01", "start": 0.0, "end": 0.7},
        {"speaker": "SPEAKER_02", "start": 0.7, "end": 2.0},
    ]
    assert mapped[2] == 2
    timing = json.loads(mapped[3])
    assert timing == {"vad_time": 0.12, "total_time": 0.34}
    assert mapped[4] == "ok"
    assert "speaker_centroids" not in timing
    assert "speaker_color_sets" not in timing
    assert "vad" not in timing


def test_no_speech_has_distinct_structured_status():
    assert SenkoDiarizationMetric()._map_result(None) == ("[]", "[]", 0, "{}", "no_speech")


@pytest.mark.parametrize(
    "change,match",
    [
        (("merged_segments", None), "must be a list"),
        (("raw_segments", [None]), "must be a dictionary"),
        (("raw_segments", [{"speaker": 1, "start": 0.0, "end": 1.0}]), "invalid speaker"),
        (("raw_segments", [{"speaker": "S", "start": float("nan"), "end": 1.0}]), "invalid span"),
        (("raw_segments", [{"speaker": "S", "start": -0.1, "end": 1.0}]), "invalid span"),
        (("raw_segments", [{"speaker": "S", "start": 1.0, "end": 1.0}]), "invalid span"),
        (("merged_speakers_detected", -1), "nonnegative integer"),
        (("timing_stats", {"total_time": float("inf")}), "finite JSON"),
        (("timing_stats", {"value": object()}), "finite JSON"),
    ],
)
def test_malformed_results_are_rejected(change, match):
    result = _speech_result()
    key, value = change
    result[key] = value

    with pytest.raises(ValueError, match=match):
        SenkoDiarizationMetric()._map_result(result)


def test_inference_is_serialized_across_metric_instances(monkeypatch):
    active_calls = 0
    max_active_calls = 0
    state_lock = threading.Lock()

    class FakeDiarizer:
        def __init__(self, **options):
            pass

        def diarize_samples(self, samples, **options):
            nonlocal active_calls, max_active_calls
            with state_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            time.sleep(0.02)
            with state_lock:
                active_calls -= 1
            return None

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=FakeDiarizer))
    metrics = [SenkoDiarizationMetric(), SenkoDiarizationMetric()]
    prepared = (np.zeros(10, dtype=np.float32), "audio.wav")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda metric: metric._infer(prepared), metrics))

    assert max_active_calls == 1


def test_cpu_fallback_has_an_independent_cached_runtime(monkeypatch):
    devices = []

    class FakeDiarizer:
        def __init__(self, **options):
            devices.append(options["device"])

        def diarize_samples(self, samples, **options):
            return None

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=FakeDiarizer))
    metric = SenkoDiarizationMetric()
    prepared = (np.zeros(10, dtype=np.float32), "audio.wav")

    metric._infer_on(prepared, "cuda")
    metric._infer_on(prepared, "cpu")
    metric._infer_on(prepared, "cpu")

    assert devices == ["cuda", "cpu"]


def test_error_sentinel_is_not_resumed_after_restart(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    _write_audio(audio_file, np.zeros(160, dtype=np.float32))
    checkpoint_folder = str(tmp_path / "checkpoints")

    class BrokenDiarizer:
        def __init__(self, **options):
            pass

        def diarize_samples(self, samples, **options):
            raise RuntimeError("broken")

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=BrokenDiarizer))
    failed_segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file))
    SenkoDiarizationMetric(checkpoint_folder=checkpoint_folder).run([failed_segment])
    assert failed_segment.metadata["senko_status"] == "error"

    calls = []

    class WorkingDiarizer:
        def __init__(self, **options):
            pass

        def diarize_samples(self, samples, **options):
            calls.append(options)
            return None

    runtime._MODEL_CACHE.clear()
    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=WorkingDiarizer))
    resumed_segment = make_segment("clip", audio_file=str(audio_file), path=str(audio_file))
    SenkoDiarizationMetric(checkpoint_folder=checkpoint_folder).run([resumed_segment])

    assert len(calls) == 1
    assert resumed_segment.metadata["senko_status"] == "no_speech"


def test_checkpoint_uses_full_audio_content_and_configuration(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    checkpoint_folder = str(tmp_path / "checkpoints")
    calls = []

    class FakeDiarizer:
        def __init__(self, **options):
            pass

        def diarize_samples(self, samples, **options):
            calls.append(samples.copy())
            return None

    monkeypatch.setitem(sys.modules, "senko", SimpleNamespace(Diarizer=FakeDiarizer))

    def segment():
        return make_segment("clip", audio_file=str(audio_file), path=str(audio_file))

    _write_audio(audio_file, np.zeros(160, dtype=np.float32))
    SenkoDiarizationMetric(checkpoint_folder=checkpoint_folder).run([segment()])
    SenkoDiarizationMetric(checkpoint_folder=checkpoint_folder).run([segment()])
    assert len(calls) == 1

    _write_audio(audio_file, np.full(160, 0.25, dtype=np.float32))
    SenkoDiarizationMetric(checkpoint_folder=checkpoint_folder).run([segment()])
    assert len(calls) == 2

    SenkoDiarizationMetric(accurate=True, checkpoint_folder=checkpoint_folder).run([segment()])
    assert len(calls) == 3


def test_checkpoint_identity_covers_all_behavior_and_output_options():
    baseline = SenkoDiarizationMetric()
    variants = [
        SenkoDiarizationMetric(device="cpu"),
        SenkoDiarizationMetric(vad="auto"),
        SenkoDiarizationMetric(clustering="auto"),
        SenkoDiarizationMetric(warmup=True),
        SenkoDiarizationMetric(accurate=True),
        SenkoDiarizationMetric(mer_cos=0.8),
        SenkoDiarizationMetric(segments_column="segments"),
        SenkoDiarizationMetric(raw_segments_column="raw"),
        SenkoDiarizationMetric(num_speakers_column="speakers"),
        SenkoDiarizationMetric(timing_column="timing"),
        SenkoDiarizationMetric(status_column="status"),
    ]

    assert all(metric.checkpoint_identity != baseline.checkpoint_identity for metric in variants)


def test_invalid_configuration_is_rejected_eagerly():
    with pytest.raises(ValueError, match="sample_rate=16000"):
        SenkoDiarizationMetric(sample_rate=8000)
    with pytest.raises(ValueError, match="accurate must be a boolean"):
        SenkoDiarizationMetric(accurate=None)
