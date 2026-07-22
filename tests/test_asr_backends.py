import sys
from types import SimpleNamespace

import pytest

from audiogear.pipeline.hf_snapshot import clear_hf_snapshot_cache, resolve_hf_snapshot
from audiogear.pipeline.transcribers import backends
from audiogear.pipeline.transcribers.backends import GigaAMBackend, ToneBackend, WhisperBackend
from audiogear.utils import runtime


@pytest.fixture(autouse=True)
def clear_caches():
    runtime._MODEL_CACHE.clear()
    clear_hf_snapshot_cache()
    yield
    runtime._MODEL_CACHE.clear()
    clear_hf_snapshot_cache()


def test_hf_snapshot_requires_full_revision_and_forwards_offline(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **options: calls.append(options) or "/snapshot"),
    )

    with pytest.raises(ValueError, match="40-character"):
        resolve_hf_snapshot("owner/model", "main")

    revision = "a" * 40
    assert resolve_hf_snapshot("owner/model", revision, True) == "/snapshot"
    assert calls == [
        {
            "repo_id": "owner/model",
            "revision": revision,
            "local_files_only": True,
            "allow_patterns": None,
        }
    ]


def test_whisper_loads_exact_local_snapshot(monkeypatch):
    load_calls = []

    class FakeWhisperModel:
        def __init__(self, model_path, **options):
            load_calls.append((model_path, options))

    snapshot_calls = []
    monkeypatch.setattr(
        backends,
        "resolve_hf_snapshot",
        lambda repository, revision, offline, patterns: (
            snapshot_calls.append((repository, revision, offline, patterns)) or "/exact/whisper"
        ),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    backend = WhisperBackend(local_files_only=True, device="cpu")

    _ = backend.model

    assert load_calls == [("/exact/whisper", {"device": "cpu", "compute_type": "int8_float32"})]
    assert snapshot_calls == [
        (
            "Systran/faster-whisper-large-v3",
            "edaa852ec7e145841d8ffdb056a99866b5f0a478",
            True,
            ("config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"),
        )
    ]
    assert backend.checkpoint_identity["repository"] == "Systran/faster-whisper-large-v3"
    assert backend.checkpoint_identity["revision"] == "edaa852ec7e145841d8ffdb056a99866b5f0a478"
    assert backend.checkpoint_identity["local_files_only"] is True
    assert backend.checkpoint_identity["effective_compute_type"] == "int8_float32"
    assert backend.checkpoint_identity["allow_patterns"] == (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    )


def test_whisper_keeps_configured_gpu_compute_type(monkeypatch):
    load_calls = []

    class FakeWhisperModel:
        def __init__(self, model_path, **options):
            load_calls.append((model_path, options))

    monkeypatch.setattr(backends, "resolve_hf_snapshot", lambda *args: "/exact/whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    _ = WhisperBackend(device="cuda", compute_type="int8_float16").model

    assert load_calls == [("/exact/whisper", {"device": "cuda", "compute_type": "int8_float16"})]


def test_tone_is_cpu_only_and_loads_local_model_directory(monkeypatch):
    with pytest.raises(ValueError, match="only"):
        ToneBackend(device="cuda")

    local_calls = []

    class FakePipeline:
        @classmethod
        def from_local(cls, snapshot_path):
            local_calls.append(snapshot_path)
            return cls()

    snapshot_calls = []
    monkeypatch.setattr(
        backends,
        "resolve_hf_snapshot",
        lambda repository, revision, offline, patterns: (
            snapshot_calls.append((repository, revision, offline, patterns)) or "/exact/tone"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tone",
        SimpleNamespace(StreamingCTCPipeline=FakePipeline, read_audio=lambda path: path),
    )
    backend = ToneBackend(local_files_only=True)

    _ = backend.model

    assert local_calls == ["/exact/tone"]
    assert snapshot_calls == [
        (
            "t-tech/T-one",
            "106f3b0b32a9e107eb613312e4ebc61ff3d53926",
            True,
            ("model.onnx", "kenlm.bin"),
        )
    ]
    assert backend.checkpoint_identity["repository"] == "t-tech/T-one"
    assert backend.checkpoint_identity["revision"] == "106f3b0b32a9e107eb613312e4ebc61ff3d53926"
    assert backend.checkpoint_identity["allow_patterns"] == ("model.onnx", "kenlm.bin")


def test_gigaam_current_objects_and_legacy_strings_return_text(monkeypatch):
    load_calls = []

    class FakeModel:
        def __init__(self):
            self.results = [SimpleNamespace(text="current result"), "legacy result"]

        def transcribe(self, audio_file):
            return self.results.pop(0)

        def transcribe_longform(self, audio_file):
            raise AssertionError("long-form fallback was not expected")

    fake_model = FakeModel()

    def load_model(model_name, **options):
        load_calls.append((model_name, options))
        return fake_model

    fake_gigaam = SimpleNamespace(
        load_model=load_model,
        _MODEL_HASHES={"v2_rnnt": "547460139acfebd842323f59ed54ab54"},
    )
    monkeypatch.setitem(sys.modules, "gigaam", fake_gigaam)
    backend = GigaAMBackend(device="cpu", fp16_encoder=False, use_flash=True)

    assert backend.transcribe("current.wav") == "current result"
    assert backend.transcribe("legacy.wav") == "legacy result"
    assert load_calls == [
        (
            "v2_rnnt",
            {
                "fp16_encoder": False,
                "use_flash": True,
                "device": "cpu",
                "download_root": None,
            },
        )
    ]
    assert backend.checkpoint_identity["model_checksum"] == "547460139acfebd842323f59ed54ab54"
    assert backend.checkpoint_identity["fp16_encoder"] is False
    assert backend.checkpoint_identity["use_flash"] is True


def test_gigaam_longform_object_joins_segment_text(monkeypatch):
    class FakeModel:
        def transcribe(self, audio_file):
            raise ValueError("long audio")

        def transcribe_longform(self, audio_file):
            return SimpleNamespace(segments=[SimpleNamespace(text="first"), SimpleNamespace(text="second")])

    monkeypatch.setitem(
        sys.modules,
        "gigaam",
        SimpleNamespace(
            load_model=lambda *args, **options: FakeModel(),
            _MODEL_HASHES={"v2_rnnt": "547460139acfebd842323f59ed54ab54"},
        ),
    )

    assert GigaAMBackend(device="cpu").transcribe("long.wav") == "first second"
