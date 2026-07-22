import os
import sys
from types import SimpleNamespace

import pytest

from audiogear.pipeline.qwen3_snapshot import resolve_qwen_model_path
from audiogear.pipeline.transcribers.qwen3 import Qwen3ASRBackend
from audiogear.utils import runtime


@pytest.fixture(autouse=True)
def clear_model_cache():
    runtime._MODEL_CACHE.clear()
    resolve_qwen_model_path.cache_clear()
    yield
    runtime._MODEL_CACHE.clear()
    resolve_qwen_model_path.cache_clear()


class _FakeQwenModel:
    load_calls = []
    transcription_calls = []
    results = [SimpleNamespace(text="  сделать заказ с доставкой.  ")]

    @classmethod
    def from_pretrained(cls, model_name_or_path, **options):
        cls.load_calls.append((model_name_or_path, options, os.environ.get("CUDA_VISIBLE_DEVICES")))
        return cls()

    def transcribe(self, **options):
        self.transcription_calls.append(options)
        return self.results


@pytest.fixture
def fake_qwen(monkeypatch):
    _FakeQwenModel.load_calls = []
    _FakeQwenModel.transcription_calls = []
    _FakeQwenModel.results = [SimpleNamespace(text="  сделать заказ с доставкой.  ")]
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )
    snapshot_calls = []

    def snapshot_download(**options):
        snapshot_calls.append(options)
        return "/cache/resolved-snapshot"

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=_FakeQwenModel))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    _FakeQwenModel.snapshot_calls = snapshot_calls
    return _FakeQwenModel


def test_module_import_does_not_import_optional_qwen_package():
    assert "qwen_asr" not in sys.modules


def test_load_is_lazy_cached_and_uses_logical_cuda_zero(fake_qwen, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    backend = Qwen3ASRBackend(device="cuda:7")

    assert fake_qwen.load_calls == []
    assert backend.transcribe("one.wav") == "сделать заказ с доставкой."
    assert backend.transcribe("two.wav") == "сделать заказ с доставкой."

    assert len(fake_qwen.load_calls) == 1
    model_name, options, visible_device = fake_qwen.load_calls[0]
    assert model_name == "Qwen/Qwen3-ASR-1.7B"
    assert options["device_map"] == "cuda:0"
    assert options["max_inference_batch_size"] == 1
    assert options["max_new_tokens"] == 256
    assert visible_device == "7"


def test_official_transcribe_arguments_are_forwarded(fake_qwen):
    backend = Qwen3ASRBackend(
        language="Russian",
        context="названия ресторана и блюд",
        max_inference_batch_size=2,
        max_new_tokens=128,
        device="cpu",
    )

    backend.transcribe("clip.wav")

    assert fake_qwen.transcription_calls == [
        {
            "audio": ["clip.wav"],
            "context": "названия ресторана и блюд",
            "language": "Russian",
            "return_time_stamps": False,
        }
    ]
    assert fake_qwen.load_calls[0][1]["device_map"] == "cpu"


def test_revision_dtype_and_model_are_part_of_load_and_cache_identity(fake_qwen):
    first = Qwen3ASRBackend(model_name_or_path="Qwen/Qwen3-ASR-0.6B", revision="rev-a", dtype="float16")
    second = Qwen3ASRBackend(model_name_or_path="Qwen/Qwen3-ASR-0.6B", revision="rev-b", dtype="float16")

    first.transcribe("a.wav")
    second.transcribe("b.wav")

    assert len(fake_qwen.load_calls) == 2
    assert [call[0] for call in fake_qwen.load_calls] == [
        "/cache/resolved-snapshot",
        "/cache/resolved-snapshot",
    ]
    assert all("revision" not in call[1] for call in fake_qwen.load_calls)
    assert fake_qwen.snapshot_calls == [
        {"repo_id": "Qwen/Qwen3-ASR-0.6B", "revision": "rev-a"},
        {"repo_id": "Qwen/Qwen3-ASR-0.6B", "revision": "rev-b"},
    ]
    assert first._cache_key() != second._cache_key()


def test_checkpoint_identity_covers_every_output_affecting_option():
    baseline = Qwen3ASRBackend()
    variants = [
        Qwen3ASRBackend(model_name_or_path="Qwen/Qwen3-ASR-0.6B"),
        Qwen3ASRBackend(revision="revision"),
        Qwen3ASRBackend(language="English"),
        Qwen3ASRBackend(context="context"),
        Qwen3ASRBackend(dtype="float16"),
        Qwen3ASRBackend(max_inference_batch_size=2),
        Qwen3ASRBackend(max_new_tokens=128),
        Qwen3ASRBackend(name="other"),
        Qwen3ASRBackend(device="cpu"),
    ]

    assert all(variant.checkpoint_identity != baseline.checkpoint_identity for variant in variants)


def test_result_cardinality_is_validated(fake_qwen):
    backend = Qwen3ASRBackend()
    fake_qwen.results = []

    with pytest.raises(ValueError, match="returned 0 results"):
        backend.transcribe("clip.wav")

    fake_qwen.results = [SimpleNamespace(text="one"), SimpleNamespace(text="two")]
    with pytest.raises(ValueError, match="returned 2 results"):
        backend.transcribe("clip.wav")


def test_invalid_runtime_options_fail_before_model_loading():
    with pytest.raises(ValueError, match="dtype"):
        Qwen3ASRBackend(dtype="int8")
    with pytest.raises(ValueError, match="max_inference_batch_size"):
        Qwen3ASRBackend(max_inference_batch_size=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        Qwen3ASRBackend(max_new_tokens=0)
