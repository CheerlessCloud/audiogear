import sys
from types import SimpleNamespace

import pytest

from audiogear.pipeline.metrics import hf
from audiogear.pipeline.metrics.gender import GenderMetric
from audiogear.pipeline.metrics.hf import HFAudioModelMetric
from audiogear.utils import runtime


@pytest.fixture(autouse=True)
def clear_model_cache():
    runtime._MODEL_CACHE.clear()
    yield
    runtime._MODEL_CACHE.clear()


def test_hf_metric_forwards_one_exact_snapshot_to_extractor_and_model(monkeypatch):
    calls = []

    class FakeModel:
        config = SimpleNamespace(id2label={0: "label"})

        @classmethod
        def from_pretrained(cls, model_path, **options):
            calls.append(("model", model_path, options))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeExtractor:
        @classmethod
        def from_pretrained(cls, model_path, **options):
            calls.append(("extractor", model_path, options))
            return cls()

    revision = "a" * 40
    snapshot_calls = []
    monkeypatch.setattr(
        hf,
        "resolve_hf_snapshot",
        lambda repository, exact_revision, offline, patterns: (
            snapshot_calls.append((repository, exact_revision, offline, patterns)) or "/exact/snapshot"
        ),
    )
    metric = HFAudioModelMetric(
        model_id="owner/audio-model",
        metric="label",
        revision=revision,
        local_files_only=True,
        allow_patterns=["config.json", "model.safetensors"],
        device="cpu",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForAudioClassification=FakeModel,
            AutoFeatureExtractor=FakeExtractor,
        ),
    )

    _ = metric.extractor
    _ = metric._model_on("cpu")

    assert snapshot_calls == [
        ("owner/audio-model", revision, True, ("config.json", "model.safetensors")),
        ("owner/audio-model", revision, True, ("config.json", "model.safetensors")),
    ]
    assert calls == [
        (
            "extractor",
            "/exact/snapshot",
            {"trust_remote_code": False, "local_files_only": True},
        ),
        (
            "model",
            "/exact/snapshot",
            {"trust_remote_code": False, "local_files_only": True},
        ),
    ]


def test_hf_metric_revision_and_offline_mode_change_cache_identity(monkeypatch):
    load_count = 0

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **options):
            nonlocal load_count
            load_count += 1
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(hf, "resolve_hf_snapshot", lambda repository, revision, offline, patterns: "/same/path")
    online = HFAudioModelMetric("owner/model", "label", "a" * 40, device="cpu")
    offline = HFAudioModelMetric(
        "owner/model",
        "label",
        "a" * 40,
        local_files_only=True,
        device="cpu",
    )
    other_revision = HFAudioModelMetric("owner/model", "label", "b" * 40, device="cpu")
    other_patterns = HFAudioModelMetric(
        "owner/model",
        "label",
        "a" * 40,
        allow_patterns=["config.json", "model.safetensors"],
        device="cpu",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForAudioClassification=FakeModel),
    )

    _ = online._model_on("cpu")
    _ = offline._model_on("cpu")
    _ = other_revision._model_on("cpu")
    _ = other_patterns._model_on("cpu")

    assert load_count == 4
    assert (
        len(
            {
                online.checkpoint_identity,
                offline.checkpoint_identity,
                other_revision.checkpoint_identity,
                other_patterns.checkpoint_identity,
            }
        )
        == 4
    )


def test_hf_cpu_mapping_never_consults_configured_cuda_model():
    import torch

    metric = HFAudioModelMetric("owner/model", "label", "a" * 40, device="cuda")
    model_devices = []

    def model_on(device):
        model_devices.append(device)
        return SimpleNamespace(config=SimpleNamespace(id2label={0: "first", 1: "second"}))

    metric._model_on = model_on

    assert metric._map(torch.tensor([[0.0, 1.0]]), "cpu") == ["second"]
    assert model_devices == ["cpu"]


def test_gender_defaults_to_safetensors_snapshot_files():
    metric = GenderMetric(device="cpu")

    assert metric.allow_patterns == ("config.json", "preprocessor_config.json", "model.safetensors")
    assert '"allow_patterns":["config.json","preprocessor_config.json","model.safetensors"]' in (
        metric.checkpoint_identity
    )


def test_hf_metric_rejects_mutable_revision():
    with pytest.raises(ValueError, match="40-character"):
        HFAudioModelMetric("owner/model", "label", "main", device="cpu")
