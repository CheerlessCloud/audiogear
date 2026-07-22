import csv
from pathlib import Path

from conftest import make_segment
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = str(Path(__file__).parents[1] / "configs")


def _compose(config_name, overrides=None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_default_annotation_preset_contains_only_compatible_asr_backends():
    config = _compose("annotate")

    assert [backend._target_.rsplit(".", 1)[-1] for backend in config.metrics[0].backends] == [
        "GigaAMBackend",
        "WhisperBackend",
    ]


def test_qwen3_annotation_preset_composes_as_pinned_candidate_job():
    config = _compose("annotate_qwen3")
    candidate = config.metrics[0]
    backend = candidate.backend

    assert candidate._target_ == "audiogear.pipeline.transcribers.candidate.CandidateTranscriber"
    assert candidate.status_column == "asr_qwen3_status"
    assert candidate.text_column == "asr_text_qwen3"
    assert backend.model_name_or_path == "Qwen/Qwen3-ASR-1.7B"
    assert backend.revision == "7278e1e70fe206f11671096ffdd38061171dd6e5"
    assert backend.dtype == "bfloat16"
    assert backend.max_inference_batch_size == 1
    assert config.executor.workers == 1
    assert config.executor.gpus == 1
    assert config.executor.skip_completed is False
    assert config.executor.logging_dir == "logs/qwen3_annotation"


def test_qwen3_model_and_revision_can_be_overridden_together():
    config = _compose(
        "annotate_qwen3",
        [
            "metrics.0.backend.model_name_or_path=Qwen/Qwen3-ASR-0.6B",
            f"metrics.0.backend.revision={'a' * 40}",
        ],
    )
    backend = config.metrics[0].backend

    assert backend._target_ == "audiogear.pipeline.transcribers.qwen3.Qwen3ASRBackend"
    assert backend.model_name_or_path == "Qwen/Qwen3-ASR-0.6B"
    assert backend.revision == "a" * 40


def test_qwen3_revision_can_be_overridden_without_hydra_append_syntax():
    config = _compose(
        "annotate_qwen3",
        [f"metrics.0.backend.revision={'a' * 40}"],
    )

    assert config.metrics[0].backend.revision == "a" * 40


def test_staged_consensus_preset_instantiates_without_model_dependencies():
    config = _compose("annotate_consensus")

    selector = instantiate(config.metrics[0])

    assert [candidate_id for candidate_id, _, _ in selector.candidates] == [
        "gigaam",
        "whisper",
        "tone",
        "qwen3",
    ]
    assert selector.overwrite_text is True


def test_alignment_preset_composes_without_asr_or_text_overwrite():
    config = _compose("align_qwen3")

    assert len(config.metrics) == 1
    assert config.metrics[0]._target_.endswith("Qwen3ForcedAlignmentMetric")
    assert config.metrics[0].model_name_or_path == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert config.executor.workers == 1
    assert config.executor.gpus == 1
    assert "overwrite_text" not in config.metrics[0]
    assert config.executor.skip_completed is False
    assert config.executor.logging_dir == "logs/qwen3_alignment"
    assert config.executor.logging_dir != _compose("annotate_qwen3").executor.logging_dir


def test_qwen_rank_completion_markers_never_skip_sequential_or_changed_jobs(tmp_path):
    annotation = _compose("annotate_qwen3")
    alignment = _compose("align_qwen3")
    changed_model = _compose(
        "annotate_qwen3",
        [
            "metrics.0.backend.model_name_or_path=Qwen/Qwen3-ASR-0.6B",
            f"metrics.0.backend.revision={'a' * 40}",
        ],
    )
    configs = [annotation, alignment, changed_model]

    for index, config in enumerate(configs):
        config.executor.logging_dir = str(tmp_path / f"job-{index}")
        executor = instantiate(config.executor, pipeline=[])
        executor.mark_rank_as_completed(0)
        assert executor.is_rank_completed(0) is False


def test_annotation_preset_processes_and_exactly_preserves_nonempty_reference(tmp_path):
    from audiogear.build import build_pipeline

    config = _compose("annotate_qwen3")
    config.reader.data_folder = str(tmp_path)
    config.writer.output_folder = str(tmp_path)
    config.writer.output_filename = "annotated.csv"
    steps = build_pipeline(config)
    transcriber, writer = steps[1], steps[-1]
    transcriber.backend._load = lambda: object()
    transcriber.backend.transcribe = lambda audio_file: "распознанный текст"
    audio_file = tmp_path / "reference.wav"
    audio_file.write_bytes(b"audio")
    reference = '  Ёлка | "дом", байт-в-байт.  '
    segment = make_segment(
        "reference",
        audio_file=str(audio_file),
        path=str(audio_file),
        text=reference,
    )

    transcriber.run([segment])
    with writer:
        writer.write(segment)

    with (tmp_path / "annotated.csv").open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file, delimiter="|"))

    assert segment.metadata["asr_text_qwen3"] == "распознанный текст"
    assert segment.text.encode("utf-8") == reference.encode("utf-8")
    assert row["text"].encode("utf-8") == reference.encode("utf-8")
    assert row["asr_text_qwen3"] == "распознанный текст"
    assert "_audiogear_input_fingerprint" not in row


def test_qwen_configs_instantiate_without_loading_optional_package():
    annotation_config = _compose("annotate_qwen3")
    alignment_config = _compose("align_qwen3")

    transcriber = instantiate(annotation_config.metrics[0])
    alignment = instantiate(alignment_config.metrics[0])

    assert transcriber.backend.model_name_or_path == "Qwen/Qwen3-ASR-1.7B"
    assert alignment.model_name_or_path == "Qwen/Qwen3-ForcedAligner-0.6B"


def test_senko_preset_composes_with_verified_rtx_3060_defaults():
    config = _compose("diarize_senko")
    metric = config.metrics[0]

    assert metric._target_ == "audiogear.pipeline.metrics.senko.SenkoDiarizationMetric"
    assert metric.device == "cuda"
    assert metric.vad == "silero"
    assert metric.clustering == "cpu"
    assert metric.warmup is False
    assert metric.accurate is False
    assert metric.sample_rate == 16000
    assert config.executor.tasks == 1
    assert config.executor.workers == 1
    assert config.executor.gpus == 1
    assert config.executor.skip_completed is False
    assert config.executor.logging_dir == "logs/senko_diarization"
    assert config.executor.logging_dir not in {
        _compose("annotate_qwen3").executor.logging_dir,
        _compose("align_qwen3").executor.logging_dir,
    }


def test_senko_config_instantiates_without_loading_optional_package():
    config = _compose("diarize_senko")

    metric = instantiate(config.metrics[0])

    assert metric.device == "cuda"
    assert metric.output_columns == (
        "senko_segments",
        "senko_raw_segments",
        "senko_num_speakers",
        "senko_timing",
        "senko_status",
    )
