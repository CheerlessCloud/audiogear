import re
import string

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model

_BRACKETS_RE = re.compile(r"\(.*?\)|\<.*?\>")
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}«»—…“”„]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Language-agnostic normalization for WER/CER (lowercase, strip punctuation)."""
    text = _BRACKETS_RE.sub(" ", text.lower())
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


class WhisperWer(BaseMetric):
    """Transcription error of a clip against its reference ``text``.

    Transcribes with faster-whisper in the configured ``language`` (default
    Russian) and computes WER and CER vs ``segment.text``. High error usually
    means the audio and its annotation are misaligned — a useful filter for TTS
    data. Skips clips with no reference text (returns ``-1``).
    """

    name = "🌊 WER/CER"
    gpu = True
    _requires_dependencies = ("faster_whisper", "jiwer")

    def __init__(
        self,
        whisper_model: str = "large-v3",
        compute_type: str = "int8_float16",
        device: str = "cuda",
        language: str = "ru",
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        super().__init__(metric=("whisper_wer", "whisper_cer"), file_writer=file_writer, file_reader=file_reader)
        self.whisper_model = whisper_model
        self.compute_type = compute_type
        self.device = device
        self.language = language

    def _model_on(self, device: str):
        # faster-whisper wants a bare device string ("cuda" / "cpu"). int8_float16
        # is a CUDA compute type; on CPU fall back to plain int8. Process-global
        # cache => loaded once per worker, not once per task.
        def build():
            from faster_whisper import WhisperModel

            compute_type = self.compute_type if device == "cuda" else "int8"
            return WhisperModel(self.whisper_model, device=device, compute_type=compute_type)

        return cached_model(("WhisperWer", self.whisper_model, self.compute_type, device), build)

    def _transcribe(self, audio_file: str, device: str) -> str:
        segments, _ = self._model_on(device).transcribe(audio_file, language=self.language)
        return "".join(s.text for s in segments)

    def _failed_value(self):
        # -1 marks both "no reference text" and "clip could not be scored"
        # (corrupt audio etc. — the base per-clip guard routes those here).
        # Matches the convention already baked into computed datasets, where
        # filters treat any negative value as "skip".
        return -1.0, -1.0

    def _score(self, segment: AudioSegment, device: str):
        import jiwer

        reference = normalize_text(segment.text or "")
        if not reference:
            return -1.0, -1.0
        hypothesis = normalize_text(self._transcribe(segment.audio_file, device))
        return float(jiwer.wer(reference, hypothesis)), float(jiwer.cer(reference, hypothesis))

    def compute_metric(self, segment: AudioSegment):
        from audiogear.utils.runtime import normalize_device

        return self._score(segment, normalize_device(self.device))

    def compute_metric_cpu(self, segment: AudioSegment):
        # whisper already windows internally; the reliable degraded path is CPU.
        return self._score(segment, "cpu")
