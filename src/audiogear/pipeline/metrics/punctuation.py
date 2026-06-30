from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric


class PunctuationMetric(BaseMetric):
    """Restore punctuation & casing into a SEPARATE column (default ``text_punctuated``).

    Keeps the raw consensus transcript in ``text`` untouched and adds a punctuated
    variant as its own feature. Two methods:

    - ``silero`` (default): text-based restoration of ``segment.text`` with Silero
      TE (Russian/English/…). A dedicated punctuation+capitalization model that
      runs on the transcript only. Lightweight, no extra pip deps (torch.hub).
    - ``asr``: audio-based — re-transcribe the AUDIO with a punctuating ASR
      (Whisper by default; or GigaAM-v3 e2e) whose punctuation is derived from
      the acoustics. Use this when you want punctuation grounded in the audio.

    Note on "text + audio" punctuation: a single open model that jointly ingests
    a reference transcript AND audio to place punctuation is not readily
    available. The two practical routes are text-restoration (``silero``;
    RUPunct is an alternative) and audio-native punctuating ASR (``asr`` —
    GigaAM-v3 e2e / Whisper produce punctuation from the audio). ``asr`` is the
    closest to "by audio".
    """

    name = "❡ Punctuation"

    def __init__(
        self,
        method: str = "silero",
        column: str = "text_punctuated",
        language: str = "ru",
        device: str = "cuda",
        whisper_model: str = "large-v3",
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(metric=column, file_writer=file_writer, file_reader=file_reader)
        self.method = method
        self.language = language
        self.device = device
        self.whisper_model = whisper_model
        self._apply_te = None
        self._asr = None

    @property
    def apply_te(self):
        if self._apply_te is None:
            import torch

            _, _, _, _, apply_te = torch.hub.load("snakers4/silero-models", "silero_te", trust_repo=True)
            self._apply_te = apply_te
        return self._apply_te

    @property
    def asr(self):
        if self._asr is None:
            from audiogear.pipeline.transcribers.backends import WhisperBackend

            self._asr = WhisperBackend(model_name=self.whisper_model, language=self.language, device=self.device)
        return self._asr

    def compute_metric(self, segment: AudioSegment) -> str:
        if self.method == "asr":
            # punctuation grounded in the audio (re-transcribe with a punctuating ASR)
            return self.asr.transcribe(segment.audio_file)
        # text-based restoration: feed Silero the NORMALIZED (de-punctuated,
        # lowercased) transcript so it restores cleanly instead of doubling up
        # punctuation already present in `text`.
        from audiogear.pipeline.metrics.wer import normalize_text

        clean = normalize_text(segment.text or "")
        if not clean:
            return ""
        try:
            return self.apply_te(clean, lan=self.language)
        except Exception as e:  # noqa: BLE001
            print(f"Punctuation restore failed for {segment.audio_file}: {e}")
            return segment.text or ""
