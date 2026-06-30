import re

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class SpeakingRateMetric(BaseMetric):
    """Speaking rate as phonemes-per-second (DataSpeech style).

    Counts phonemes in the reference ``text`` via `phonemizer` (espeak-ng
    backend, language-configurable; defaults to Russian) and divides by the
    clip duration. Requires the `espeak-ng` system binary. Also emits
    characters-per-second, which is used as a fallback when no phonemizer is
    available or the text is empty.

    Emits: ``speaking_rate`` (phonemes/s), ``phonemes_per_word``,
    ``char_rate`` (chars/s).
    """

    name = "⏩ SpeakingRate"
    _requires_dependencies = ("phonemizer",)

    def __init__(
        self,
        language: str = "ru",
        use_phonemes: bool = True,
        file_writer=None,
        file_reader=None,
    ):
        super().__init__(
            metric=("speaking_rate", "phonemes_per_word", "char_rate"),
            file_writer=file_writer,
            file_reader=file_reader,
        )
        self.language = language
        self.use_phonemes = use_phonemes
        self._backend = None

    @property
    def backend(self):
        if self._backend is None and self.use_phonemes:
            from phonemizer.backend import EspeakBackend

            self._backend = EspeakBackend(self.language, with_stress=False)
        return self._backend

    def _duration(self, segment: AudioSegment) -> float:
        if segment.duration:
            return float(segment.duration)
        from audiogear.audio import load_audio

        audio, sr = load_audio(segment.audio_file, mono=True)
        return audio.shape[-1] / sr

    def compute_metric(self, segment: AudioSegment):
        text = (segment.text or "").strip()
        duration = self._duration(segment)
        n_words = len(_WORD_RE.findall(text))
        char_rate = len(text) / duration if duration > 0 else 0.0

        if not text or duration <= 0:
            return 0.0, 0.0, char_rate

        if self.use_phonemes:
            phonemes = self.backend.phonemize([text], strip=True)[0]
            # espeak separates phonemes with spaces between words; count phone symbols
            n_phonemes = len(phonemes.replace(" ", ""))
        else:
            n_phonemes = len(text.replace(" ", ""))

        speaking_rate = n_phonemes / duration
        phonemes_per_word = n_phonemes / n_words if n_words else 0.0
        return speaking_rate, phonemes_per_word, char_rate
