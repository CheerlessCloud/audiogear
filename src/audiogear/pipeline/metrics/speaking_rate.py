import re

from loguru import logger

from audiogear.data import AudioSegment
from audiogear.pipeline.metrics.base import BaseMetric

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Texts per phonemize() call in the shard-level pre-pass. One call per chunk
# amortises phonemizer's per-call overhead (punctuation processing setup etc.)
# across many clips, which is where the serial-path time went.
_PHONEMIZE_CHUNK = 1024


class SpeakingRateMetric(BaseMetric):
    """Speaking rate as phonemes-per-second (DataSpeech style).

    Counts phonemes in the reference ``text`` via `phonemizer` (espeak-ng
    backend, language-configurable; defaults to Russian) and divides by the
    clip duration. Requires the `espeak-ng` system binary. Also emits
    characters-per-second, which is used as a fallback when no phonemizer is
    available or the text is empty.

    espeak-ng keeps global C state and is NOT thread-safe, so this metric must
    not use ``parallel_cpu`` (concurrent calls garble the phoneme output).
    Throughput comes from batching instead: ``run`` phonemizes the whole shard
    in large chunked calls up front and the per-clip loop reads cached counts.

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
        self._phoneme_counts: dict[str, int] = {}

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

    @staticmethod
    def _count_phonemes(phonemized: str) -> int:
        # espeak separates phonemes with spaces between words; count phone symbols
        return len(phonemized.replace(" ", ""))

    def _prephonemize(self, data) -> None:
        """Phonemize the shard's texts in chunked batch calls (see class doc)."""
        todo = [(s.id, (s.text or "").strip()) for s in data]
        todo = [(sid, text) for sid, text in todo if text and sid not in self._phoneme_counts]
        if not todo:
            return
        logger.info(f"Phonemizing {len(todo)} texts in chunks of {_PHONEMIZE_CHUNK}")
        for start in range(0, len(todo), _PHONEMIZE_CHUNK):
            chunk = todo[start : start + _PHONEMIZE_CHUNK]
            try:
                phonemized = self.backend.phonemize([text for _, text in chunk], strip=True)
            except Exception as e:
                # leave these ids uncached — compute_metric retries them one by
                # one, and the base per-clip guard isolates the true culprit
                logger.warning(f"Batch phonemize failed on chunk at {start} ({e}); falling back per clip")
                continue
            for (sid, _), phon in zip(chunk, phonemized):
                self._phoneme_counts[sid] = self._count_phonemes(phon)

    def run(self, data, rank: int = 0, world_size: int = 1):
        self._phoneme_counts = {}  # don't let counts accumulate across shards
        if self.use_phonemes and len(data) > 1:
            self._prephonemize(data)
        return super().run(data, rank, world_size)

    def compute_metric(self, segment: AudioSegment):
        text = (segment.text or "").strip()
        duration = self._duration(segment)
        n_words = len(_WORD_RE.findall(text))
        char_rate = len(text) / duration if duration > 0 else 0.0

        if not text or duration <= 0:
            return 0.0, 0.0, char_rate

        if self.use_phonemes:
            n_phonemes = self._phoneme_counts.get(segment.id)
            if n_phonemes is None:
                n_phonemes = self._count_phonemes(self.backend.phonemize([text], strip=True)[0])
        else:
            n_phonemes = len(text.replace(" ", ""))

        speaking_rate = n_phonemes / duration
        phonemes_per_word = n_phonemes / n_words if n_words else 0.0
        return speaking_rate, phonemes_per_word, char_rate
