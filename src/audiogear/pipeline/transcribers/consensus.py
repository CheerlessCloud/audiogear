"""Multi-ASR consensus transcription.

Runs N ASR backends on each clip and selects the most reliable transcript by
cross-model agreement: the *medoid* hypothesis — the one with the lowest mean
character-error-rate (CER) to all the others. Intuition: a correct transcript is
close to the other correct ones; a model that hallucinated sits far from the
pack and is rejected. With 3+ diverse backends (Conformer / Whisper / CTC) this
is robust to any single model's failure mode.

Emits per-backend transcripts (``asr_text_<name>``), the chosen text (written to
``segment.text``), the winning backend (``asr_chosen_backend``), and an
``asr_agreement`` score in [0, 1] (1 = all backends produced identical text).
"""

from __future__ import annotations

import re
from itertools import combinations

from loguru import logger

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.metrics.wer import normalize_text
from audiogear.pipeline.transcribers.base import ASRBackend
from audiogear.utils.progress import tqdm
from audiogear.utils.runtime import free_cuda, is_oom_error

# sentence punctuation marks used to detect whether a hypothesis is punctuated
_PUNCT_RE = re.compile(r"[.,!?…;:—]")


class ConsensusTranscriber(PipelineStep):
    type = "🗣️ - TRANSCRIBER"
    name = "🗳️ Consensus ASR"

    def __init__(
        self,
        backends: list[ASRBackend],
        only_missing: bool = True,
        overwrite_text: bool = True,
        min_agreement: float | None = None,
        prefer_punctuated: bool = True,
        file_writer=None,
        file_reader=None,
    ):
        """
        Args:
            backends: ASR backends to ensemble (>=2 for a meaningful consensus).
            only_missing: if True, only transcribe clips whose ``text`` is empty
                (annotate the rest is skipped). If False, transcribe everything.
            overwrite_text: write the chosen transcript back to ``segment.text``.
            min_agreement: if set, clips whose agreement is below this threshold
                get ``asr_low_confidence=True`` flagged in metadata (useful to
                filter out clips the models disagree on).
            prefer_punctuated: the accuracy "winner" is always the medoid (lowest
                mean CER). But some strong models (e.g. GigaAM v2) emit lowercase,
                unpunctuated text. When True, the transcript SAVED to ``text`` is
                instead the punctuated hypothesis closest to the medoid (usually
                Whisper) — punctuation derived from the audio by that ASR. Falls
                back to the medoid if no hypothesis has punctuation.
        """
        super().__init__()
        self.backends = backends
        self.only_missing = only_missing
        self.overwrite_text = overwrite_text
        self.min_agreement = min_agreement
        self.prefer_punctuated = prefer_punctuated
        self.file_writer = file_writer
        self.file_reader = file_reader
        # backends that failed to *load* (e.g. optional T-one not installed) are
        # disabled for the rest of the run after one warning, instead of raising
        # an ImportError on every single clip.
        self._dead: set[str] = set()

    @staticmethod
    def _agreement(hyps: list[str]) -> tuple[int, float, list[float]]:
        """Return (medoid index, agreement in [0,1], per-hyp mean CER) over hyps."""
        import jiwer

        n = len(hyps)
        if n == 1:
            return 0, 1.0, [0.0]
        norm = [normalize_text(h) for h in hyps]
        # pairwise CER matrix (symmetric-ish; we average both directions)
        dist = [[0.0] * n for _ in range(n)]
        for i, j in combinations(range(n), 2):
            a, b = norm[i], norm[j]
            if not a and not b:
                d = 0.0
            elif not a or not b:
                d = 1.0
            else:
                d = float(jiwer.cer(a, b))
            dist[i][j] = dist[j][i] = min(d, 1.0)
        mean_dist = [sum(row) / (n - 1) for row in dist]
        medoid = min(range(n), key=lambda i: mean_dist[i])
        agreement = 1.0 - mean_dist[medoid]
        return medoid, max(0.0, agreement), mean_dist

    @staticmethod
    def _has_punctuation(text: str) -> bool:
        return bool(_PUNCT_RE.search(text))

    def _process(self, segment: AudioSegment):
        hyps: list[str] = []
        live = []  # backends that actually ran (parallel to hyps)
        for b in self.backends:
            if b.name in self._dead:
                continue
            try:
                text = b.transcribe(segment.audio_file)
            except (ImportError, ModuleNotFoundError) as e:
                # load failure -> the backend can never work this run; disable it
                self._dead.add(b.name)
                logger.warning(f"ASR backend {b.name} unavailable ({type(e).__name__}: {e}); disabling for this run")
                continue
            except Exception as e:  # noqa: BLE001 — per-clip failure, keep the backend
                # On CUDA OOM (often a very long clip) free the cache so the
                # next backend / clip is not starved by this one's leftovers.
                if is_oom_error(e):
                    free_cuda()
                logger.warning(f"ASR backend {b.name} failed on {segment.audio_file}: {e}")
                text = ""
            hyps.append(text)
            live.append(b)
            segment.metadata[f"asr_text_{b.name}"] = text

        if not hyps:  # every backend dead/failed
            segment.metadata["asr_agreement"] = 0.0
            return

        medoid, agreement, mean_dist = self._agreement(hyps)

        # The agreement winner is the medoid; but for the SAVED text prefer a
        # punctuated hypothesis (punctuation that an audio-aware ASR produced),
        # choosing the punctuated one closest to the consensus.
        chosen_idx = medoid
        if self.prefer_punctuated and not self._has_punctuation(hyps[medoid]):
            punct = [i for i, h in enumerate(hyps) if h.strip() and self._has_punctuation(h)]
            if punct:
                chosen_idx = min(punct, key=lambda i: mean_dist[i])

        chosen = hyps[chosen_idx]
        segment.metadata["asr_chosen_backend"] = live[chosen_idx].name
        segment.metadata["asr_agreement"] = round(agreement, 4)
        if self.min_agreement is not None:
            segment.metadata["asr_low_confidence"] = agreement < self.min_agreement
        if self.overwrite_text and chosen:
            segment.text = chosen

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        logger.info(f"Consensus ASR ({len(self.backends)} backends) over {len(data)} segments")
        for segment in tqdm(data):
            if self.only_missing and (segment.text or "").strip():
                continue
            self._process(segment)
            if self.file_writer:
                self.file_writer.write(segment)
        return data
