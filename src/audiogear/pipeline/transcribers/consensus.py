"""Multi-ASR consensus transcription.

Runs N ASR backends on each clip and selects the most reliable transcript by
cross-model agreement: the *medoid* hypothesis — the one with the lowest mean
character-error-rate (CER) to all the others. Intuition: a correct transcript is
close to the other correct ones; a model that hallucinated sits far from the
pack and is rejected. With 3+ diverse backends (Conformer / Whisper / CTC) this
is robust to any single model's failure mode.

Emits per-backend transcripts (``asr_text_<name>``), the optionally overwritten
``segment.text``, the winning backend (``asr_chosen_backend``), and an
``asr_agreement`` score in [0, 1] (1 = all backends produced identical text).
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations

from loguru import logger

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.checkpoint import MetricCheckpoint, fingerprint_audio_file, inputFingerprintField
from audiogear.pipeline.metrics.wer import normalize_text
from audiogear.pipeline.transcribers.base import ASRBackend
from audiogear.utils.progress import tqdm
from audiogear.utils.runtime import free_cuda, is_oom_error

# sentence punctuation marks used to detect whether a hypothesis is punctuated
_PUNCT_RE = re.compile(r"[.,!?…;:—]")


class ConsensusTranscriber(PipelineStep):
    type = "🗣️ - TRANSCRIBER"
    checkpoint_capable = True
    name = "🗳️ Consensus ASR"

    def __init__(
        self,
        backends: list[ASRBackend],
        only_missing: bool = True,
        overwrite_text: bool = True,
        min_agreement: float | None = None,
        prefer_punctuated: bool = True,
        max_consecutive_failures: int = 50,
        checkpoint_folder=None,
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
        backend_names = [backend.name for backend in backends]
        duplicate_names = sorted({name for name in backend_names if backend_names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"ASR backend names must be unique: {', '.join(duplicate_names)}")
        self.backends = backends
        self.only_missing = only_missing
        self.overwrite_text = overwrite_text
        self.min_agreement = min_agreement
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self.prefer_punctuated = prefer_punctuated
        self.max_consecutive_failures = max_consecutive_failures
        self.checkpoint_folder = checkpoint_folder
        self.file_writer = file_writer
        self.file_reader = file_reader
        # backends that failed to *load* (e.g. optional T-one not installed) are
        # disabled for the rest of the run after one warning, instead of raising
        # an ImportError on every single clip.
        self._dead: set[str] = set()
        self._checkpoint = None

    @property
    def output_columns(self) -> tuple[str, ...]:
        columns = [f"asr_text_{backend.name}" for backend in self.backends]
        columns.extend(("asr_chosen_backend", "asr_agreement"))
        if self.min_agreement is not None:
            columns.append("asr_low_confidence")
        return tuple(columns)

    @property
    def checkpoint_identity(self) -> str:
        identity = {
            "backends": [backend.checkpoint_identity for backend in self.backends],
            "only_missing": self.only_missing,
            "overwrite_text": self.overwrite_text,
            "min_agreement": self.min_agreement,
            "prefer_punctuated": self.prefer_punctuated,
            "output_columns": self.output_columns,
        }
        return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _agreement(hyps: list[str]) -> tuple[int, float, list[float]]:
        """Return (medoid index, agreement in [0,1], per-hyp mean CER) over hyps."""
        n = len(hyps)
        if n == 1:
            return 0, 1.0, [0.0]

        import jiwer

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

    def _process(self, segment: AudioSegment) -> bool:
        hypotheses: list[str] = []
        successful_backends: list[ASRBackend] = []
        has_successful_backend_call = False
        for backend in self.backends:
            output_column = f"asr_text_{backend.name}"
            segment.metadata[output_column] = ""
            if backend.name in self._dead:
                continue
            try:
                text = backend.transcribe(segment.audio_file)
                if not isinstance(text, str):
                    raise TypeError(f"expected str, got {type(text).__name__}")
            except (ImportError, ModuleNotFoundError) as error:
                self._dead.add(backend.name)
                logger.warning(
                    f"ASR backend {backend.name} unavailable "
                    f"({type(error).__name__}: {error}); disabling for this run"
                )
                continue
            except Exception as error:  # noqa: BLE001 — per-clip failure, keep the backend
                if is_oom_error(error):
                    free_cuda()
                logger.warning(f"ASR backend {backend.name} failed on {segment.audio_file}: {error}")
                continue
            has_successful_backend_call = True
            segment.metadata[output_column] = text
            if text.strip():
                hypotheses.append(text)
                successful_backends.append(backend)

        if not hypotheses:
            segment.metadata["asr_chosen_backend"] = ""
            segment.metadata["asr_agreement"] = 0.0
            if self.min_agreement is not None:
                segment.metadata["asr_low_confidence"] = 0.0 < self.min_agreement
            return has_successful_backend_call

        hyps = hypotheses
        live = successful_backends

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
        return has_successful_backend_call

    def _open_checkpoint(self, rank: int) -> dict[str, dict]:
        identity_hash = hashlib.sha256(self.checkpoint_identity.encode("utf-8")).hexdigest()
        slug = f"{type(self).__name__}.{self.output_columns[0]}.{identity_hash}"
        self._checkpoint = MetricCheckpoint(self.checkpoint_folder, slug, rank)
        return self._checkpoint.load()

    def _restore_checkpoint(self, segment: AudioSegment, row: dict, input_fingerprint: str) -> bool:
        stored_fingerprint = row.pop(inputFingerprintField, None)
        if stored_fingerprint != input_fingerprint:
            return False
        chosen_text = row.pop("_audiogear_text", None)
        segment.metadata.update(row)
        if self.overwrite_text and chosen_text is not None:
            segment.text = chosen_text
        return True

    def _append_checkpoint(self, segment: AudioSegment, input_fingerprint: str) -> None:
        values = {column: segment.metadata.get(column) for column in self.output_columns}
        if self.overwrite_text:
            values["_audiogear_text"] = segment.text
        self._checkpoint.append(segment.id, values, input_fingerprint=input_fingerprint)

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        logger.info(f"Consensus ASR ({len(self.backends)} backends) over {len(data)} segments")
        cache = self._open_checkpoint(rank) if self.checkpoint_folder else {}
        consecutive_failures = 0
        try:
            for segment in tqdm(data):
                if self.only_missing and (segment.text or "").strip():
                    continue
                input_fingerprint = fingerprint_audio_file(segment.audio_file) if self._checkpoint else None
                row = cache.get(str(segment.id))
                if row is not None and self._restore_checkpoint(segment, row, input_fingerprint):
                    continue
                has_successful_backend_call = self._process(segment)
                if has_successful_backend_call:
                    consecutive_failures = 0
                    if self._checkpoint is not None:
                        self._append_checkpoint(segment, input_fingerprint)
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        raise RuntimeError(
                            f"Consensus ASR: {consecutive_failures} clips had no successful backend "
                            "call in an unbroken row"
                        )
                if self.file_writer:
                    self.file_writer.write(segment)
        finally:
            if self._checkpoint is not None:
                self._checkpoint.close()
                self._checkpoint = None
        return data
