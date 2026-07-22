import difflib
import json
import re

from audiogear.data import AudioSegment
from audiogear.pipeline.hf_snapshot import validate_hf_revision
from audiogear.pipeline.metrics.base import BaseMetric

# --- punctuation transfer (reference words + ASR punctuation) -----------------
# The words of a human-labelled transcript are the truth (they drive
# phonemization); a punctuating ASR (GigaAM-v3 e2e) is the truth for pauses and
# question intonation it HEARS in the audio. transfer_punctuation() combines
# them: difflib word alignment on normalized forms, trailing marks copied from
# the hypothesis onto the reference words. Battle-tested on the VoXtream-RU
# fine-tune corpus (~1900 h) before landing here.

TRANSFER_MARKS = ".,!?"
sileroRepository = "snakers4/silero-models"
sileroRevision = "9190f499588c31c24bcc1d957d40704dfd1cdf6f"
_word_norm_re = re.compile(r"[^а-яa-z0-9]+")
_tok_re = re.compile(rf"^(.*?)([{re.escape(TRANSFER_MARKS)}]*)$")


def _norm_word(w: str) -> str:
    return _word_norm_re.sub("", w.lower().replace("ё", "е"))


def split_trailing_punct(text: str) -> tuple[list[str], list[str]]:
    """-> (words with trailing ``.,!?`` stripped, one trailing mark per word or '')."""
    words, puncts = [], []
    for tok in str(text).split():
        m = _tok_re.match(tok)
        w, p = m.group(1), m.group(2)
        if not w:
            continue
        words.append(w)
        puncts.append(p[-1] if p else "")
    return words, puncts


def transfer_punctuation(reference: str, hypothesis: str, min_match: float = 0.6) -> tuple[str | None, float]:
    """Copy trailing punctuation from ``hypothesis`` onto ``reference`` words.

    Words are aligned with difflib on normalized forms (lower, ё->е, letters and
    digits only), so ASR word errors do not derail the copy. Returns
    ``(new_text, matched_fraction)``; ``new_text`` is ``None`` when fewer than
    ``min_match`` of the reference words aligned (unreliable hypothesis — leave
    the reference untouched). The terminal mark falls back to '.' when the
    hypothesis did not supply one.
    """
    gw, _ = split_trailing_punct(reference)
    hw, hp = split_trailing_punct(hypothesis)
    if not gw or not hw:
        return None, 0.0
    gn = [_norm_word(w) for w in gw]
    hn = [_norm_word(w) for w in hw]
    sm = difflib.SequenceMatcher(None, gn, hn, autojunk=False)
    new_p = [""] * len(gw)
    matched = 0
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            new_p[a + k] = hp[b + k]
            matched += 1
    frac = matched / len(gw)
    if frac < min_match:
        return None, frac
    # exact membership: `"" in ".!?"` is True (substring test), which would
    # silently skip the fallback for hypotheses lacking a terminal mark
    if new_p[-1] not in (".", "!", "?"):
        new_p[-1] = "."
    return " ".join(w + p for w, p in zip(gw, new_p)), frac


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
    available, but the combination IS: the ``GigaAMv3`` metric transcribes the
    audio with a punctuating ASR and transfers the heard punctuation onto the
    reference words via ``transfer_punctuation`` (its ``text_punctuated``
    column). Prefer it for Russian — it keeps the human words and grounds every
    mark in the acoustics; ``silero`` remains the text-only fallback and
    ``asr`` the raw audio-transcript route.
    """

    name = "❡ Punctuation"

    def __init__(
        self,
        method: str = "silero",
        column: str = "text_punctuated",
        language: str = "ru",
        device: str = "cuda",
        whisper_model: str = "large-v3",
        silero_repository: str = sileroRepository,
        silero_revision: str = sileroRevision,
        file_writer=None,
        file_reader=None,
    ):
        if method not in {"asr", "silero"}:
            raise ValueError(f"Unknown punctuation method: {method}")
        validate_hf_revision(silero_revision)
        checkpoint_identity = json.dumps(
            {
                "method": method,
                "language": language,
                "device": device,
                "whisper_model": whisper_model,
                "silero_repository": silero_repository,
                "silero_revision": silero_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(
            metric=column,
            file_writer=file_writer,
            file_reader=file_reader,
            checkpoint_identity=checkpoint_identity,
        )
        self.method = method
        self.language = language
        self.device = device
        self.whisper_model = whisper_model
        self.silero_repository = silero_repository
        self.silero_revision = silero_revision
        self._apply_te = None
        self._asr = None

    @property
    def apply_te(self):
        if self._apply_te is None:
            import torch

            repository = f"{self.silero_repository}:{self.silero_revision}"
            _, _, _, _, apply_te = torch.hub.load(repository, "silero_te", trust_repo=True)
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
        return self.apply_te(clean, lan=self.language)
