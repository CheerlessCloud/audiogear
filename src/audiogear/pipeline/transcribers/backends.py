"""Concrete ASR backends for consensus transcription.

All are Russian-capable and open. Add more by subclassing
:class:`~audiogear.pipeline.transcribers.base.ASRBackend`.
"""

from __future__ import annotations

from audiogear.pipeline.transcribers.base import ASRBackend


class GigaAMBackend(ASRBackend):
    """GigaAM (Salute) — Conformer Russian ASR, tops the open Russian ASR
    leaderboards.

    The pip ``gigaam`` (0.1.0) ships up to **v2** (``v2_rnnt``/``v2_ctc``), which
    emit lowercase, UNPUNCTUATED text. **GigaAM-v3** (``ai-sage/GigaAM-v3``) adds
    end-to-end variants (``e2e_rnnt``/``e2e_ctc``) that output **punctuated,
    normalized** text directly from the audio — the single-model way to get
    punctuation. v3 is newer than the pip release: install the package from
    source (``salute-developers/GigaAM``) or use an ONNX export, then set
    ``model_name`` to a v3 e2e variant.
    """

    backend_name = "gigaam"

    def __init__(self, model_name: str = "v2_rnnt", name: str | None = None, device: str = "cuda"):
        super().__init__(name=name, device=device)
        self.model_name = model_name

    def _cache_key(self):
        return (type(self).__name__, self.model_name, self.device)

    def _load(self):
        import gigaam

        from audiogear.utils.runtime import normalize_device

        return gigaam.load_model(self.model_name, device=normalize_device(self.device))

    def transcribe(self, audio_file: str) -> str:
        try:
            return self.model.transcribe(audio_file)
        except Exception:
            # long files exceed the default window; fall back to chunked decoding
            out = self.model.transcribe_longform(audio_file)
            if isinstance(out, list):
                return " ".join(seg.get("transcription", "") for seg in out)
            return str(out)


class WhisperBackend(ASRBackend):
    """OpenAI Whisper via faster-whisper. Default ``large-v3``."""

    backend_name = "whisper"

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "ru",
        compute_type: str = "int8_float16",
        name: str | None = None,
        device: str = "cuda",
    ):
        super().__init__(name=name, device=device)
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type

    def _cache_key(self):
        return (type(self).__name__, self.model_name, self.compute_type, self.device)

    def _load(self):
        from faster_whisper import WhisperModel

        from audiogear.utils.runtime import normalize_device

        return WhisperModel(self.model_name, device=normalize_device(self.device), compute_type=self.compute_type)

    def transcribe(self, audio_file: str) -> str:
        segments, _ = self.model.transcribe(audio_file, language=self.language)
        return "".join(s.text for s in segments).strip()


class Wav2Vec2Backend(ASRBackend):
    """A transformers CTC ASR model — architectural diversity vs the
    Conformer/Whisper backends, making the consensus more robust.

    NOTE: with torch<2.6 (required by gigaam), transformers refuses to load
    ``pytorch_model.bin`` checkpoints, so the model MUST ship a ``.safetensors``
    file. The default (``UrukHan/wav2vec2-russian``) does; popular ``.bin``-only
    models (jonatasgrosman, bond005) will not load until torch>=2.6.
    """

    backend_name = "wav2vec2"

    def __init__(
        self,
        model_id: str = "UrukHan/wav2vec2-russian",
        use_safetensors: bool = True,
        name: str | None = None,
        device: str = "cuda",
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.use_safetensors = use_safetensors

    def _cache_key(self):
        return (type(self).__name__, self.model_id, self.device)

    def _load(self):
        from transformers import pipeline

        return pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            model_kwargs={"use_safetensors": self.use_safetensors},
        )

    def transcribe(self, audio_file: str) -> str:
        return self.model(audio_file)["text"].strip()


class ToneBackend(ASRBackend):
    """T-one (t-tech) — streaming Conformer-CTC Russian ASR.

    Install the real package from git (the PyPI ``tone`` is an unrelated
    name-squat)::

        uv pip install "tone @ git+https://github.com/voicekit-team/T-one.git"

    Weights are pulled from ``t-tech/T-one`` on first use.
    """

    backend_name = "tone"

    def _load(self):
        from tone import StreamingCTCPipeline

        return StreamingCTCPipeline.from_hugging_face()

    def transcribe(self, audio_file: str) -> str:
        from tone import read_audio

        audio = read_audio(audio_file)
        phrases = self.model.forward_offline(audio)
        # forward_offline returns a list of TextPhrase objects with a `.text`.
        return " ".join(getattr(p, "text", str(p)) for p in phrases).strip()
