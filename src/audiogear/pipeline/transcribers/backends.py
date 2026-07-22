"""Concrete ASR backends for consensus transcription.

All are Russian-capable and open. Add more by subclassing
:class:`~audiogear.pipeline.transcribers.base.ASRBackend`.
"""

from __future__ import annotations

import hashlib
import os

from audiogear.pipeline.hf_snapshot import normalize_allow_patterns, resolve_hf_snapshot, validate_hf_revision
from audiogear.pipeline.transcribers.base import ASRBackend

gigaamChecksums = {
    "v1_ctc": "f027f199e590a391d015aeede2e66174",
    "v1_rnnt": "02c758999bcdc6afcb2087ef256d47ef",
    "v1_ssl": "dc7f7b231f7f91c4968dc21910e7b396",
    "v2_ctc": "e00f59cb5d39624fb30d1786044795bf",
    "v2_rnnt": "547460139acfebd842323f59ed54ab54",
    "v2_ssl": "cd4cf819c8191a07b9d7edcad111668e",
    "v3_ctc": "73413e7be9c6a5935827bfab5c0dd678",
    "v3_rnnt": "0fd2c9a1ff66abd8d32a3a07f7592815",
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "v3_e2e_rnnt": "2730de7545ac43ad256485a462b0a27a",
    "v3_ssl": "70cbf5ed7303a0ed242ddb257e9dc6a6",
    "multilingual_ctc": "5379d887c53ccd9cb95981e2a1832720",
    "multilingual_ssl": "af54fed7a0337eeae7c4a25b2f8779c8",
    "multilingual_large_ctc": "79a9adde50dd7f35bbf70927cb6557d0",
    "multilingual_large_ssl": "2ef65a2ca413f6e1f99a4df0e86c1cee",
}
gigaamShortNames = {
    "ctc": "v3_ctc",
    "rnnt": "v3_rnnt",
    "e2e_ctc": "v3_e2e_ctc",
    "e2e_rnnt": "v3_e2e_rnnt",
    "ssl": "v3_ssl",
}
whisperRepository = "Systran/faster-whisper-large-v3"
whisperRevision = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
whisperAllowPatterns = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
toneRepository = "t-tech/T-one"
toneRevision = "106f3b0b32a9e107eb613312e4ebc61ff3d53926"
toneAllowPatterns = ("model.onnx", "kenlm.bin")


class GigaAMBackend(ASRBackend):
    """GigaAM (Salute) Conformer Russian ASR.

    GigaAM 0.2 returns structured transcription objects. Older versions returned
    strings, which remain supported for existing installations. V2 models emit
    lowercase unpunctuated text; v3 end-to-end variants emit punctuated,
    normalized text.
    """

    backend_name = "gigaam"

    def __init__(
        self,
        model_name: str = "v2_rnnt",
        model_checksum: str | None = None,
        fp16_encoder: bool = True,
        use_flash: bool | None = False,
        name: str | None = None,
        device: str = "cuda",
        download_root: str | None = None,
    ):
        super().__init__(name=name, device=device)
        normalized_model_name = gigaamShortNames.get(model_name, model_name)
        self.model_name = model_name
        self.model_checksum = model_checksum or gigaamChecksums.get(normalized_model_name)
        self.fp16_encoder = fp16_encoder
        self.use_flash = use_flash
        self.download_root = download_root

    @property
    def checkpoint_identity(self) -> dict:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "model_name": self.model_name,
            "model_checksum": self.model_checksum,
            "fp16_encoder": self.fp16_encoder,
            "use_flash": self.use_flash,
            "device": self.device,
            "download_root": self.download_root,
        }

    def _cache_key(self):
        return (
            type(self).__name__,
            self.model_name,
            self.model_checksum,
            self.fp16_encoder,
            self.use_flash,
            self.device,
            self.download_root,
        )

    def _validate_model_checksum(self, gigaam) -> None:
        if self.model_checksum is None:
            return
        local_path = os.path.expanduser(self.model_name)
        if os.path.isfile(local_path):
            digest = hashlib.md5()
            with open(local_path, "rb") as model_file:
                while chunk := model_file.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != self.model_checksum:
                raise ValueError("GigaAM model checksum does not match the configured checksum")
            return

        normalized_model_name = gigaamShortNames.get(self.model_name, self.model_name)
        package_checksum = getattr(gigaam, "_MODEL_HASHES", {}).get(normalized_model_name)
        if package_checksum is not None and package_checksum != self.model_checksum:
            raise ValueError("GigaAM package checksum does not match the configured checksum")

    def _load(self):
        import gigaam

        from audiogear.utils.runtime import normalize_device

        self._validate_model_checksum(gigaam)
        return gigaam.load_model(
            self.model_name,
            fp16_encoder=self.fp16_encoder,
            use_flash=self.use_flash,
            device=normalize_device(self.device),
            download_root=self.download_root,
        )

    @staticmethod
    def _result_text(result) -> str:
        if isinstance(result, str):
            return result.strip()

        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()

        segments = getattr(result, "segments", result if isinstance(result, list) else None)
        if segments is None:
            raise TypeError(f"Unsupported GigaAM transcription result: {type(result).__name__}")

        texts = []
        for segment in segments:
            if isinstance(segment, str):
                segment_text = segment
            elif isinstance(segment, dict):
                segment_text = segment.get("text", segment.get("transcription", ""))
            else:
                segment_text = getattr(segment, "text", None)
            if not isinstance(segment_text, str):
                raise TypeError(f"Unsupported GigaAM segment result: {type(segment).__name__}")
            if segment_text.strip():
                texts.append(segment_text.strip())
        return " ".join(texts)

    def transcribe(self, audio_file: str) -> str:
        try:
            return self._result_text(self.model.transcribe(audio_file))
        except Exception:
            # Long files exceed the short-form window.
            return self._result_text(self.model.transcribe_longform(audio_file))


class WhisperBackend(ASRBackend):
    """OpenAI Whisper via faster-whisper, pinned to an immutable HF snapshot."""

    backend_name = "whisper"

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "ru",
        compute_type: str = "int8_float16",
        name: str | None = None,
        device: str = "cuda",
        repository: str = whisperRepository,
        revision: str = whisperRevision,
        local_files_only: bool = False,
        allow_patterns: tuple[str, ...] | list[str] | None = whisperAllowPatterns,
    ):
        super().__init__(name=name, device=device)
        validate_hf_revision(revision)
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type
        self.repository = repository
        self.revision = revision
        self.local_files_only = local_files_only
        self.allow_patterns = normalize_allow_patterns(allow_patterns)

    @property
    def effective_compute_type(self) -> str:
        from audiogear.utils.runtime import normalize_device

        if normalize_device(self.device) == "cpu" and self.compute_type == "int8_float16":
            return "int8_float32"
        return self.compute_type

    @property
    def checkpoint_identity(self) -> dict:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "repository": self.repository,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "allow_patterns": self.allow_patterns,
            "compute_type": self.compute_type,
            "effective_compute_type": self.effective_compute_type,
            "device": self.device,
            "language": self.language,
        }

    def _cache_key(self):
        return (
            type(self).__name__,
            self.repository,
            self.revision,
            self.local_files_only,
            self.allow_patterns,
            self.effective_compute_type,
            self.device,
        )

    def _load(self):
        from faster_whisper import WhisperModel

        from audiogear.utils.runtime import normalize_device

        snapshot_path = resolve_hf_snapshot(
            self.repository,
            self.revision,
            self.local_files_only,
            self.allow_patterns,
        )
        return WhisperModel(
            snapshot_path,
            device=normalize_device(self.device),
            compute_type=self.effective_compute_type,
        )

    def transcribe(self, audio_file: str) -> str:
        segments, _ = self.model.transcribe(audio_file, language=self.language)
        return "".join(segment.text for segment in segments).strip()


class Wav2Vec2Backend(ASRBackend):
    """A transformers CTC ASR model, architecturally diverse from Whisper and GigaAM.

    The default model ships safetensors, which avoids restricted loading of
    older ``pytorch_model.bin`` checkpoints with some Torch versions.
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
    """T-one streaming Conformer-CTC ASR using local immutable HF artifacts.

    T-one is CPU-only. The pinned snapshot supplies both ``model.onnx`` and
    ``kenlm.bin`` to ``StreamingCTCPipeline.from_local``.
    """

    backend_name = "tone"

    def __init__(
        self,
        repository: str = toneRepository,
        revision: str = toneRevision,
        local_files_only: bool = False,
        allow_patterns: tuple[str, ...] | list[str] | None = toneAllowPatterns,
        name: str | None = None,
        device: str = "cpu",
    ):
        if str(device) != "cpu":
            raise ValueError("T-one supports only device='cpu'")
        super().__init__(name=name, device=device)
        validate_hf_revision(revision)
        self.repository = repository
        self.revision = revision
        self.local_files_only = local_files_only
        self.allow_patterns = normalize_allow_patterns(allow_patterns)

    @property
    def checkpoint_identity(self) -> dict:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "repository": self.repository,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "allow_patterns": self.allow_patterns,
            "device": self.device,
        }

    def _cache_key(self):
        return (
            type(self).__name__,
            self.repository,
            self.revision,
            self.local_files_only,
            self.allow_patterns,
            self.device,
        )

    def _load(self):
        from tone import StreamingCTCPipeline

        snapshot_path = resolve_hf_snapshot(
            self.repository,
            self.revision,
            self.local_files_only,
            self.allow_patterns,
        )
        return StreamingCTCPipeline.from_local(snapshot_path)

    def transcribe(self, audio_file: str) -> str:
        from tone import read_audio

        audio = read_audio(audio_file)
        phrases = self.model.forward_offline(audio)
        # forward_offline returns TextPhrase objects with a text field.
        return " ".join(getattr(phrase, "text", str(phrase)) for phrase in phrases).strip()
