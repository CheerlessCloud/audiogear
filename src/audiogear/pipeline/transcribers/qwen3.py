from __future__ import annotations

from audiogear.pipeline.qwen3_snapshot import resolve_qwen_model_path
from audiogear.pipeline.transcribers.base import ASRBackend


class Qwen3ASRBackend(ASRBackend):
    backend_name = "qwen3"

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-ASR-1.7B",
        revision: str | None = None,
        language: str = "Russian",
        context: str = "",
        dtype: str = "bfloat16",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
        name: str | None = None,
        device: str = "cuda",
    ):
        super().__init__(name=name, device=device)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Qwen3 dtype: {dtype}")
        if max_inference_batch_size == 0 or max_inference_batch_size < -1:
            raise ValueError("max_inference_batch_size must be positive or -1")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.language = language
        self.context = context
        self.dtype = dtype
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens

    def _device_map(self) -> str:
        if str(self.device).startswith("cuda"):
            return "cuda:0"
        if str(self.device) == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported Qwen3 device: {self.device}")

    @property
    def checkpoint_identity(self) -> dict:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "model_name_or_path": self.model_name_or_path,
            "revision": self.revision,
            "language": self.language,
            "context": self.context,
            "dtype": self.dtype,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "device": self._device_map(),
        }

    def _cache_key(self):
        return (
            type(self).__name__,
            self.model_name_or_path,
            self.revision,
            self.dtype,
            self.max_inference_batch_size,
            self.max_new_tokens,
            self._device_map(),
        )

    def _load(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        model_path = resolve_qwen_model_path(self.model_name_or_path, self.revision)
        load_options = {
            "dtype": getattr(torch, self.dtype),
            "device_map": self._device_map(),
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        return Qwen3ASRModel.from_pretrained(model_path, **load_options)

    def transcribe(self, audio_file: str) -> str:
        results = self.model.transcribe(
            audio=[audio_file],
            context=self.context,
            language=self.language,
            return_time_stamps=False,
        )
        try:
            result_count = len(results)
        except TypeError as error:
            raise ValueError("Qwen3 ASR returned a non-sequence result") from error
        if result_count != 1:
            raise ValueError(f"Qwen3 ASR returned {result_count} results for one audio input")
        text = getattr(results[0], "text", None)
        if not isinstance(text, str):
            raise ValueError("Qwen3 ASR result has no string text field")
        return text.strip()
