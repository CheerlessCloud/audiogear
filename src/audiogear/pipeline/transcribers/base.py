"""ASR backend interface for multi-model consensus transcription.

A backend wraps a single ASR model behind a uniform ``transcribe(path) -> str``
call with lazy model loading (so it can be pickled to a worker before a GPU is
pinned). Add a new backend by subclassing :class:`ASRBackend` and implementing
``_load`` + ``transcribe`` — then list it in the consensus config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audiogear.utils.runtime import cached_model


class ASRBackend(ABC):
    """One ASR model. Subclasses implement lazy loading + transcription."""

    #: short, stable identifier used in metadata keys (e.g. ``asr_text_gigaam``)
    backend_name: str = "asr"

    def __init__(self, name: str | None = None, device: str = "cuda"):
        self.name = name or self.backend_name
        self.device = device

    def _cache_key(self):
        """Identity for the process-global model cache. Override if two backends
        of the same class differ by checkpoint (e.g. ``model_name``)."""
        return (type(self).__name__, self.backend_name, self.device)

    @property
    def checkpoint_identity(self) -> dict:
        options = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))
        }
        return {"type": type(self).__name__, **options}

    @property
    def model(self):
        # Process-global cache so the model loads once per worker, not once per
        # task (EXPERIENCE ❌5). Keeps the backend instance cheap to pickle.
        return cached_model(self._cache_key(), self._load)

    @abstractmethod
    def _load(self):
        """Construct and return the underlying model, on ``self.device``."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        """Return the raw transcript for ``audio_file``."""
        raise NotImplementedError
