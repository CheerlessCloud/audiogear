"""audiogear — a configurable pipeline for preparing/annotating speech datasets for TTS.

Importing this package is intentionally lightweight: it loads environment
variables from a local ``.env`` (so ``HF_TOKEN`` and friends are available) but
does NOT import torch/transformers/heavy metric backends. Those are pulled in
lazily by the individual pipeline blocks when constructed.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Load .env once, as early as possible, without overriding real env vars.
try:  # python-dotenv is a core dependency, but stay defensive.
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except Exception:  # pragma: no cover
    pass

from audiogear.data import AudioPipeline, AudioSegment  # noqa: E402

__all__ = ["AudioSegment", "AudioPipeline", "__version__"]
