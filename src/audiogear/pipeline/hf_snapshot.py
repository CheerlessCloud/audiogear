from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

fullRevisionPattern = re.compile(r"[0-9a-f]{40}")


def validate_hf_revision(revision: str) -> str:
    if not isinstance(revision, str) or fullRevisionPattern.fullmatch(revision) is None:
        raise ValueError("Hugging Face revision must be a full 40-character lowercase hexadecimal commit hash")
    return revision


def normalize_allow_patterns(allow_patterns: Sequence[str] | None) -> tuple[str, ...] | None:
    if allow_patterns is None:
        return None
    patterns = tuple(allow_patterns)
    if not patterns or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        raise ValueError("Hugging Face allow patterns must be nonempty strings")
    return patterns


@lru_cache(maxsize=None)
def _resolve_hf_snapshot(
    repository: str,
    revision: str,
    local_files_only: bool,
    allow_patterns: tuple[str, ...] | None,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repository,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
    )


def resolve_hf_snapshot(
    repository: str,
    revision: str,
    local_files_only: bool = False,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("Hugging Face repository must be a nonempty string")
    validate_hf_revision(revision)
    return _resolve_hf_snapshot(
        repository,
        revision,
        local_files_only,
        normalize_allow_patterns(allow_patterns),
    )


def clear_hf_snapshot_cache() -> None:
    _resolve_hf_snapshot.cache_clear()
