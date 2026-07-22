from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def resolve_qwen_model_path(model_name_or_path: str, revision: str | None) -> str:
    if revision is None:
        return model_name_or_path

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name_or_path, revision=revision)
