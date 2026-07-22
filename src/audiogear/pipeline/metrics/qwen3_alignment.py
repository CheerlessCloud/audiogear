from __future__ import annotations

import hashlib
import json
import math

from audiogear.data import AudioSegment
from audiogear.pipeline.checkpoint import fingerprint_audio_file
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.qwen3_snapshot import resolve_qwen_model_path
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model


class Qwen3ForcedAlignmentMetric(BaseMetric):
    name = "⏱ Qwen3 Forced Alignment"
    gpu = True
    supports_batch = True

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        revision: str | None = None,
        language: str = "Russian",
        dtype: str = "bfloat16",
        device: str = "cuda",
        alignment_column: str = "qwen3_alignment",
        status_column: str = "qwen3_alignment_status",
        batch_size: int = 1,
        max_batch_seconds: float = 300.0,
        max_consecutive_failures: int = 50,
        checkpoint_folder=None,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Qwen3 dtype: {dtype}")
        output_columns = (alignment_column, status_column)
        normalized_device = self._device_map(device)
        effective_dtype = "float32" if normalized_device == "cpu" else dtype
        checkpoint_identity = json.dumps(
            {
                "model_name_or_path": model_name_or_path,
                "revision": revision,
                "language": language,
                "device": normalized_device,
                "dtype": dtype,
                "effective_dtype": effective_dtype,
                "batch_size": batch_size,
                "max_batch_seconds": max_batch_seconds,
                "output_columns": output_columns,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(
            metric=output_columns,
            file_writer=file_writer,
            file_reader=file_reader,
            batch_size=batch_size,
            max_batch_seconds=max_batch_seconds,
            max_consecutive_failures=max_consecutive_failures,
            checkpoint_folder=checkpoint_folder,
            checkpoint_identity=checkpoint_identity,
        )
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.language = language
        self.dtype = dtype
        self.device = device

    def checkpoint_input_fingerprint(self, segment: AudioSegment) -> str:
        input_identity = json.dumps(
            {
                "audio_sha256": fingerprint_audio_file(segment.audio_file),
                "text": segment.text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(input_identity.encode("utf-8")).hexdigest()

    def checkpoint_result_is_resumable(self, segment: AudioSegment) -> bool:
        return segment.metadata.get(self.metric[1]) != "error"

    def _device_map(self, device: str) -> str:
        if str(device).startswith("cuda"):
            return "cuda:0"
        if str(device) == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported Qwen3 device: {device}")

    def _model_on(self, device: str):
        device_map = self._device_map(device)
        resolved_dtype = "float32" if device_map == "cpu" else self.dtype

        def build():
            import torch
            from qwen_asr import Qwen3ForcedAligner

            model_path = resolve_qwen_model_path(self.model_name_or_path, self.revision)
            load_options = {
                "dtype": getattr(torch, resolved_dtype),
                "device_map": device_map,
            }
            return Qwen3ForcedAligner.from_pretrained(model_path, **load_options)

        cache_key = (
            type(self).__name__,
            self.model_name_or_path,
            self.revision,
            resolved_dtype,
            device_map,
        )
        return cached_model(cache_key, build)

    def _failed_value(self):
        return "[]", "error"

    @staticmethod
    def _serialize_result(result) -> str:
        items = getattr(result, "items", None)
        if items is None:
            raise ValueError("Qwen3 forced aligner result has no items field")
        words = []
        previous_end = 0.0
        for item_index, item in enumerate(items):
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                raise ValueError(f"Qwen3 alignment item {item_index} has no string text field")
            try:
                raw_start = float(item.start_time)
                raw_end = float(item.end_time)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Qwen3 alignment item {item_index} has invalid timestamps") from error
            timestamps_are_valid = (
                math.isfinite(raw_start)
                and math.isfinite(raw_end)
                and raw_start >= 0
                and raw_start <= raw_end
            )
            if not timestamps_are_valid:
                raise ValueError(f"Qwen3 alignment item {item_index} has invalid span {raw_start}..{raw_end}")
            if raw_start < previous_end:
                raise ValueError(f"Qwen3 alignment item {item_index} is not monotonic")
            words.append({"text": text, "start": round(raw_start, 3), "end": round(raw_end, 3)})
            previous_end = raw_end
        return json.dumps(words, ensure_ascii=False, separators=(",", ":"))

    def _align(self, segments: list[AudioSegment], device: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str] | None] = [None] * len(segments)
        nonempty_positions = []
        for index, segment in enumerate(segments):
            if (segment.text or "").strip():
                nonempty_positions.append(index)
                continue
            results[index] = ("[]", "empty_text")
        if nonempty_positions:
            model = self._model_on(device)
            aligned = model.align(
                audio=[segments[index].audio_file for index in nonempty_positions],
                text=[segments[index].text for index in nonempty_positions],
                language=[self.language] * len(nonempty_positions),
            )
            try:
                aligned_count = len(aligned)
            except TypeError as error:
                raise ValueError("Qwen3 forced aligner returned a non-sequence result") from error
            if aligned_count != len(nonempty_positions):
                raise ValueError(
                    f"Qwen3 forced aligner returned {aligned_count} results "
                    f"for {len(nonempty_positions)} inputs"
                )
            for position, result in zip(nonempty_positions, aligned):
                results[position] = (self._serialize_result(result), "ok")
        if any(result is None for result in results):
            raise RuntimeError("Qwen3 forced alignment result mapping is incomplete")
        return results

    def compute_batch(self, segments: list[AudioSegment]):
        return self._align(segments, self.device)

    def compute_metric(self, segment: AudioSegment):
        return self.compute_batch([segment])[0]

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._align([segment], "cpu")[0]
