from __future__ import annotations

import json
import math
import threading

import numpy as np

from audiogear.data import AudioSegment
from audiogear.pipeline.checkpoint import fingerprint_audio_file
from audiogear.pipeline.metrics.base import BaseMetric
from audiogear.pipeline.readers.base import BaseDiskReader
from audiogear.pipeline.writers.base_disk import DiskWriter
from audiogear.utils.runtime import cached_model

senkoCacheLock = threading.Lock()


class SenkoDiarizationMetric(BaseMetric):
    name = "🗣 Senko Diarization"
    senkoCommit = "ba0e12ed923ff49e8c2d9d9a3e42d7923cb95724"
    semanticMappingVersion = 2
    gpu = True
    prefetch = True

    def __init__(
        self,
        device: str = "cuda",
        vad: str = "silero",
        clustering: str = "cpu",
        warmup: bool = False,
        accurate: bool = False,
        mer_cos: float | None = None,
        sample_rate: int = 16000,
        segments_column: str = "senko_segments",
        raw_segments_column: str = "senko_raw_segments",
        num_speakers_column: str = "senko_num_speakers",
        timing_column: str = "senko_timing",
        status_column: str = "senko_status",
        max_consecutive_failures: int = 50,
        checkpoint_folder=None,
        file_writer: DiskWriter = None,
        file_reader: BaseDiskReader = None,
    ):
        normalized_device = self._normalize_device(device)
        if vad not in {"auto", "pyannote", "silero"}:
            raise ValueError(f"Unsupported Senko VAD: {vad}")
        if clustering not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unsupported Senko clustering device: {clustering}")
        if not isinstance(warmup, bool):
            raise ValueError("Senko warmup must be a boolean")
        if not isinstance(accurate, bool):
            raise ValueError("Senko accurate must be a boolean")
        if mer_cos is not None and not 0 < mer_cos <= 1:
            raise ValueError("Senko mer_cos must be greater than 0 and at most 1")
        if sample_rate != 16000:
            raise ValueError("Senko diarize_samples requires sample_rate=16000")

        output_columns = (
            segments_column,
            raw_segments_column,
            num_speakers_column,
            timing_column,
            status_column,
        )
        checkpoint_identity = json.dumps(
            {
                "senko_commit": self.senkoCommit,
                "semantic_mapping_version": self.semanticMappingVersion,
                "device": normalized_device,
                "vad": vad,
                "clustering": clustering,
                "warmup": warmup,
                "accurate": accurate,
                "mer_cos": mer_cos,
                "sample_rate": sample_rate,
                "output_columns": output_columns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(
            metric=output_columns,
            file_writer=file_writer,
            file_reader=file_reader,
            max_consecutive_failures=max_consecutive_failures,
            checkpoint_folder=checkpoint_folder,
            checkpoint_identity=checkpoint_identity,
        )
        self.device = device
        self.vad = vad
        self.clustering = clustering
        self.warmup = warmup
        self.accurate = accurate
        self.mer_cos = mer_cos
        self.sample_rate = sample_rate

    @staticmethod
    def _normalize_device(device: str) -> str:
        if str(device).startswith("cuda"):
            return "cuda"
        if str(device) == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported Senko device: {device}")

    def checkpoint_input_fingerprint(self, segment: AudioSegment) -> str:
        return fingerprint_audio_file(segment.audio_file)

    def checkpoint_result_is_resumable(self, segment: AudioSegment) -> bool:
        return segment.metadata.get(self.metric[4]) != "error"

    def _runtime_on(self, device: str):
        normalized_device = self._normalize_device(device)

        def build():
            import senko

            diarizer = senko.Diarizer(
                device=normalized_device,
                vad=self.vad,
                clustering=self.clustering,
                warmup=self.warmup,
                quiet=True,
                mer_cos=self.mer_cos,
            )
            return diarizer, threading.Lock()

        cache_key = (
            type(self).__name__,
            normalized_device,
            self.vad,
            self.clustering,
            self.warmup,
            self.accurate,
            self.mer_cos,
        )
        with senkoCacheLock:
            return cached_model(cache_key, build)

    def _prepare(self, segment: AudioSegment):
        import soundfile
        from scipy.signal import resample_poly

        source_samples, source_sample_rate = soundfile.read(
            segment.audio_file,
            dtype="float32",
            always_2d=True,
        )
        if source_sample_rate <= 0:
            raise ValueError(f"Invalid source sample rate: {source_sample_rate}")
        if not np.all(np.isfinite(source_samples)):
            raise ValueError("Source audio samples must be finite")
        if source_samples.size and float(np.max(np.abs(source_samples))) > 1.0:
            raise ValueError("Source audio samples must be normalized to [-1, 1]")

        mono_samples = source_samples.mean(axis=1, dtype=np.float32)
        if source_sample_rate != self.sample_rate and mono_samples.size:
            divisor = math.gcd(int(source_sample_rate), self.sample_rate)
            mono_samples = resample_poly(
                mono_samples,
                self.sample_rate // divisor,
                int(source_sample_rate) // divisor,
            )
            if not np.all(np.isfinite(mono_samples)):
                raise ValueError("Resampled audio samples must be finite")
            mono_samples = np.clip(mono_samples, -1.0, 1.0)

        samples = np.ascontiguousarray(mono_samples, dtype=np.float32)
        return samples, segment.audio_file

    @staticmethod
    def _serialize_segments(result, key: str) -> str:
        segments = result.get(key)
        if not isinstance(segments, list):
            raise ValueError(f"Senko result {key} must be a list")
        serialized_segments = []
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(f"Senko {key} segment {segment_index} must be a dictionary")
            speaker = segment.get("speaker")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError(f"Senko {key} segment {segment_index} has an invalid speaker")
            try:
                raw_start = float(segment["start"])
                raw_end = float(segment["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Senko {key} segment {segment_index} has invalid timestamps") from error
            span_is_valid = math.isfinite(raw_start) and math.isfinite(raw_end) and 0 <= raw_start < raw_end
            if not span_is_valid:
                raise ValueError(f"Senko {key} segment {segment_index} has invalid span {raw_start}..{raw_end}")
            serialized_segments.append(
                {
                    "speaker": speaker,
                    "start": round(raw_start, 3),
                    "end": round(raw_end, 3),
                }
            )
        return json.dumps(serialized_segments, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _serialize_timing(result) -> str:
        timing = result.get("timing_stats")
        if not isinstance(timing, dict):
            raise ValueError("Senko result timing_stats must be a dictionary")
        if not timing:
            raise ValueError("Senko result timing_stats must be nonempty")
        timing_values_are_finite = all(
            isinstance(timing_value, (int, float))
            and not isinstance(timing_value, bool)
            and math.isfinite(timing_value)
            for timing_value in timing.values()
        )
        if not timing_values_are_finite:
            raise ValueError("Senko result timing_stats must contain finite numeric values")
        try:
            return json.dumps(
                timing,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Senko result timing_stats must contain finite JSON data") from error

    def _map_result(self, result):
        if result is None:
            return "[]", "[]", 0, "{}", "no_speech"
        if not isinstance(result, dict):
            raise ValueError("Senko diarize_samples must return a dictionary or None")

        segments = self._serialize_segments(result, "merged_segments")
        raw_segments = self._serialize_segments(result, "raw_segments")
        num_speakers = result.get("merged_speakers_detected")
        if isinstance(num_speakers, bool) or not isinstance(num_speakers, int) or num_speakers < 0:
            raise ValueError("Senko result merged_speakers_detected must be a nonnegative integer")
        timing = self._serialize_timing(result)

        merged_segment_values = result["merged_segments"]
        raw_segment_values = result["raw_segments"]
        distinct_merged_speakers = len({segment["speaker"] for segment in merged_segment_values})
        if num_speakers != distinct_merged_speakers:
            raise ValueError(
                "Senko result merged_speakers_detected must equal the number of distinct merged speaker labels"
            )
        if merged_segment_values and raw_segment_values:
            return segments, raw_segments, num_speakers, timing, "ok"
        if not merged_segment_values and raw_segment_values:
            return segments, raw_segments, num_speakers, timing, "raw_only"
        raise ValueError("Senko result must contain raw segments and may contain merged segments")

    def _infer_on(self, prepared, device: str):
        samples, source_name = prepared
        diarizer, inference_lock = self._runtime_on(device)
        with inference_lock:
            result = diarizer.diarize_samples(
                samples,
                sample_rate=self.sample_rate,
                accurate=self.accurate,
                generate_colors=False,
                source_name=source_name,
            )
        return self._map_result(result)

    def _infer(self, prepared):
        return self._infer_on(prepared, self.device)

    def compute_metric(self, segment: AudioSegment):
        return self._infer(self._prepare(segment))

    def compute_metric_cpu(self, segment: AudioSegment):
        return self._infer_on(self._prepare(segment), "cpu")

    def _failed_value(self):
        return "[]", "[]", 0, "{}", "error"
