from __future__ import annotations

import hashlib
import json

from loguru import logger

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.checkpoint import MetricCheckpoint, fingerprint_audio_file, inputFingerprintField
from audiogear.pipeline.transcribers.base import ASRBackend
from audiogear.utils.progress import tqdm
from audiogear.utils.runtime import free_cuda, is_oom_error

resumableStatuses = frozenset({"ok", "no_speech"})


class CandidateTranscriber(PipelineStep):
    type = "🗣️ - TRANSCRIBER"
    checkpoint_capable = True
    name = "📝 ASR candidate"

    def __init__(
        self,
        backend: ASRBackend,
        candidate_id_column: str = "asr_candidate_id",
        status_column: str = "asr_status",
        text_column: str = "asr_text",
        error_code_column: str = "asr_error_code",
        checkpoint_folder=None,
        file_writer=None,
        file_reader=None,
    ):
        super().__init__()
        columns = (candidate_id_column, status_column, text_column, error_code_column)
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("Candidate output column names must be nonempty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("Candidate output column names must be unique")
        self.backend = backend
        self.candidate_id_column = candidate_id_column
        self.status_column = status_column
        self.text_column = text_column
        self.error_code_column = error_code_column
        self.checkpoint_folder = checkpoint_folder
        self.file_writer = file_writer
        self.file_reader = file_reader
        self._checkpoint = None

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (
            self.candidate_id_column,
            self.status_column,
            self.text_column,
            self.error_code_column,
        )

    @property
    def checkpoint_identity(self) -> str:
        identity = {
            "backend": self.backend.checkpoint_identity,
            "output_columns": self.output_columns,
        }
        return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def error_code(error: BaseException) -> str:
        if is_oom_error(error):
            return "out_of_memory"
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            return "dependency_error"
        return "inference_error"

    def _assign(self, segment: AudioSegment, status: str, text: str = "", error_code: str = "") -> None:
        segment.metadata[self.candidate_id_column] = self.backend.name
        segment.metadata[self.status_column] = status
        segment.metadata[self.text_column] = text
        segment.metadata[self.error_code_column] = error_code

    def _process(self, segment: AudioSegment) -> None:
        try:
            text = self.backend.transcribe(segment.audio_file)
        except Exception as error:
            if is_oom_error(error):
                free_cuda()
            error_code = self.error_code(error)
            logger.warning(
                f"ASR backend {self.backend.name} failed on id={segment.id} ({segment.audio_file}) with {error_code}"
            )
            self._assign(segment, "error", error_code=error_code)
            return

        if not isinstance(text, str):
            logger.warning(f"ASR backend {self.backend.name} returned an invalid result for id={segment.id}")
            self._assign(segment, "error", error_code="invalid_result")
            return

        text = text.strip()
        if not text:
            self._assign(segment, "no_speech")
            return
        self._assign(segment, "ok", text=text)

    def _open_checkpoint(self, rank: int) -> dict[str, dict]:
        identity_hash = hashlib.sha256(self.checkpoint_identity.encode("utf-8")).hexdigest()
        slug = f"{type(self).__name__}.{self.status_column}.{identity_hash}"
        self._checkpoint = MetricCheckpoint(self.checkpoint_folder, slug, rank)
        return self._checkpoint.load()

    def _restore_checkpoint(self, segment: AudioSegment, row: dict, input_fingerprint: str) -> bool:
        stored_fingerprint = row.pop(inputFingerprintField, None)
        if stored_fingerprint != input_fingerprint:
            return False
        if row.get(self.status_column) not in resumableStatuses:
            return False
        values = {column: row.get(column) for column in self.output_columns}
        if values[self.candidate_id_column] != self.backend.name or values[self.error_code_column] != "":
            return False
        text = values[self.text_column]
        if values[self.status_column] == "ok" and (not isinstance(text, str) or not text.strip()):
            return False
        if values[self.status_column] == "no_speech" and text != "":
            return False
        segment.metadata.update(values)
        return True

    def _append_checkpoint(self, segment: AudioSegment, input_fingerprint: str) -> None:
        if segment.metadata[self.status_column] not in resumableStatuses:
            return
        values = {column: segment.metadata[column] for column in self.output_columns}
        self._checkpoint.append(segment.id, values, input_fingerprint=input_fingerprint)

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        _ = self.backend.model
        cache = self._open_checkpoint(rank) if self.checkpoint_folder else {}
        try:
            for segment in tqdm(data):
                input_fingerprint = fingerprint_audio_file(segment.audio_file) if self._checkpoint else None
                row = cache.get(str(segment.id))
                if row is not None and self._restore_checkpoint(segment, row, input_fingerprint):
                    continue
                self._process(segment)
                if self._checkpoint is not None:
                    self._append_checkpoint(segment, input_fingerprint)
                if self.file_writer:
                    self.file_writer.write(segment)
        finally:
            if self._checkpoint is not None:
                self._checkpoint.close()
                self._checkpoint = None
        return data
