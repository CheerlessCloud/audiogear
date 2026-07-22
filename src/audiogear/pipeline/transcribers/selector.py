from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from audiogear.data import AudioPipeline, AudioSegment
from audiogear.pipeline.base import PipelineStep
from audiogear.pipeline.transcribers.selection import TranscriptionCandidate, select_candidate
from audiogear.utils.progress import tqdm

candidateStatuses = frozenset({"ok", "no_speech", "error"})


class ConsensusSelector(PipelineStep):
    type = "🗣️ - TRANSCRIBER"
    name = "🗳️ ASR consensus selector"

    def __init__(
        self,
        candidates: Sequence[Mapping[str, str]],
        prefer_punctuated: bool = True,
        overwrite_text: bool = False,
        medoid_candidate_id_column: str = "asr_medoid_candidate_id",
        selected_candidate_id_column: str = "asr_selected_candidate_id",
        selected_text_column: str = "asr_selected_text",
        eligible_candidate_ids_column: str = "asr_eligible_candidate_ids",
        pairwise_distances_column: str = "asr_pairwise_distances",
        mean_distances_column: str = "asr_mean_distances",
        agreement_column: str = "asr_agreement",
        punctuation_candidate_ids_column: str = "asr_punctuation_candidate_ids",
        punctuation_changed_selection_column: str = "asr_punctuation_changed_selection",
        selection_status_column: str = "asr_selection_status",
    ):
        super().__init__()
        self.candidates = self._validate_candidate_config(candidates)
        self.prefer_punctuated = prefer_punctuated
        self.overwrite_text = overwrite_text
        self.medoid_candidate_id_column = medoid_candidate_id_column
        self.selected_candidate_id_column = selected_candidate_id_column
        self.selected_text_column = selected_text_column
        self.eligible_candidate_ids_column = eligible_candidate_ids_column
        self.pairwise_distances_column = pairwise_distances_column
        self.mean_distances_column = mean_distances_column
        self.agreement_column = agreement_column
        self.punctuation_candidate_ids_column = punctuation_candidate_ids_column
        self.punctuation_changed_selection_column = punctuation_changed_selection_column
        self.selection_status_column = selection_status_column
        if any(not isinstance(column, str) or not column for column in self.output_columns):
            raise ValueError("Consensus output column names must be nonempty strings")
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError("Consensus output column names must be unique")

    @staticmethod
    def _validate_candidate_config(
        candidates: Sequence[Mapping[str, str]],
    ) -> tuple[tuple[str, str, str], ...]:
        if not candidates:
            raise ValueError("ConsensusSelector requires at least one candidate")
        configured_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("Consensus candidate configurations must be mappings")
            candidate_id = candidate.get("candidate_id")
            status_column = candidate.get("status_column")
            text_column = candidate.get("text_column")
            values = (candidate_id, status_column, text_column)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("Each consensus candidate requires candidate_id, status_column, and text_column")
            configured_candidates.append(values)

        candidate_ids = [candidate_id for candidate_id, _, _ in configured_candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Consensus candidate IDs must be unique")
        input_columns = [
            column
            for _, status_column, text_column in configured_candidates
            for column in (status_column, text_column)
        ]
        if len(set(input_columns)) != len(input_columns):
            raise ValueError("Consensus candidate status and text columns must be unique")
        return tuple(configured_candidates)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (
            self.medoid_candidate_id_column,
            self.selected_candidate_id_column,
            self.selected_text_column,
            self.eligible_candidate_ids_column,
            self.pairwise_distances_column,
            self.mean_distances_column,
            self.agreement_column,
            self.punctuation_candidate_ids_column,
            self.punctuation_changed_selection_column,
            self.selection_status_column,
        )

    @staticmethod
    def _metadata_value(segment: AudioSegment, candidate_id: str, column: str):
        if column not in segment.metadata:
            raise KeyError(f"Candidate {candidate_id!r} is missing metadata column {column!r} for id={segment.id}")
        return segment.metadata[column]

    def _read_candidates(self, segment: AudioSegment) -> list[TranscriptionCandidate]:
        persisted_candidates = []
        for candidate_id, status_column, text_column in self.candidates:
            status = self._metadata_value(segment, candidate_id, status_column)
            text = self._metadata_value(segment, candidate_id, text_column)
            if not isinstance(status, str) or status not in candidateStatuses:
                raise ValueError(f"Candidate {candidate_id!r} has invalid status for id={segment.id}")
            if not isinstance(text, str):
                raise ValueError(f"Candidate {candidate_id!r} has non-string text for id={segment.id}")
            if status == "ok" and not text.strip():
                raise ValueError(f"Candidate {candidate_id!r} has status 'ok' with blank text for id={segment.id}")
            if status != "ok" and text != "":
                raise ValueError(
                    f"Candidate {candidate_id!r} has status {status!r} with nonempty text for id={segment.id}"
                )
            persisted_candidates.append(TranscriptionCandidate(candidate_id, status, text))
        return persisted_candidates

    @staticmethod
    def _compact_json(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _assign(self, segment: AudioSegment) -> None:
        candidates = self._read_candidates(segment)
        candidate_order = [candidate.candidate_id for candidate in candidates]
        selection = select_candidate(candidates, candidate_order, self.prefer_punctuated)
        pairwise_distances = [
            {"candidate_ids": [first_id, second_id], "distance": distance}
            for (first_id, second_id), distance in selection.pairwise_distances.items()
        ]
        segment.metadata[self.medoid_candidate_id_column] = selection.medoid_candidate_id or ""
        segment.metadata[self.selected_candidate_id_column] = selection.selected_candidate_id or ""
        segment.metadata[self.selected_text_column] = selection.selected_text
        segment.metadata[self.eligible_candidate_ids_column] = self._compact_json(selection.eligible_candidate_ids)
        segment.metadata[self.pairwise_distances_column] = self._compact_json(pairwise_distances)
        segment.metadata[self.mean_distances_column] = self._compact_json(selection.mean_distances)
        segment.metadata[self.agreement_column] = selection.agreement
        segment.metadata[self.punctuation_candidate_ids_column] = self._compact_json(
            selection.punctuation_candidate_ids
        )
        segment.metadata[self.punctuation_changed_selection_column] = selection.prefer_punctuated_changed_selection
        segment.metadata[self.selection_status_column] = selection.status
        if self.overwrite_text and selection.selected_text:
            segment.text = selection.selected_text

    def run(self, data: AudioPipeline, rank: int = 0, world_size: int = 1) -> AudioPipeline:
        for segment in tqdm(data):
            self._assign(segment)
        return data
