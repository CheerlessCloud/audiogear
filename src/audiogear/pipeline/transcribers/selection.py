from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import combinations
from typing import Any

from audiogear.pipeline.metrics.wer import normalize_text

punctuationPattern = re.compile(r"[.,!?…;:—]")


@dataclass(frozen=True)
class TranscriptionCandidate:
    candidate_id: str
    status: str
    text: str


@dataclass(frozen=True)
class SelectionResult(Mapping[str, Any]):
    eligible_candidate_ids: tuple[str, ...]
    pairwise_distances: dict[tuple[str, str], float]
    mean_distances: dict[str, float]
    medoid_candidate_id: str | None
    selected_candidate_id: str | None
    selected_text: str
    agreement: float | None
    status: str
    punctuation_candidate_ids: tuple[str, ...]
    prefer_punctuated_changed_selection: bool

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "eligible_ids": "eligible_candidate_ids",
            "means": "mean_distances",
            "medoid": "medoid_candidate_id",
            "selected_candidate": "selected_candidate_id",
            "selection_status": "status",
            "punctuation_candidates": "punctuation_candidate_ids",
        }
        try:
            return getattr(self, aliases.get(key, key))
        except AttributeError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[str]:
        return (field.name for field in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return self.eligible_candidate_ids

    @property
    def means(self) -> dict[str, float]:
        return self.mean_distances

    @property
    def medoid(self) -> str | None:
        return self.medoid_candidate_id

    @property
    def selected_candidate(self) -> str | None:
        return self.selected_candidate_id

    @property
    def selection_status(self) -> str:
        return self.status

    @property
    def punctuation_candidates(self) -> tuple[str, ...]:
        return self.punctuation_candidate_ids


def _coerce_candidate(candidate_id: str, candidate: Any) -> TranscriptionCandidate:
    if isinstance(candidate, TranscriptionCandidate):
        if candidate.candidate_id != candidate_id:
            raise ValueError(f"Candidate ID mismatch: {candidate_id!r} != {candidate.candidate_id!r}")
        return candidate
    if isinstance(candidate, Mapping):
        mapped_id = candidate.get("candidate_id", candidate.get("id", candidate_id))
        if mapped_id != candidate_id:
            raise ValueError(f"Candidate ID mismatch: {candidate_id!r} != {mapped_id!r}")
        status = candidate.get("status", "")
        text = candidate.get("text", "")
        if not isinstance(status, str) or not isinstance(text, str):
            raise TypeError("Candidate status and text must be strings")
        return TranscriptionCandidate(candidate_id=candidate_id, status=status, text=text)
    raise TypeError("Candidates must be TranscriptionCandidate objects or mappings")


def _ordered_candidates(
    candidates: Mapping[str, Any] | Sequence[TranscriptionCandidate | Mapping[str, Any]],
    candidate_order: Sequence[str],
) -> list[TranscriptionCandidate]:
    order = list(candidate_order)
    if len(order) != len(set(order)):
        raise ValueError("Candidate order must not contain duplicate IDs")

    if isinstance(candidates, Mapping):
        unexpected_ids = set(candidates).difference(order)
        if unexpected_ids:
            raise ValueError(f"Candidate order is missing IDs: {', '.join(sorted(unexpected_ids))}")
        return [
            _coerce_candidate(candidate_id, candidates[candidate_id])
            for candidate_id in order
            if candidate_id in candidates
        ]

    by_id: dict[str, TranscriptionCandidate] = {}
    for candidate in candidates:
        if isinstance(candidate, TranscriptionCandidate):
            candidate_id = candidate.candidate_id
            coerced = candidate
        elif isinstance(candidate, Mapping):
            candidate_id = candidate.get("candidate_id", candidate.get("id"))
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("Candidate mappings in a sequence require a nonempty candidate_id")
            coerced = _coerce_candidate(candidate_id, candidate)
        else:
            raise TypeError("Candidate sequences must contain TranscriptionCandidate objects or mappings")
        if candidate_id in by_id:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        by_id[candidate_id] = coerced
    unexpected_ids = set(by_id).difference(order)
    if unexpected_ids:
        raise ValueError(f"Candidate order is missing IDs: {', '.join(sorted(unexpected_ids))}")
    return [by_id[candidate_id] for candidate_id in order if candidate_id in by_id]


def _pair_distance(first: str, second: str) -> float:
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0

    import jiwer

    first_to_second = float(jiwer.cer(first, second))
    second_to_first = float(jiwer.cer(second, first))
    return min(1.0, (first_to_second + second_to_first) / 2.0)


def select_candidate(
    candidates: Mapping[str, Any] | Sequence[TranscriptionCandidate | Mapping[str, Any]],
    candidate_order: Sequence[str],
    prefer_punctuated: bool = True,
) -> SelectionResult:
    ordered = _ordered_candidates(candidates, candidate_order)
    eligible = [candidate for candidate in ordered if candidate.status == "ok" and candidate.text.strip()]
    eligible_ids = tuple(candidate.candidate_id for candidate in eligible)
    punctuation_ids = tuple(
        candidate.candidate_id for candidate in eligible if punctuationPattern.search(candidate.text) is not None
    )

    if not eligible:
        return SelectionResult(
            eligible_candidate_ids=(),
            pairwise_distances={},
            mean_distances={},
            medoid_candidate_id=None,
            selected_candidate_id=None,
            selected_text="",
            agreement=None,
            status="no_candidate",
            punctuation_candidate_ids=(),
            prefer_punctuated_changed_selection=False,
        )

    normalized = {candidate.candidate_id: normalize_text(candidate.text) for candidate in eligible}
    pairwise_distances: dict[tuple[str, str], float] = {}
    distance_sums = dict.fromkeys(eligible_ids, 0.0)
    for first_id, second_id in combinations(eligible_ids, 2):
        distance = _pair_distance(normalized[first_id], normalized[second_id])
        pairwise_distances[(first_id, second_id)] = distance
        distance_sums[first_id] += distance
        distance_sums[second_id] += distance

    divisor = max(1, len(eligible_ids) - 1)
    mean_distances = {candidate_id: distance_sums[candidate_id] / divisor for candidate_id in eligible_ids}
    medoid_id = min(eligible_ids, key=mean_distances.__getitem__)
    selected_id = medoid_id
    if prefer_punctuated and medoid_id not in punctuation_ids and punctuation_ids:
        selected_id = min(punctuation_ids, key=mean_distances.__getitem__)

    candidate_by_id = {candidate.candidate_id: candidate for candidate in eligible}
    agreement = None
    status = "single_candidate"
    if len(eligible_ids) >= 2:
        agreement = max(0.0, 1.0 - mean_distances[medoid_id])
        status = "ok"

    return SelectionResult(
        eligible_candidate_ids=eligible_ids,
        pairwise_distances=pairwise_distances,
        mean_distances=mean_distances,
        medoid_candidate_id=medoid_id,
        selected_candidate_id=selected_id,
        selected_text=candidate_by_id[selected_id].text,
        agreement=agreement,
        status=status,
        punctuation_candidate_ids=punctuation_ids,
        prefer_punctuated_changed_selection=selected_id != medoid_id,
    )
