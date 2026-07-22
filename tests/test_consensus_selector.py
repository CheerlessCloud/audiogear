import json

import pytest
from conftest import make_segment
from hydra.utils import instantiate
from omegaconf import OmegaConf

from audiogear.pipeline.transcribers.selector import ConsensusSelector

candidate_columns = [
    {
        "candidate_id": candidate_id,
        "status_column": f"asr_{candidate_id}_status",
        "text_column": f"asr_{candidate_id}_text",
    }
    for candidate_id in ("gigaam", "whisper", "tone", "qwen3")
]


def _persisted_segment():
    segment = make_segment("clip", text="original")
    persisted = {
        "gigaam": ("ok", "привет мир"),
        "whisper": ("ok", "Привет, мир!"),
        "tone": ("ok", "привет мир"),
        "qwen3": ("ok", "привет мир"),
    }
    for candidate_id, (status, text) in persisted.items():
        segment.metadata[f"asr_{candidate_id}_status"] = status
        segment.metadata[f"asr_{candidate_id}_text"] = text
    return segment


def test_four_persisted_candidates_reduce_with_explicit_selection_fields():
    segment = _persisted_segment()
    selector = ConsensusSelector(candidate_columns, overwrite_text=True)

    assert selector.run([segment]) == [segment]

    assert segment.metadata["asr_medoid_candidate_id"] == "gigaam"
    assert segment.metadata["asr_selected_candidate_id"] == "whisper"
    assert segment.metadata["asr_selected_text"] == "Привет, мир!"
    assert json.loads(segment.metadata["asr_eligible_candidate_ids"]) == [
        "gigaam",
        "whisper",
        "tone",
        "qwen3",
    ]
    assert len(json.loads(segment.metadata["asr_pairwise_distances"])) == 6
    assert json.loads(segment.metadata["asr_mean_distances"]) == {
        "gigaam": 0.0,
        "whisper": 0.0,
        "tone": 0.0,
        "qwen3": 0.0,
    }
    assert segment.metadata["asr_agreement"] == 1.0
    assert json.loads(segment.metadata["asr_punctuation_candidate_ids"]) == ["whisper"]
    assert segment.metadata["asr_punctuation_changed_selection"] is True
    assert segment.metadata["asr_selection_status"] == "ok"
    assert segment.text == "Привет, мир!"


def test_selector_is_instantiable_from_hydra_config():
    config = OmegaConf.create(
        {
            "_target_": "audiogear.pipeline.transcribers.selector.ConsensusSelector",
            "candidates": candidate_columns,
            "overwrite_text": True,
        }
    )

    selector = instantiate(config)

    assert isinstance(selector, ConsensusSelector)
    assert selector.output_columns[0] == "asr_medoid_candidate_id"


@pytest.mark.parametrize(
    ("metadata_change", "error_type", "message"),
    [
        ({"asr_qwen3_text": None}, KeyError, "missing metadata column"),
        ({"asr_tone_status": "ok", "asr_tone_text": ""}, ValueError, "status 'ok' with blank text"),
        ({"asr_gigaam_status": "error", "asr_gigaam_text": "private"}, ValueError, "nonempty text"),
    ],
)
def test_selector_rejects_missing_or_contradictory_candidate_columns(metadata_change, error_type, message):
    segment = _persisted_segment()
    for column, value in metadata_change.items():
        if value is None:
            segment.metadata.pop(column)
        else:
            segment.metadata[column] = value

    with pytest.raises(error_type, match=message):
        ConsensusSelector(candidate_columns).run([segment])
