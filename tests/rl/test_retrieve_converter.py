from __future__ import annotations

from llm_gateway.rl.slime_train.retrieve.convert_to_slime_format import (  # noqa: E402
    build_fs_structure_from_state,
    convert_sample,
)


def test_build_fs_structure_from_state_uses_metadata_description():
    fs_state = {
        "files": {
            "people/alex/preferences.md": '{"description": "Alex preferences"}\nlikes pasta',
            "timeline.md": "plain content",
        }
    }

    tree = build_fs_structure_from_state(fs_state)

    assert "filesystem/" in tree
    assert "preferences.md — Alex preferences" in tree
    assert "timeline.md" in tree


def test_convert_sample_fill_question_metadata():
    sample = convert_sample(
        {"files": {"people/alex/preferences.md": '{"description": "Alex preferences"}\nlikes pasta'}},
        {
            "question_type": "fill_in_the_blank",
            "probe_query": "What food does Alex like?",
            "alt_queries": ["Alex favorite food?"],
            "ground_truth": ["handmade pasta", "Italian food"],
            "source_evidence": "likes pasta",
            "probe_type": "atomic",
            "answerable": True,
            "generation_mode": "unit-test",
        },
        {
            "snapshot_id": "snap1",
            "trajectory_id": "traj1",
            "user_id": "user1",
            "session_id": "session1",
        },
    )

    assert sample["label"] is None
    assert sample["metadata"]["snapshot_id"] == "snap1"
    assert sample["metadata"]["ground_truth"]["answer_type"] == "fill"
    assert sample["metadata"]["ground_truth"]["answer"] == "handmade pasta"
    assert sample["metadata"]["ground_truth"]["acceptable_answers"] == ["handmade pasta", "Italian food"]
    assert sample["prompt"][0]["role"] == "system"
    assert sample["prompt"][1]["role"] == "user"


def test_convert_sample_mcq_question_metadata():
    sample = convert_sample(
        {"files": {}},
        {
            "question_type": "multiple_choice",
            "probe_query": "Which option is correct?",
            "ground_truth": "B",
            "options": {"A": "tea", "B": "coffee"},
        },
        {"snapshot_id": "snap1", "trajectory_id": "traj1"},
    )

    gt = sample["metadata"]["ground_truth"]
    assert gt["answer_type"] == "mcq"
    assert gt["answer"] == "B"
    assert gt["correct_option"] == "B"
    assert gt["options"] == ["A. tea", "B. coffee"]
