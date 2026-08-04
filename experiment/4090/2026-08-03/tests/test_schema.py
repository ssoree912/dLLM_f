from __future__ import annotations

from pathlib import Path

import pytest
import torch
from step_distill.schema import (
    SchemaError,
    ShardMetadata,
    TeacherShard,
    load_teacher_shard,
    save_teacher_shard_atomic,
)


def make_shard() -> TeacherShard:
    return TeacherShard(
        sample_id="sample/1",
        dataset="2wikimqa",
        prompt_input_ids=torch.tensor([10, 11, 12]),
        question_token_indices=torch.tensor([1, 2]),
        generated_input_ids=torch.tensor([20, 21]),
        valid_step_mask=torch.tensor([True, True]),
        commit_positions=torch.tensor([[0, -1], [1, -1]]),
        commit_counts=torch.tensor([1, 1]),
        commit_confidence=torch.tensor([[0.8, 0.0], [0.7, 0.0]]),
        context_pre=torch.ones((2, 1, 4)),
        top_order=torch.tensor([[[0, 1]], [[1, 0]]]),
        candidate_scores=torch.tensor([[[0.8, 0.7]], [[0.9, 0.6]]]),
        metadata=ShardMetadata(
            max_length=2048,
            gen_length=2,
            steps=2,
            block_length=2,
            confidence_weight=True,
            context_timing="pre_step",
        ),
    )


def test_teacher_shard_preserves_step_axis() -> None:
    # Given / When
    shard = make_shard()

    # Then
    assert shard.step_count == 2
    assert shard.top_order.shape == (2, 1, 2)
    assert shard.context_pre.shape == (2, 1, 4)


def test_teacher_shard_rejects_mismatched_step_axis() -> None:
    # Given
    shard = make_shard()

    # When / Then
    with pytest.raises(SchemaError, match="step axis"):
        TeacherShard(
            sample_id=shard.sample_id,
            dataset=shard.dataset,
            prompt_input_ids=shard.prompt_input_ids,
            question_token_indices=shard.question_token_indices,
            generated_input_ids=shard.generated_input_ids,
            valid_step_mask=torch.tensor([True]),
            commit_positions=shard.commit_positions,
            commit_counts=shard.commit_counts,
            commit_confidence=shard.commit_confidence,
            context_pre=shard.context_pre,
            top_order=shard.top_order,
            candidate_scores=shard.candidate_scores,
            metadata=shard.metadata,
        )


def test_atomic_shard_round_trip_revalidates_content(tmp_path: Path) -> None:
    # Given
    shard = make_shard()
    path = tmp_path / "sample.pt"

    # When
    save_teacher_shard_atomic(shard, path)
    loaded = load_teacher_shard(path, expected_sample_id="sample/1")

    # Then
    assert loaded.sample_id == shard.sample_id
    torch.testing.assert_close(loaded.top_order, shard.top_order)


def test_teacher_shard_record_omits_redundant_diversity_order() -> None:
    # Given
    shard = make_shard()

    # When
    record = shard.to_record()

    # Then
    assert "diverse_order" not in record
    assert record["schema_version"] == 3


def test_load_teacher_shard_accepts_legacy_plain_target(tmp_path: Path) -> None:
    # Given
    shard = make_shard()
    record = dict(shard.to_record())
    record["schema_version"] = 2
    record["diverse_order"] = shard.top_order.clone()
    record["metadata"] = {
        **record["metadata"],
        "gamma": 0.0,
        "similarity_source": "prompt_prefill_hidden",
    }
    path = tmp_path / "legacy.pt"
    torch.save(record, path)

    # When
    loaded = load_teacher_shard(path)

    # Then
    torch.testing.assert_close(loaded.top_order, shard.top_order)
    assert loaded.metadata == shard.metadata
