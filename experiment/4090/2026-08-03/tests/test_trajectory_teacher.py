from __future__ import annotations

import torch
from step_distill.trajectory_teacher import (
    build_step_target,
    pool_pre_step_context,
    select_candidates,
)


def test_build_step_target_normalizes_confidence_inside_each_step() -> None:
    # Given
    prompt_attention = torch.tensor(
        [
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ]
    )
    positions = torch.tensor([0, 1])
    confidence = torch.tensor([0.75, 0.25])

    # When
    target = build_step_target(
        prompt_attention,
        positions,
        confidence,
        confidence_weight=True,
    )

    # Then
    torch.testing.assert_close(target, torch.tensor([[0.7, 0.3]]))


def test_pool_pre_step_context_uses_query_when_no_suffix_is_committed() -> None:
    # Given
    layer_inputs = torch.tensor(
        [
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
            ]
        ]
    )
    committed = torch.tensor([False, False])
    question_indices = torch.tensor([0, 1])

    # When
    context = pool_pre_step_context(
        layer_inputs,
        prompt_length=2,
        committed_suffix=committed,
        question_indices=question_indices,
    )

    # Then
    torch.testing.assert_close(context, torch.tensor([[2.0, 0.0]]))


def test_pool_pre_step_context_uses_only_already_committed_suffix() -> None:
    # Given
    layer_inputs = torch.tensor(
        [
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [10.0, 2.0],
                [20.0, 4.0],
            ]
        ]
    )
    committed = torch.tensor([False, True])
    question_indices = torch.tensor([0, 1])

    # When
    context = pool_pre_step_context(
        layer_inputs,
        prompt_length=2,
        committed_suffix=committed,
        question_indices=question_indices,
    )

    # Then
    torch.testing.assert_close(context, torch.tensor([[20.0, 4.0]]))


def test_select_candidates_at_zero_temperature_does_not_exponentiate_logits() -> None:
    # Given
    logits = torch.tensor([[[999.0, 1000.0]]])
    suffix_ids = torch.tensor([[126336]])
    mask_index = torch.tensor([[True]])

    # When
    token_ids, _confidence = select_candidates(
        logits,
        suffix_ids,
        mask_index,
        temperature=0.0,
    )

    # Then
    assert token_ids.item() == 1
