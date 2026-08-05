from __future__ import annotations

import torch
from step_distill.distribution_student import (
    SelectorConfig,
    StateConditionedSelector,
    commit_ranking_loss,
    distribution_kd_loss,
    distribution_kl,
)


def test_selector_scores_each_layer_and_changes_with_state() -> None:
    # Given
    torch.manual_seed(7)
    selector = StateConditionedSelector(
        SelectorConfig(layer_count=2, hidden_dim=4, projection_dim=3, mlp_dim=5)
    )
    prompt_features = torch.randn(2, 6, 4)
    first_state = torch.zeros(4)
    later_state = torch.ones(4)

    # When
    first_scores = selector(prompt_features, first_state)
    later_scores = selector(prompt_features, later_state)

    # Then
    assert first_scores.shape == (2, 6)
    assert not torch.equal(first_scores, later_scores)


def test_distribution_kd_is_zero_for_identical_logits() -> None:
    # Given
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    uncommitted = torch.tensor([[True, True, False]])
    teacher_commit = torch.tensor([[False, True, False]])

    # When
    loss = distribution_kd_loss(
        logits,
        logits,
        uncommitted,
        teacher_commit,
        temperature=2.0,
        commit_weight=3.0,
    )

    # Then
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0.0)


def test_commit_ranking_penalizes_reversed_teacher_order() -> None:
    # Given
    teacher_commit = torch.tensor([[True, False, False]])
    eligible = torch.tensor([[True, True, True]])
    correct = torch.tensor([[0.9, 0.4, 0.3]])
    reversed_order = torch.tensor([[0.2, 0.8, 0.7]])

    # When
    correct_loss = commit_ranking_loss(
        correct, teacher_commit, eligible, margin=0.1
    )
    reversed_loss = commit_ranking_loss(
        reversed_order, teacher_commit, eligible, margin=0.1
    )

    # Then
    torch.testing.assert_close(correct_loss, torch.zeros_like(correct_loss))
    assert reversed_loss > 0.0


def test_distribution_kl_is_unweighted_over_uncommitted_positions() -> None:
    # Given
    full = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    pruned = torch.tensor([[[0.0, 2.0], [0.0, 2.0], [9.0, -9.0]]])
    uncommitted = torch.tensor([[True, True, False]])
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(pruned[:, :2], dim=-1),
        torch.softmax(full[:, :2], dim=-1),
        reduction="batchmean",
    ) / 2

    # When
    actual = distribution_kl(full, pruned, uncommitted, temperature=1.0)

    # Then
    torch.testing.assert_close(actual, expected)
