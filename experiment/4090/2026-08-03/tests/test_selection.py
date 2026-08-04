from __future__ import annotations

import torch
from step_distill.selection import greedy_mmr_order, stable_top_order


def test_stable_top_order_keeps_score_order_when_scores_tie() -> None:
    # Given
    scores = torch.tensor([0.8, 0.8, 0.2])

    # When
    order = stable_top_order(scores, max_k=3)

    # Then
    assert order.tolist() == [0, 1, 2]


def test_greedy_mmr_order_avoids_a_redundant_second_token() -> None:
    # Given
    scores = torch.tensor([1.0, 0.9, 0.8])
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # When
    order = greedy_mmr_order(scores, embeddings, max_k=3, gamma=0.5)

    # Then
    assert order.tolist() == [0, 2, 1]


def test_zero_gamma_mmr_matches_plain_order() -> None:
    # Given
    scores = torch.tensor([0.2, 0.9, 0.5])
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # When
    order = greedy_mmr_order(scores, embeddings, max_k=3, gamma=0.0)

    # Then
    assert order.tolist() == [1, 2, 0]
