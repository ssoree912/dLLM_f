from __future__ import annotations

import torch
from step_distill import selection
from step_distill.selection import stable_top_order


def test_stable_top_order_keeps_score_order_when_scores_tie() -> None:
    # Given
    scores = torch.tensor([0.8, 0.8, 0.2])

    # When
    order = stable_top_order(scores, max_k=3)

    # Then
    assert order.tolist() == [0, 1, 2]


def test_selection_module_exposes_no_diversity_selector() -> None:
    # Given / When / Then
    assert not hasattr(selection, "greedy_mmr_order")
