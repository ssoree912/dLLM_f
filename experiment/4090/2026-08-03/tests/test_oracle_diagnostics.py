from __future__ import annotations

import torch
from step_distill.oracle_diagnostics import effective_budget, temporal_metrics


def test_temporal_metrics_report_adjacent_change_and_union_growth() -> None:
    # Given
    order = torch.tensor(
        [
            [[0, 1]],
            [[1, 2]],
            [[2, 3]],
        ]
    )
    valid_steps = torch.tensor([True, True, True])

    # When
    metrics = temporal_metrics(order, valid_steps, budget=2)

    # Then
    assert metrics.adjacent_jaccard == 1.0 / 3.0
    assert metrics.early_late_jaccard == 0.0
    assert metrics.union_ratio == 2.0


def test_effective_budget_caps_request_at_short_prompt_order_width() -> None:
    # Given / When
    budget = effective_budget(requested=64, prompt_length=55, order_width=55)

    # Then
    assert budget == 55
