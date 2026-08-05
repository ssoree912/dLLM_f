from __future__ import annotations

import torch
from step_distill.distribution_rollout import pool_causal_state
from torch import nn


def test_causal_state_uses_request_then_only_committed_suffix() -> None:
    # Given
    embedding = nn.Embedding.from_pretrained(
        torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [4.0, 4.0],
                [8.0, 0.0],
                [0.0, 8.0],
            ]
        )
    )
    prompt_ids = torch.tensor([[0, 1, 2]])
    initial_indices = torch.tensor([1, 2])
    all_masked = torch.tensor([[5, 5]])
    partly_committed = torch.tensor([[3, 5]])

    # When
    initial = pool_causal_state(
        embedding,
        prompt_ids,
        all_masked,
        initial_indices,
        mask_id=5,
    )
    later = pool_causal_state(
        embedding,
        prompt_ids,
        partly_committed,
        initial_indices,
        mask_id=5,
    )

    # Then
    torch.testing.assert_close(initial, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(later, torch.tensor([4.0, 4.0]))
