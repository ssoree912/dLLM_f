from __future__ import annotations

from dataclasses import dataclass

import torch
from step_distill.distribution_pruning import (
    DistributionPruningController,
    straight_through_pruned_attention,
)
from torch import nn


@dataclass(frozen=True, slots=True)
class FakeAttentionConfig:
    n_heads: int = 1
    effective_n_kv_heads: int = 1
    rope: bool = False
    attention_dropout: float = 0.0


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeAttentionConfig()
        self.layer_id = 0
        self.attn_out = nn.Identity()
        self.q_norm = None
        self.k_norm = None


def test_controller_full_mode_uses_every_prompt_token() -> None:
    # Given
    controller = DistributionPruningController(prompt_length=961, layer_count=2)

    # When
    controller.enable_full_attention()

    # Then
    assert controller.is_full_attention
    assert controller.layer_scores(0) is None


def test_st_attention_matches_hard_gather_and_trains_dropped_scores() -> None:
    # Given
    block = FakeBlock()
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]])
    k = q.clone()
    v = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 4.0]]])
    scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)

    # When
    output, keep = straight_through_pruned_attention(
        block,
        q,
        k,
        v,
        attention_bias=None,
        prompt_length=3,
        selector_scores=scores,
        keep_count=2,
        gate_temperature=1.0,
    )
    selected_keys = torch.tensor([0, 1, 3])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.view(1, 1, 4, 2),
        k.index_select(1, selected_keys).view(1, 1, 3, 2),
        v.index_select(1, selected_keys).view(1, 1, 3, 2),
    ).view_as(output)
    output.square().mean().backward()

    # Then
    assert keep.tolist() == [[0, 1]]
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    assert scores.grad is not None
    assert scores.grad[0, 2].abs() > 0.0
