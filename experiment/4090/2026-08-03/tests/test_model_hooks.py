from __future__ import annotations

from dataclasses import dataclass

import torch
from step_distill.model_hooks import SuffixPromptAttentionCollector
from torch import nn


@dataclass(frozen=True, slots=True)
class FakeAttentionConfig:
    n_heads: int = 1
    effective_n_kv_heads: int = 1
    rope: bool = False


class FakeAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeAttentionConfig()
        self.layer_id = 0


def test_collector_matches_full_softmax_then_prompt_slice() -> None:
    # Given
    block = FakeAttentionBlock()
    collector = SuffixPromptAttentionCollector(
        prompt_length=2,
        suffix_length=1,
        layer_count=1,
    )
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = q.clone()
    expected = torch.softmax(q[:, 2:, :] @ k.transpose(-1, -2) / (2.0**0.5), dim=-1)[
        0, :, :2
    ]

    # When
    collector.capture(block, q, k, attention_bias=None)
    captured = collector.stacked_attention()

    # Then
    torch.testing.assert_close(captured, expected.unsqueeze(0))
