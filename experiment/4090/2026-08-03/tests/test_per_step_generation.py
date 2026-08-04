from __future__ import annotations

from dataclasses import dataclass

import torch
from step_distill.teacher_generation import (
    PerStepTeacherConfig,
    generate_per_step_teacher,
)
from torch import nn


@dataclass(frozen=True, slots=True)
class FakeAttentionConfig:
    n_heads: int = 1
    effective_n_kv_heads: int = 1
    rope: bool = False


@dataclass(frozen=True, slots=True)
class FakeOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeAttentionConfig()
        self.layer_id = 0

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        *,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None]:
        del k, v, attention_bias, layer_past, use_cache
        return q, None


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock()])


class FakeInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = FakeTransformer()


class FakeModel(nn.Module):
    def __init__(self, *, prompt_length: int, mask_id: int) -> None:
        super().__init__()
        self.model = FakeInnerModel()
        self.prompt_length = prompt_length
        self.mask_id = mask_id

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        output_hidden_states: bool,
    ) -> FakeOutput:
        del attention_mask, use_cache, return_dict, output_hidden_states
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1)
        token_signal = input_ids.remainder(7).float().unsqueeze(-1)
        hidden = torch.cat([positions + 1.0, token_signal + 1.0], dim=-1)
        block = self.model.transformer.blocks[0]
        block.attention(hidden, hidden, hidden, None, layer_past=None, use_cache=False)

        logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
        if input_ids.shape[1] > self.prompt_length:
            suffix = input_ids[:, self.prompt_length :]
            logits[:, self.prompt_length :, 1] = torch.where(
                suffix == self.mask_id,
                torch.tensor([9.0, 6.0]),
                torch.tensor([0.0, 0.0]),
            )
        return FakeOutput(logits=logits, hidden_states=(hidden, hidden))


def test_generate_per_step_teacher_keeps_distinct_t_targets_and_pre_states() -> None:
    # Given
    mask_id = 99
    model = FakeModel(prompt_length=2, mask_id=mask_id)
    prompt_ids = torch.tensor([[5, 6]])
    question_indices = torch.tensor([1])
    config = PerStepTeacherConfig(
        gen_length=2,
        block_length=2,
        steps=2,
        temperature=0.0,
        confidence_weight=True,
        max_target_k=2,
        diversity_gamma=0.1,
        mask_id=mask_id,
    )

    # When
    result = generate_per_step_teacher(model, prompt_ids, question_indices, config)

    # Then
    assert result.step_scores.shape == (2, 1, 2)
    assert result.context_pre.shape == (2, 1, 2)
    assert result.commit_positions[:, 0].tolist() == [0, 1]
    assert not torch.equal(result.context_pre[0], result.context_pre[1])
    assert result.top_order.shape == (2, 1, 2)
