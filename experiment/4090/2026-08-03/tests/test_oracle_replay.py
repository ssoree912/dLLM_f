from __future__ import annotations

from dataclasses import dataclass

import torch
from step_distill.oracle_generation import (
    ReplayGenerationConfig,
    generate_offline_replay,
)
from step_distill.oracle_pruning import (
    OfflineReplayController,
    ReplayMode,
    oracle_pruned_attention,
)
from step_distill.qa_metrics import qa_scores
from torch import nn


@dataclass(frozen=True, slots=True)
class FakeAttentionConfig:
    n_heads: int = 1
    effective_n_kv_heads: int = 1
    rope: bool = False


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeAttentionConfig()
        self.layer_id = 0
        self.attn_out = nn.Identity()
        self.q_norm = None
        self.k_norm = None


@dataclass(frozen=True, slots=True)
class FakeOutput:
    logits: torch.Tensor


class FakeModel(nn.Module):
    def __init__(self, *, prompt_length: int, mask_id: int) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.mask_id = mask_id

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> FakeOutput:
        del attention_mask, use_cache, return_dict
        logits = torch.zeros((*input_ids.shape, 3), dtype=torch.float32)
        suffix = input_ids[:, self.prompt_length :]
        confidence = torch.tensor([9.0, 6.0])[: suffix.shape[1]]
        logits[:, self.prompt_length :, 1] = torch.where(
            suffix == self.mask_id,
            confidence,
            torch.zeros_like(confidence),
        )
        return FakeOutput(logits=logits)


class StepRecorder:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def set_step(self, step_id: int) -> None:
        self.steps.append(step_id)


def test_controller_uses_step_zero_only_for_static_replay() -> None:
    # Given
    order = torch.tensor([[[2, 1, 0]], [[0, 1, 2]]])
    static = OfflineReplayController(3, 2, order, ReplayMode.STATIC)
    dynamic = OfflineReplayController(3, 2, order, ReplayMode.DYNAMIC)
    static.set_step(1)
    dynamic.set_step(1)

    # When
    static_indices = static.prompt_indices(0, torch.device("cpu"))
    dynamic_indices = dynamic.prompt_indices(0, torch.device("cpu"))

    # Then
    assert static_indices.tolist() == [2, 1]
    assert dynamic_indices.tolist() == [0, 1]


def test_controller_caps_budget_at_short_prompt_length() -> None:
    # Given
    controller = OfflineReplayController(
        prompt_length=3,
        budget=512,
        order=torch.tensor([[[2, 1, 0]]]),
        mode=ReplayMode.STATIC,
    )

    # When
    indices = controller.prompt_indices(0, torch.device("cpu"))

    # Then
    assert indices.tolist() == [2, 1, 0]


def test_pruned_attention_keeps_selected_prompt_and_complete_suffix() -> None:
    # Given
    controller = OfflineReplayController(
        prompt_length=3,
        budget=1,
        order=torch.tensor([[[1]]]),
        mode=ReplayMode.DYNAMIC,
    )
    block = FakeBlock()
    q = torch.zeros((1, 4, 1))
    k = torch.zeros((1, 4, 1))
    v = torch.tensor([[[1.0], [4.0], [10.0], [20.0]]])

    # When
    output = oracle_pruned_attention(block, q, k, v, None, controller)

    # Then
    torch.testing.assert_close(output, torch.full_like(output, 12.0))


def test_generate_offline_replay_advances_controller_once_per_step() -> None:
    # Given
    mask_id = 99
    model = FakeModel(prompt_length=2, mask_id=mask_id)
    controller = StepRecorder()
    config = ReplayGenerationConfig(
        gen_length=2,
        block_length=2,
        steps=2,
        temperature=0.0,
        mask_id=mask_id,
    )

    # When
    generated = generate_offline_replay(
        model,
        torch.tensor([[5, 6]]),
        config,
        step_controller=controller,
    )

    # Then
    assert controller.steps == [0, 1]
    assert generated.tolist() == [[1, 1]]


def test_qa_scores_separate_answer_recall_from_trailing_noise_penalty() -> None:
    # Given / When
    scores = qa_scores("No extra words", "no")

    # Then
    assert scores.recall == 1.0
    assert scores.f1 == 0.5
