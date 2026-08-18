from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from dllm_cache.budget.future_pool_teacher import FuturePoolTeacherConfig, generate_with_future_pool_teacher
from dllm_cache.budget.offline_hybrid_teacher import (
    OfflineHybridCollector,
    OfflineHybridTeacherConfig,
    generate_with_offline_hybrid_teacher,
    normalized_prompt_movement,
)


class FakeBlock(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = SimpleNamespace(
            n_heads=1,
            effective_n_kv_heads=1,
            rope=False,
        )
        self.q_norm = None
        self.k_norm = None

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        block_mask: torch.Tensor | None = None,
    ):
        del k, v, attention_bias, layer_past, use_cache, block_mask
        return q, None


class FakeTransformer(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(layer_id) for layer_id in range(layer_count)])


class FakeDecoder(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.transformer = FakeTransformer(layer_count)


class FakeModel(nn.Module):
    """Small deterministic model exposing the module path used by LLaDA."""

    def __init__(self, prompt_length: int, layer_count: int = 2, vocab_size: int = 10) -> None:
        super().__init__()
        self.model = FakeDecoder(layer_count)
        self.prompt_length = prompt_length
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ):
        del attention_mask, use_cache, return_dict
        state = input_ids.float()
        suffix_context = state[:, self.prompt_length :].sum(dim=1, keepdim=True)
        prompt_context = torch.cat(
            [
                suffix_context.expand(-1, self.prompt_length),
                torch.zeros_like(state[:, self.prompt_length :]),
            ],
            dim=1,
        )
        for layer_id, block in enumerate(self.model.transformer.blocks):
            # Prompt K/V changes whenever another suffix token is committed.
            layer_scale = float(layer_id + 1)
            q = (state + 1.0 + layer_scale).unsqueeze(-1)
            k = (state + 2.0 + prompt_context * (0.05 * layer_scale)).unsqueeze(-1)
            v = (state + 3.0 + prompt_context * (0.10 * layer_scale)).unsqueeze(-1)
            block.attention(q, k, v)
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., 1] = 5.0
        return SimpleNamespace(logits=logits)


def test_normalized_prompt_movement_matches_relative_l2_per_position() -> None:
    previous = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    current = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
    movement = normalized_prompt_movement(current, previous)
    assert torch.allclose(movement, torch.tensor([0.5, 0.5]))


def test_collector_accumulates_kv_path_length_and_step_max() -> None:
    block = FakeBlock(layer_id=0)
    collector = OfflineHybridCollector(prompt_length=2, suffix_length=1, layer_count=1)
    q = torch.ones((1, 3, 1))

    collector.capture(
        block,
        q,
        torch.tensor([[[1.0], [2.0], [1.0]]]),
        torch.tensor([[[2.0], [4.0], [1.0]]]),
        None,
    )
    collector.capture(
        block,
        q,
        torch.tensor([[[2.0], [4.0], [1.0]]]),
        torch.tensor([[[4.0], [8.0], [1.0]]]),
        None,
    )
    collector.capture(
        block,
        q,
        torch.tensor([[[4.0], [4.0], [1.0]]]),
        torch.tensor([[[4.0], [16.0], [1.0]]]),
        None,
    )

    delta_sum, delta_max, observations = collector.delta_result()
    assert observations == 2
    assert torch.allclose(delta_sum, torch.tensor([[0.75, 0.75]]))
    assert torch.allclose(delta_max, torch.tensor([[0.5, 0.5]]))


def test_reference_signal_matches_existing_future_pool_teacher() -> None:
    prompt_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    old_model = FakeModel(prompt_length=3)
    new_model = FakeModel(prompt_length=3)
    old_config = FuturePoolTeacherConfig(
        gen_length=4,
        block_length=2,
        steps=4,
        active_top_k=2,
        temperature=0.0,
        confidence_weight=True,
        target_aggregation="sum",
        mask_id=9,
    )
    new_config = OfflineHybridTeacherConfig(
        gen_length=4,
        block_length=2,
        steps=4,
        active_top_k=2,
        temperature=0.0,
        confidence_weight=True,
        target_aggregation="sum",
        mask_id=9,
    )

    old_result = generate_with_future_pool_teacher(old_model, prompt_ids, old_config)
    new_result = generate_with_offline_hybrid_teacher(new_model, prompt_ids, new_config)

    assert torch.equal(new_result.generated_ids, old_result.generated_ids)
    assert torch.allclose(new_result.teacher_raw, old_result.teacher_raw)
    assert torch.allclose(new_result.teacher_norm, old_result.teacher_norm)
    assert torch.equal(new_result.ref_union_mask, old_result.union_mask)
    assert torch.equal(new_result.ref_union_size_by_layer, old_result.union_size_by_layer)
    assert new_result.commit_count == old_result.commit_count
    assert new_result.weight_sum == old_result.weight_sum
    assert new_result.reference_step_count == old_result.frequency_denominator
    assert new_result.delta_observation_count == new_config.steps - 1
    assert new_result.delta_raw.shape == (2, 3)
    assert torch.all(new_result.delta_raw >= 0)
    assert torch.any(new_result.delta_raw > 0)
    assert not hasattr(new_result, "future_step_masks")


def test_attention_method_is_restored_after_generation() -> None:
    model = FakeModel(prompt_length=2, layer_count=1)
    block = model.model.transformer.blocks[0]
    original_function = block.attention.__func__
    config = OfflineHybridTeacherConfig(
        gen_length=2,
        block_length=2,
        steps=2,
        active_top_k=1,
        temperature=0.0,
        confidence_weight=False,
        target_aggregation="max",
        mask_id=9,
    )
    generate_with_offline_hybrid_teacher(model, torch.tensor([[2, 3]]), config)
    assert block.attention.__func__ is original_function


def test_lifetime_mask_reference_uses_all_active_steps_without_topk_support() -> None:
    prompt_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    commit_result = generate_with_offline_hybrid_teacher(
        FakeModel(prompt_length=3),
        prompt_ids,
        OfflineHybridTeacherConfig(
            gen_length=4,
            block_length=2,
            steps=4,
            active_top_k=0,
            temperature=0.0,
            confidence_weight=False,
            target_aggregation="sum",
            reference_query_mode="commit",
            mask_id=9,
        ),
    )
    lifetime_result = generate_with_offline_hybrid_teacher(
        FakeModel(prompt_length=3),
        prompt_ids,
        OfflineHybridTeacherConfig(
            gen_length=4,
            block_length=2,
            steps=4,
            active_top_k=0,
            temperature=0.0,
            confidence_weight=False,
            target_aggregation="sum",
            reference_query_mode="lifetime_mask",
            mask_id=9,
        ),
    )

    assert lifetime_result.reference_step_count == 4
    assert lifetime_result.commit_count == 4
    assert lifetime_result.weight_sum == 6.0
    assert torch.all(lifetime_result.ref_union_mask)
    assert torch.equal(lifetime_result.ref_union_size_by_layer, torch.tensor([3, 3]))
    assert torch.all(lifetime_result.teacher_raw > commit_result.teacher_raw)
    expected_norm = lifetime_result.teacher_raw / lifetime_result.teacher_raw.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    assert torch.allclose(lifetime_result.teacher_norm, expected_norm)
    assert lifetime_result.delta_observation_count == 3


if __name__ == "__main__":
    test_normalized_prompt_movement_matches_relative_l2_per_position()
    test_collector_accumulates_kv_path_length_and_step_max()
    test_reference_signal_matches_existing_future_pool_teacher()
    test_attention_method_is_restored_after_generation()
    test_lifetime_mask_reference_uses_all_active_steps_without_topk_support()
    print("ok")
