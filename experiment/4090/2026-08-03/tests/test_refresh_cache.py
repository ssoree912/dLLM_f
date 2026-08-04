from __future__ import annotations

import torch
from step_distill.oracle_generation import ReplayGenerationConfig
from step_distill.refresh_cache import (
    AttentionPolicy,
    RefreshCacheConfig,
    RefreshPromptKVCache,
    RefreshTeacherTargets,
    build_refresh_targets,
)
from step_distill.refresh_cache_generation import generate_with_refresh_cache


class FakeRefreshRunner:
    def __init__(self, *, prompt_length: int, mask_id: int) -> None:
        self.prompt_length = prompt_length
        self.mask_id = mask_id
        self.step_id = 0
        self.refresh_steps: list[int] = []
        self.cached_steps: list[int] = []

    def start_step(self, step_id: int) -> bool:
        self.step_id = step_id
        return step_id % 2 == 0

    def refresh_logits(
        self,
        prompt_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        del prompt_ids
        self.refresh_steps.append(self.step_id)
        return self._logits(suffix_ids)

    def cached_logits(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        self.cached_steps.append(self.step_id)
        return self._logits(suffix_ids)

    def _logits(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*suffix_ids.shape, 3), dtype=torch.float32)
        confidence = torch.arange(
            suffix_ids.shape[1],
            0,
            -1,
            dtype=torch.float32,
        )
        logits[..., 1] = torch.where(
            suffix_ids == self.mask_id,
            confidence,
            torch.zeros_like(confidence),
        )
        return logits


def _orders() -> torch.Tensor:
    return torch.tensor(
        [
            [[4, 3, 2, 1, 0]],
            [[0, 1, 3, 2, 4]],
            [[2, 4, 1, 0, 3]],
            [[1, 0, 4, 3, 2]],
            [[0, 1, 2, 3, 4]],
        ]
    )


def _targets() -> RefreshTeacherTargets:
    order = _orders()
    ranked_scores = torch.arange(5, 0, -1, dtype=torch.float32).view(1, 1, 5)
    return build_refresh_targets(order, ranked_scores.expand_as(order))


def test_dynamic_topk_uses_current_step_within_stale_outer_cache() -> None:
    # Given
    cache = RefreshPromptKVCache(
        prompt_length=5,
        targets=_targets(),
        config=RefreshCacheConfig(
            cache_budget=3,
            attention_budget=2,
            refresh_interval=4,
            attention_policy=AttentionPolicy.DYNAMIC_TOPK,
        ),
    )
    cache.start_step(0)

    # When
    cache.start_step(1)
    selected = cache.attention_prompt_indices(0, torch.device("cpu"))

    # Then
    assert selected.tolist() == [3, 2]


def test_outer_cache_is_reselected_only_on_refresh_steps() -> None:
    # Given
    cache = RefreshPromptKVCache(
        prompt_length=5,
        targets=_targets(),
        config=RefreshCacheConfig(
            cache_budget=3,
            attention_budget=2,
            refresh_interval=4,
            attention_policy=AttentionPolicy.DYNAMIC_TOPK,
        ),
    )

    # When
    first_refresh = cache.start_step(0)
    initial = cache.cached_prompt_indices(torch.device("cpu"))
    stale_refresh = cache.start_step(3)
    stale = cache.cached_prompt_indices(torch.device("cpu"))
    next_refresh = cache.start_step(4)
    updated = cache.cached_prompt_indices(torch.device("cpu"))

    # Then
    assert first_refresh
    assert not stale_refresh
    assert next_refresh
    assert initial.tolist() == [4, 3, 2]
    assert stale.tolist() == [4, 3, 2]
    assert updated.tolist() == [0, 1, 2]


def test_all_cache_policy_attends_every_cached_prompt_position() -> None:
    # Given
    cache = RefreshPromptKVCache(
        prompt_length=5,
        targets=_targets(),
        config=RefreshCacheConfig(
            cache_budget=3,
            attention_budget=2,
            refresh_interval=4,
            attention_policy=AttentionPolicy.ALL_CACHE,
        ),
    )
    cache.start_step(0)

    # When
    selected = cache.attention_prompt_indices(0, torch.device("cpu"))

    # Then
    assert selected.tolist() == [4, 3, 2]


def test_outer_cache_uses_global_mean_teacher_score() -> None:
    # Given
    dense_scores = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [0.0, 8.0, 7.0, 0.0]]])
    order = torch.argsort(dense_scores, dim=-1, descending=True, stable=True)
    targets = build_refresh_targets(order, dense_scores.gather(-1, order))
    cache = RefreshPromptKVCache(
        prompt_length=4,
        targets=targets,
        config=RefreshCacheConfig(
            cache_budget=2,
            attention_budget=2,
            refresh_interval=4,
            attention_policy=AttentionPolicy.ALL_CACHE,
        ),
    )

    # When
    cache.start_step(0)
    selected = cache.cached_prompt_indices(torch.device("cpu"))

    # Then
    assert selected.tolist() == [0, 1]


def test_generation_refreshes_on_the_runner_schedule() -> None:
    # Given
    mask_id = 99
    runner = FakeRefreshRunner(prompt_length=2, mask_id=mask_id)
    config = ReplayGenerationConfig(
        gen_length=4,
        block_length=4,
        steps=4,
        temperature=0.0,
        mask_id=mask_id,
    )

    # When
    generated = generate_with_refresh_cache(
        runner,
        torch.tensor([[7, 8]]),
        config,
    )

    # Then
    assert runner.refresh_steps == [0, 2]
    assert runner.cached_steps == [1, 3]
    assert generated.tolist() == [[1, 1, 1, 1]]
