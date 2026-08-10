"""The periodic full refresh must fire on a fixed period and never on step 0.

Step 0 runs against caches the prefill just wrote, so a full pass there would
repeat exact work. Everything else about measured mode is a GPU forward; the
schedule is the part that can be checked without one.
"""

from __future__ import annotations

import torch

from dllm_cache.budget.layer_split_prompt_kv import LayerSplitPromptCache


def make_cache(interval: int) -> LayerSplitPromptCache:
    return LayerSplitPromptCache(
        prompt_length=64,
        budget=32,
        frozen_layers=16,
        keep_indices=torch.arange(32),
        refresh_indices=torch.arange(32),
        stale_indices=torch.empty(0, dtype=torch.long),
        measured_tokens=8,
        full_refresh_interval=interval,
    )


def fired(interval: int, steps: int) -> list[int]:
    cache = make_cache(interval)
    hits = []
    for _ in range(steps):
        if cache.full_refresh_due():
            hits.append(cache.step)
        cache.step += 1
    return hits


def test_disabled_never_fires() -> None:
    assert fired(0, 32) == []


def test_period_is_exact_and_skips_step_zero() -> None:
    assert fired(8, 33) == [8, 16, 24, 32]
    assert fired(4, 17) == [4, 8, 12, 16]


def test_interval_one_refreshes_every_step_but_the_first() -> None:
    assert fired(1, 5) == [1, 2, 3, 4]


def test_128_steps_at_interval_8_costs_15_full_passes() -> None:
    # 15/128 = 11.7% of steps pay the full deep-layer prompt forward.
    assert len(fired(8, 128)) == 15


if __name__ == "__main__":
    test_disabled_never_fires()
    test_period_is_exact_and_skips_step_zero()
    test_interval_one_refreshes_every_step_but_the_first()
    test_128_steps_at_interval_8_costs_15_full_passes()
    print("ok")
