"""The two baselines must differ exactly where they are supposed to.

`refresh` leaves untouched positions holding whatever they last wrote, so their
measured movement keeps growing; `step` rebases everything each call, so an
unselected position reports only its most recent hop. Both run inside the same
GPU forward, so the update rule is checked here on plain tensors.
"""

from __future__ import annotations

import torch

from dllm_cache.budget.layer_split_prompt_kv import LayerSplitPromptCache


def make_cache(baseline: str) -> LayerSplitPromptCache:
    cache = LayerSplitPromptCache(
        prompt_length=8,
        budget=4,
        frozen_layers=0,
        keep_indices=torch.arange(4),
        refresh_indices=torch.arange(4),
        stale_indices=torch.empty(0, dtype=torch.long),
        measured_tokens=2,
        measured_baseline=baseline,
    )
    cache.attn[0] = torch.zeros(1, 4, 3)
    return cache


def apply(cache: LayerSplitPromptCache, att: torch.Tensor, due: torch.Tensor) -> None:
    """The baseline write from _run_measured_block, isolated."""
    slots = due.view(1, -1, 1).expand(1, -1, att.shape[-1])
    if cache.measured_baseline == "step":
        cache.attn[0].copy_(att)
    else:
        cache.attn[0].scatter_(1, slots, torch.gather(att, 1, slots))


def test_refresh_keeps_unselected_positions_stale() -> None:
    cache = make_cache("refresh")
    att = torch.tensor([[[1.0, 1, 1], [2.0, 2, 2], [3.0, 3, 3], [4.0, 4, 4]]])
    apply(cache, att, torch.tensor([0, 1]))
    # Selected rows took the new value; the rest still hold the prefill zeros, so
    # their next measurement is against a baseline two steps old.
    assert torch.equal(cache.attn[0][0, 0], torch.tensor([1.0, 1, 1]))
    assert torch.equal(cache.attn[0][0, 2], torch.zeros(3))
    assert torch.equal(cache.attn[0][0, 3], torch.zeros(3))


def test_step_rebases_every_position() -> None:
    cache = make_cache("step")
    att = torch.tensor([[[1.0, 1, 1], [2.0, 2, 2], [3.0, 3, 3], [4.0, 4, 4]]])
    apply(cache, att, torch.tensor([0, 1]))
    assert torch.equal(cache.attn[0], att)


def test_refresh_accumulates_across_steps_but_step_does_not() -> None:
    # A position that is never selected drifts by 1.0 per step for three steps.
    drifting = 3
    for baseline, expected in (("refresh", 3.0), ("step", 1.0)):
        cache = make_cache(baseline)
        for step in range(1, 4):
            att = torch.zeros(1, 4, 3)
            att[0, drifting] = float(step)
            before = cache.attn[0][0, drifting].clone()
            apply(cache, att, torch.tensor([0, 1]))
            moved = float((att[0, drifting] - before).abs().max())
        assert moved == expected, (baseline, moved)


if __name__ == "__main__":
    test_refresh_keeps_unselected_positions_stale()
    test_step_rebases_every_position()
    test_refresh_accumulates_across_steps_but_step_does_not()
    print("ok")
