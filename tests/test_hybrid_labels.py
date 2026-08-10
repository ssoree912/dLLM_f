from __future__ import annotations

import torch

from dllm_cache.budget.hybrid_labels import (
    build_hybrid_label_masks,
    mask_jaccard,
    sample_generator,
)


def scores_pair() -> tuple[torch.Tensor, torch.Tensor]:
    # Reference favors low indices, delta favors high indices, so the two
    # top-k sets are disjoint and every variant is distinguishable.
    ref = torch.arange(8.0, 0.0, -1.0).repeat(2, 1)
    delta = torch.arange(1.0, 9.0).repeat(2, 1)
    return ref, delta


def test_every_variant_meets_the_budget_exactly() -> None:
    ref, delta = scores_pair()
    result = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(0, "s"))
    for name, mask in result.masks.items():
        assert mask.shape == ref.shape, name
        assert mask.sum(dim=-1).tolist() == [4, 4], name


def test_reference_delta_takes_ref_topk_then_delta_topk_from_rest() -> None:
    ref, delta = scores_pair()
    result = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(0, "s"))
    expected = torch.zeros((2, 8), dtype=torch.bool)
    expected[:, [0, 1]] = True  # reference top-2
    expected[:, [6, 7]] = True  # delta top-2 outside the reference set
    assert torch.equal(result.masks["reference_delta"], expected)


def test_ref_ratio_one_degenerates_to_reference_only() -> None:
    ref, delta = scores_pair()
    result = build_hybrid_label_masks(ref, delta, budget=4, ref_k=4, generator=sample_generator(0, "s"))
    assert torch.equal(result.masks["reference_delta"], result.masks["reference_only"])
    assert torch.equal(result.masks["reference_random"], result.masks["reference_only"])


def test_random_fill_is_deterministic_per_seed_and_sample() -> None:
    ref, delta = scores_pair()
    first = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(7, "abc"))
    second = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(7, "abc"))
    other = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(7, "xyz"))
    assert torch.equal(first.masks["reference_random"], second.masks["reference_random"])
    assert first.masks["reference_random"][:, [0, 1]].all()
    assert not torch.equal(first.masks["reference_random"], other.masks["reference_random"])
    assert other.masks["reference_random"].sum(dim=-1).tolist() == [4, 4]


def test_budget_larger_than_prompt_selects_everything() -> None:
    ref, delta = scores_pair()
    result = build_hybrid_label_masks(ref, delta, budget=100, ref_k=75, generator=sample_generator(0, "s"))
    for mask in result.masks.values():
        assert bool(mask.all())
    assert result.budget == 8


def test_jaccard_matches_hand_computation() -> None:
    left = torch.tensor([[True, True, False, False]])
    right = torch.tensor([[False, True, True, False]])
    assert torch.allclose(mask_jaccard(left, right), torch.tensor([1.0 / 3.0]))
    ref, delta = scores_pair()
    result = build_hybrid_label_masks(ref, delta, budget=4, ref_k=2, generator=sample_generator(0, "s"))
    # top-4 of ref = {0,1,2,3}, top-4 of delta = {4,5,6,7}: disjoint.
    assert torch.allclose(result.ref_delta_jaccard, torch.zeros(2))


if __name__ == "__main__":
    test_every_variant_meets_the_budget_exactly()
    test_reference_delta_takes_ref_topk_then_delta_topk_from_rest()
    test_ref_ratio_one_degenerates_to_reference_only()
    test_random_fill_is_deterministic_per_seed_and_sample()
    test_budget_larger_than_prompt_selects_everything()
    test_jaccard_matches_hand_computation()
    print("ok")
