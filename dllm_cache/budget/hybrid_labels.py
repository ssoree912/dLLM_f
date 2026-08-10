"""Turn offline hybrid teacher scores into fixed-budget selection masks.

Extraction shards carry two continuous ``[layer, prompt]`` scores: the
reference signal (how much the committed suffix read each prompt token) and
the delta signal (how far each prompt token's K/V travelled across the
trajectory).  This module assembles the actual selection targets from them:
for a budget ``B``, reference fills the first ``K_R`` slots and the remaining
``B - K_R`` go to the highest-delta tokens outside that set.  The comparison
baselines (reference-only, delta-only, reference+random) come from the same
scores so all four variants share one extraction run.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch


MASK_VARIANTS = (
    "reference_only",
    "delta_only",
    "reference_random",
    "reference_delta",
)


@dataclass(frozen=True, slots=True)
class HybridLabelResult:
    masks: dict[str, torch.Tensor]
    ref_delta_jaccard: torch.Tensor
    budget: int
    ref_k: int


def build_hybrid_label_masks(
    ref_scores: torch.Tensor,
    delta_scores: torch.Tensor,
    budget: int,
    ref_k: int,
    generator: torch.Generator,
) -> HybridLabelResult:
    validate_score_pair(ref_scores, delta_scores)
    prompt_length = int(ref_scores.shape[-1])
    effective_budget = min(budget, prompt_length)
    effective_ref_k = min(ref_k, effective_budget)
    if budget <= 0:
        raise RuntimeError("budget must be positive")
    if ref_k < 0:
        raise RuntimeError("ref_k must be non-negative")

    ref_base = topk_mask(ref_scores, effective_ref_k)
    fill_count = effective_budget - effective_ref_k
    masks = {
        "reference_only": topk_mask(ref_scores, effective_budget),
        "delta_only": topk_mask(delta_scores, effective_budget),
        "reference_random": random_fill(ref_base, fill_count, generator),
        "reference_delta": score_fill(ref_base, delta_scores, fill_count),
    }
    for name, mask in masks.items():
        sizes = mask.sum(dim=-1)
        if not bool((sizes == effective_budget).all()):
            raise RuntimeError(
                f"mask {name} does not meet budget {effective_budget}: sizes={sizes.tolist()}"
            )
    return HybridLabelResult(
        masks=masks,
        ref_delta_jaccard=mask_jaccard(masks["reference_only"], masks["delta_only"]),
        budget=effective_budget,
        ref_k=effective_ref_k,
    )


def validate_score_pair(ref_scores: torch.Tensor, delta_scores: torch.Tensor) -> None:
    if ref_scores.ndim != 2 or delta_scores.ndim != 2:
        raise RuntimeError("scores must have shape [layer, prompt]")
    if ref_scores.shape != delta_scores.shape:
        raise RuntimeError(
            "reference and delta score shapes differ: "
            f"{tuple(ref_scores.shape)} vs {tuple(delta_scores.shape)}"
        )


def topk_mask(scores: torch.Tensor, count: int) -> torch.Tensor:
    mask = torch.zeros(scores.shape, dtype=torch.bool)
    if count <= 0:
        return mask
    take = min(count, int(scores.shape[-1]))
    indices = torch.topk(scores.float(), k=take, dim=-1, largest=True).indices
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def score_fill(base_mask: torch.Tensor, scores: torch.Tensor, fill_count: int) -> torch.Tensor:
    if fill_count <= 0:
        return base_mask.clone()
    blocked = scores.float().masked_fill(base_mask, float("-inf"))
    return base_mask | topk_mask(blocked, fill_count)


def random_fill(
    base_mask: torch.Tensor,
    fill_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = base_mask.clone()
    if fill_count <= 0:
        return mask
    for layer_id in range(mask.shape[0]):
        rest = (~mask[layer_id]).nonzero(as_tuple=False).flatten()
        if fill_count > rest.numel():
            raise RuntimeError(
                f"layer {layer_id}: cannot fill {fill_count} tokens from {rest.numel()} remaining"
            )
        order = torch.randperm(rest.numel(), generator=generator)
        mask[layer_id, rest[order[:fill_count]]] = True
    return mask


def mask_jaccard(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection = (left & right).sum(dim=-1).float()
    union = (left | right).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def sample_generator(seed: int, sample_id: str) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed((seed ^ zlib.crc32(sample_id.encode("utf-8"))) & 0x7FFF_FFFF)
    return generator


__all__ = [
    "MASK_VARIANTS",
    "HybridLabelResult",
    "build_hybrid_label_masks",
    "mask_jaccard",
    "random_fill",
    "sample_generator",
    "score_fill",
    "topk_mask",
]
