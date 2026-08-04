from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class SelectionError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def stable_top_order(scores: torch.Tensor, max_k: int) -> torch.Tensor:
    """Return a deterministic descending order for one score vector."""
    if scores.ndim != 1:
        raise SelectionError("scores must be one-dimensional")
    if max_k <= 0:
        raise SelectionError("max_k must be positive")
    if not torch.isfinite(scores).all():
        raise SelectionError("scores must be finite")
    k = min(max_k, int(scores.numel()))
    return torch.argsort(scores, descending=True, stable=True)[:k]


def greedy_mmr_order(
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    max_k: int,
    gamma: float,
) -> torch.Tensor:
    """Select a max-relevance/min-redundancy order without a P-by-P matrix."""
    if scores.ndim != 1:
        raise SelectionError("scores must be one-dimensional")
    if embeddings.ndim != 2 or embeddings.shape[0] != scores.shape[0]:
        raise SelectionError("embeddings must have shape [prompt, feature]")
    if max_k <= 0:
        raise SelectionError("max_k must be positive")
    if gamma < 0.0:
        raise SelectionError("gamma must be non-negative")
    if not torch.isfinite(scores).all() or not torch.isfinite(embeddings).all():
        raise SelectionError("scores and embeddings must be finite")
    if gamma == 0.0:
        return stable_top_order(scores, max_k)

    prompt_length = int(scores.numel())
    k = min(max_k, prompt_length)
    relevance = _min_max_normalize(scores.float())
    normalized_embeddings = functional.normalize(embeddings.float(), dim=-1)
    redundancy = torch.zeros_like(relevance)
    available = torch.ones(prompt_length, dtype=torch.bool, device=scores.device)
    selected: list[torch.Tensor] = []

    for _ in range(k):
        utility = relevance - gamma * redundancy
        utility = utility.masked_fill(~available, -torch.inf)
        index = torch.argmax(utility)
        selected.append(index)
        available[index] = False
        similarity = normalized_embeddings @ normalized_embeddings[index]
        redundancy = torch.maximum(redundancy, similarity.clamp_min(0.0))

    return torch.stack(selected).to(dtype=torch.long)


def _min_max_normalize(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.min()
    width = scores.max() - minimum
    if float(width) <= 0.0:
        return torch.zeros_like(scores)
    return (scores - minimum) / width
