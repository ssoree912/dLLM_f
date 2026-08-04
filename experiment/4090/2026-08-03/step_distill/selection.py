from __future__ import annotations

from dataclasses import dataclass

import torch


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
