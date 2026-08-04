from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class TrajectoryError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_step_target(
    prompt_attention: torch.Tensor,
    commit_positions: torch.Tensor,
    commit_confidence: torch.Tensor,
    *,
    confidence_weight: bool,
) -> torch.Tensor:
    """Aggregate selected pre-commit rows into one target per layer."""
    if prompt_attention.ndim != 3:
        raise TrajectoryError(
            "prompt_attention must have shape [layer, suffix, prompt]"
        )
    if commit_positions.ndim != 1 or commit_confidence.ndim != 1:
        raise TrajectoryError("commit positions and confidence must be one-dimensional")
    if commit_positions.numel() != commit_confidence.numel():
        raise TrajectoryError("commit positions and confidence must have equal length")
    if commit_positions.numel() == 0:
        return torch.zeros(
            (prompt_attention.shape[0], prompt_attention.shape[2]),
            dtype=torch.float32,
            device=prompt_attention.device,
        )

    selected = prompt_attention.index_select(1, commit_positions).float()
    weights = _within_step_weights(
        commit_confidence, confidence_weight=confidence_weight
    )
    return (selected * weights.view(1, -1, 1)).sum(dim=1)


def pool_pre_step_context(
    layer_inputs: torch.Tensor,
    *,
    prompt_length: int,
    committed_suffix: torch.Tensor,
    question_indices: torch.Tensor,
) -> torch.Tensor:
    """Pool committed suffix states, falling back to the query span at step zero."""
    if layer_inputs.ndim != 3:
        raise TrajectoryError("layer_inputs must have shape [layer, sequence, hidden]")
    if committed_suffix.ndim != 1 or committed_suffix.dtype != torch.bool:
        raise TrajectoryError("committed_suffix must be a one-dimensional bool tensor")
    if question_indices.ndim != 1 or question_indices.numel() == 0:
        raise TrajectoryError("question_indices must be a non-empty vector")
    suffix_length = layer_inputs.shape[1] - prompt_length
    if suffix_length != committed_suffix.numel():
        raise TrajectoryError("committed suffix width does not match layer inputs")

    committed_positions = committed_suffix.nonzero(as_tuple=False).flatten()
    if committed_positions.numel() > 0:
        sequence_positions = committed_positions + prompt_length
        return layer_inputs.index_select(1, sequence_positions).float().mean(dim=1)
    return layer_inputs.index_select(1, question_indices).float().mean(dim=1)


def select_candidates(
    logits: torch.Tensor,
    suffix_ids: torch.Tensor,
    mask_index: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select candidate tokens with a numerically safe zero-temperature path."""
    if temperature < 0.0:
        raise TrajectoryError("temperature must be non-negative")
    if logits.shape[:-1] != suffix_ids.shape or suffix_ids.shape != mask_index.shape:
        raise TrajectoryError("logits, suffix_ids, and mask_index shapes do not align")

    candidate_logits = logits.float()
    if temperature > 0.0:
        uniform = torch.rand_like(candidate_logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        candidate_logits = candidate_logits + temperature * gumbel
    token_ids = torch.argmax(candidate_logits, dim=-1)
    probabilities = functional.softmax(logits.float(), dim=-1)
    confidence = probabilities.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    token_ids = torch.where(mask_index, token_ids, suffix_ids)
    confidence = torch.where(
        mask_index, confidence, torch.full_like(confidence, -torch.inf)
    )
    return token_ids, confidence


def _within_step_weights(
    confidence: torch.Tensor, *, confidence_weight: bool
) -> torch.Tensor:
    if not confidence_weight:
        return torch.full_like(
            confidence, 1.0 / float(confidence.numel()), dtype=torch.float32
        )
    positive = confidence.float().clamp_min(0.0)
    total = positive.sum()
    if float(total) <= 0.0:
        return torch.full_like(positive, 1.0 / float(positive.numel()))
    return positive / total


__all__ = [
    "build_step_target",
    "pool_pre_step_context",
    "select_candidates",
]
