from __future__ import annotations

import torch

from .selection import greedy_mmr_order, stable_top_order


def rank_targets(
    scores: torch.Tensor,
    similarity_hidden: torch.Tensor,
    *,
    max_k: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build plain and diverse per-step orders from dense in-memory scores."""
    top_steps: list[torch.Tensor] = []
    diverse_steps: list[torch.Tensor] = []
    for step_id in range(scores.shape[0]):
        top_layers: list[torch.Tensor] = []
        diverse_layers: list[torch.Tensor] = []
        for layer_id in range(scores.shape[1]):
            layer_scores = scores[step_id, layer_id]
            top_layers.append(stable_top_order(layer_scores, max_k))
            diverse_layers.append(
                greedy_mmr_order(
                    layer_scores,
                    similarity_hidden[layer_id],
                    max_k,
                    gamma,
                )
            )
        top_steps.append(torch.stack(top_layers))
        diverse_steps.append(torch.stack(diverse_layers))
    top_order = torch.stack(top_steps).cpu()
    diverse_order = torch.stack(diverse_steps).cpu()
    candidate_scores = scores.detach().cpu().gather(-1, top_order)
    return top_order, diverse_order, candidate_scores


def pad_commits(
    positions: list[torch.Tensor],
    confidences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-size simultaneous commits with the schema's sentinel values."""
    max_count = max(1, max(int(step.numel()) for step in positions))
    padded_positions = torch.full((len(positions), max_count), -1, dtype=torch.long)
    padded_confidence = torch.zeros((len(positions), max_count), dtype=torch.float32)
    for step_id, (step_positions, step_confidence) in enumerate(
        zip(positions, confidences, strict=True)
    ):
        count = int(step_positions.numel())
        padded_positions[step_id, :count] = step_positions.detach().cpu()
        padded_confidence[step_id, :count] = step_confidence.detach().cpu().float()
    return padded_positions, padded_confidence
