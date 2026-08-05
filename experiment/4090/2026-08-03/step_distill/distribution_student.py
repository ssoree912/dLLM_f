from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    layer_count: int
    hidden_dim: int
    projection_dim: int = 256
    mlp_dim: int = 512

    def __post_init__(self) -> None:
        if min(
            self.layer_count,
            self.hidden_dim,
            self.projection_dim,
            self.mlp_dim,
        ) <= 0:
            raise ValueError("selector dimensions must be positive")


class StateConditionedSelector(nn.Module):
    """Shared prompt selector conditioned on the causal denoising state."""

    def __init__(self, config: SelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.token_projection = nn.Linear(
            config.hidden_dim,
            config.projection_dim,
        )
        self.context_projection = nn.Linear(
            config.hidden_dim,
            config.projection_dim,
        )
        self.layer_embedding = nn.Embedding(
            config.layer_count,
            config.projection_dim,
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.projection_dim * 4, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, 1),
        )

    def forward(
        self,
        prompt_features: torch.Tensor,
        state_context: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_features.ndim != 3:
            raise ValueError("prompt features must have shape [layer, prompt, hidden]")
        if prompt_features.shape[0] != self.config.layer_count:
            raise ValueError("prompt feature layers do not match selector layers")
        if prompt_features.shape[2] != self.config.hidden_dim:
            raise ValueError("prompt feature width does not match selector hidden dim")
        if state_context.shape != (self.config.hidden_dim,):
            raise ValueError("state context must have shape [hidden]")

        token = self.token_projection(prompt_features)
        context = self.context_projection(state_context).to(token.dtype)
        context = context.view(1, 1, -1).expand_as(token)
        layer_ids = torch.arange(
            self.config.layer_count,
            device=prompt_features.device,
        )
        layer = self.layer_embedding(layer_ids).to(token.dtype)
        layer = layer.unsqueeze(1).expand_as(token)
        fused = torch.cat((token, context, token * context, layer), dim=-1)
        return self.score_head(fused).squeeze(-1)


def distribution_kd_loss(
    full_logits: torch.Tensor,
    pruned_logits: torch.Tensor,
    uncommitted_mask: torch.Tensor,
    teacher_commit_mask: torch.Tensor,
    *,
    temperature: float,
    commit_weight: float,
) -> torch.Tensor:
    """Forward KL on the currently uncommitted suffix positions."""
    if full_logits.shape != pruned_logits.shape or full_logits.ndim != 3:
        raise ValueError("full and pruned logits must share [batch, suffix, vocab]")
    expected_mask_shape = full_logits.shape[:-1]
    if (
        uncommitted_mask.shape != expected_mask_shape
        or teacher_commit_mask.shape != expected_mask_shape
    ):
        raise ValueError("distillation masks must match [batch, suffix]")
    if temperature <= 0.0 or commit_weight < 0.0:
        raise ValueError("temperature must be positive and commit weight non-negative")
    count = uncommitted_mask.sum()
    if int(count) == 0:
        raise ValueError("distillation requires at least one uncommitted position")

    teacher = functional.softmax(full_logits.float() / temperature, dim=-1)
    student = functional.log_softmax(pruned_logits.float() / temperature, dim=-1)
    per_position = functional.kl_div(
        student,
        teacher,
        reduction="none",
    ).sum(dim=-1)
    weights = 1.0 + commit_weight * teacher_commit_mask.float()
    return (
        per_position * weights * uncommitted_mask.float()
    ).sum() * (temperature**2) / count


def distribution_kl(
    full_logits: torch.Tensor,
    pruned_logits: torch.Tensor,
    uncommitted_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Unweighted diagnostic KL averaged over currently unknown positions."""
    if full_logits.shape != pruned_logits.shape or full_logits.ndim != 3:
        raise ValueError("full and pruned logits must share [batch, suffix, vocab]")
    if uncommitted_mask.shape != full_logits.shape[:-1]:
        raise ValueError("uncommitted mask must match [batch, suffix]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    count = uncommitted_mask.sum()
    if int(count) == 0:
        raise ValueError("diagnostic KL requires an uncommitted position")
    teacher = functional.softmax(full_logits.float() / temperature, dim=-1)
    student = functional.log_softmax(pruned_logits.float() / temperature, dim=-1)
    per_position = functional.kl_div(student, teacher, reduction="none").sum(-1)
    return (per_position * uncommitted_mask.float()).sum() / count


def commit_ranking_loss(
    pruned_confidence: torch.Tensor,
    teacher_commit_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Rank teacher commit positions above other positions in the active block."""
    if (
        pruned_confidence.ndim != 2
        or teacher_commit_mask.shape != pruned_confidence.shape
        or eligible_mask.shape != pruned_confidence.shape
    ):
        raise ValueError("commit tensors must share [batch, suffix]")
    if margin < 0.0:
        raise ValueError("commit margin must be non-negative")
    losses: list[torch.Tensor] = []
    for batch_id in range(pruned_confidence.shape[0]):
        positives = pruned_confidence[batch_id][teacher_commit_mask[batch_id]]
        negatives = pruned_confidence[batch_id][
            eligible_mask[batch_id] & ~teacher_commit_mask[batch_id]
        ]
        if positives.numel() and negatives.numel():
            pairwise = positives.unsqueeze(1) - negatives.unsqueeze(0)
            losses.append(functional.relu(margin - pairwise).mean())
    if not losses:
        return pruned_confidence.sum() * 0.0
    return torch.stack(losses).mean()


__all__ = [
    "SelectorConfig",
    "StateConditionedSelector",
    "commit_ranking_loss",
    "distribution_kd_loss",
    "distribution_kl",
]
