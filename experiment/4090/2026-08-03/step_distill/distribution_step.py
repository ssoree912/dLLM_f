from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .distribution_forward import run_suffix_logits
from .distribution_pruning import DistributionPruningController
from .distribution_student import (
    StateConditionedSelector,
    commit_ranking_loss,
    distribution_kd_loss,
    distribution_kl,
)
from .oracle_generation import _build_transfer_index
from .trajectory_teacher import select_candidates


@dataclass(frozen=True, slots=True)
class DistributionRolloutConfig:
    gen_length: int = 128
    block_length: int = 32
    steps: int = 128
    distill_temperature: float = 1.0
    generation_temperature: float = 0.0
    teacher_commit_weight: float = 2.0
    commit_loss_weight: float = 0.1
    commit_margin: float = 0.1
    gate_temperature: float = 1.0
    max_grad_norm: float = 1.0
    mask_id: int = 126336

    def __post_init__(self) -> None:
        if min(self.gen_length, self.block_length, self.steps) <= 0:
            raise ValueError("generation lengths and steps must be positive")
        if self.gen_length % self.block_length:
            raise ValueError("generation length must be divisible by block length")
        block_count = self.gen_length // self.block_length
        if self.steps % block_count:
            raise ValueError("steps must be divisible by block count")
        if min(
            self.distill_temperature,
            self.gate_temperature,
            self.max_grad_norm,
        ) <= 0.0:
            raise ValueError("distillation, gate, and gradient values must be positive")
        if min(
            self.generation_temperature,
            self.teacher_commit_weight,
            self.commit_loss_weight,
            self.commit_margin,
        ) < 0.0:
            raise ValueError("sampling and loss weights must be non-negative")


@dataclass(frozen=True, slots=True)
class DistributionStepReport:
    step_id: int
    loss: float
    kd_loss: float
    diagnostic_kl: float
    commit_loss: float
    selector_grad_norm: float
    token_top1_agreement: float
    commit_jaccard: float


@torch.no_grad()
def pool_causal_state(
    embedding: nn.Embedding,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    initial_state_indices: torch.Tensor,
    *,
    mask_id: int,
) -> torch.Tensor:
    """Pool only state tokens known before the current pruned forward."""
    if prompt_ids.ndim != 2 or suffix_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt and suffix ids must have batch size one")
    committed = suffix_ids[0] != mask_id
    if bool(committed.any()):
        state_ids = suffix_ids[0][committed]
    else:
        if initial_state_indices.ndim != 1 or not initial_state_indices.numel():
            raise ValueError("initial state indices must be a non-empty vector")
        state_ids = prompt_ids[0].index_select(0, initial_state_indices)
    return embedding(state_ids).mean(dim=0).detach()


def run_distribution_step(
    model: nn.Module,
    selector: StateConditionedSelector,
    prompt_ids: torch.Tensor,
    initial_indices: torch.Tensor,
    suffix_ids: torch.Tensor,
    prompt_features: torch.Tensor,
    controller: DistributionPruningController,
    config: DistributionRolloutConfig,
    optimizer: torch.optim.Optimizer | None,
    embedding: nn.Embedding,
    start: int,
    end: int,
    transfer_count: torch.Tensor,
    step_id: int,
) -> tuple[DistributionStepReport, torch.Tensor, torch.Tensor]:
    uncommitted = suffix_ids == config.mask_id
    sequence = torch.cat((prompt_ids, suffix_ids), dim=1)
    controller.enable_full_attention()
    with torch.no_grad():
        full_logits = run_suffix_logits(model, sequence, prompt_length=prompt_ids.shape[1])
        _, full_confidence = select_candidates(
            full_logits,
            suffix_ids,
            uncommitted,
            temperature=config.generation_temperature,
        )
        full_confidence[:, end:] = -torch.inf
        teacher_commit = _build_transfer_index(full_confidence, transfer_count)
    state = pool_causal_state(
        embedding,
        prompt_ids,
        suffix_ids,
        initial_indices,
        mask_id=config.mask_id,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    device_type = prompt_ids.device.type
    with torch.set_grad_enabled(optimizer is not None), torch.autocast(
        device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        scores = selector(prompt_features, state)
        controller.set_scores(scores)
        pruned_logits = run_suffix_logits(
            model,
            sequence,
            prompt_length=prompt_ids.shape[1],
            controller=controller,
            layer_scores=scores,
            checkpoint_blocks=optimizer is not None,
        )
        kd = distribution_kd_loss(
            full_logits,
            pruned_logits,
            uncommitted,
            teacher_commit,
            temperature=config.distill_temperature,
            commit_weight=config.teacher_commit_weight,
        )
        diagnostic_kl = distribution_kl(
            full_logits,
            pruned_logits,
            uncommitted,
            temperature=config.distill_temperature,
        )
        confidence = torch.softmax(
            pruned_logits.float() / config.distill_temperature,
            dim=-1,
        ).amax(dim=-1)
        eligible = uncommitted.clone()
        eligible[:, :start] = False
        eligible[:, end:] = False
        commit = commit_ranking_loss(
            confidence,
            teacher_commit,
            eligible,
            margin=config.commit_margin,
        )
        loss = kd + config.commit_loss_weight * commit
    grad_norm = 0.0
    if optimizer is not None:
        loss.backward()
        norm = nn.utils.clip_grad_norm_(selector.parameters(), config.max_grad_norm)
        grad_norm = float(norm.detach().cpu())
        optimizer.step()
    with torch.no_grad():
        candidates, pruned_confidence = select_candidates(
            pruned_logits.detach(),
            suffix_ids,
            uncommitted,
            temperature=config.generation_temperature,
        )
        pruned_confidence[:, end:] = -torch.inf
        transfer = _build_transfer_index(pruned_confidence, transfer_count)
        agreement = (full_logits.argmax(-1) == pruned_logits.argmax(-1))[
            uncommitted
        ].float().mean()
        union = (teacher_commit | transfer).sum().clamp_min(1)
        jaccard = (teacher_commit & transfer).sum().float() / union
    return (
        DistributionStepReport(
            step_id,
            float(loss.detach().cpu()),
            float(kd.detach().cpu()),
            float(diagnostic_kl.detach().cpu()),
            float(commit.detach().cpu()),
            grad_norm,
            float(agreement.cpu()),
            float(jaccard.cpu()),
        ),
        transfer,
        candidates,
    )


__all__ = [
    "DistributionRolloutConfig",
    "DistributionStepReport",
    "pool_causal_state",
    "run_distribution_step",
]
