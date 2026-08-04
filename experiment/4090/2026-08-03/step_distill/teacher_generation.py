from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .model_hooks import (
    SuffixPromptAttentionCollector,
    find_transformer_blocks,
    install_suffix_prompt_attention_collector,
)
from .target_ranking import pad_commits, rank_targets


class TeacherForwardOutput(Protocol):
    logits: torch.Tensor
    hidden_states: Sequence[torch.Tensor]


@dataclass(frozen=True, slots=True)
class TeacherGenerationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PerStepTeacherConfig:
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    confidence_weight: bool
    max_target_k: int
    diversity_gamma: float
    mask_id: int = 126336

    def __post_init__(self) -> None:
        if self.gen_length <= 0 or self.block_length <= 0 or self.steps <= 0:
            raise TeacherGenerationError(
                "generation lengths and steps must be positive"
            )
        if self.gen_length % self.block_length != 0:
            raise TeacherGenerationError("gen_length must be divisible by block_length")
        block_count = self.gen_length // self.block_length
        if self.steps % block_count != 0:
            raise TeacherGenerationError(
                "steps must be divisible by the number of blocks"
            )
        if (
            self.temperature < 0.0
            or self.max_target_k <= 0
            or self.diversity_gamma < 0.0
        ):
            raise TeacherGenerationError(
                "temperature, max_target_k, and gamma must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class PerStepTeacherResult:
    generated_ids: torch.Tensor
    valid_step_mask: torch.Tensor
    commit_positions: torch.Tensor
    commit_counts: torch.Tensor
    commit_confidence: torch.Tensor
    context_pre: torch.Tensor
    step_scores: torch.Tensor
    top_order: torch.Tensor
    diverse_order: torch.Tensor
    candidate_scores: torch.Tensor


@torch.inference_mode()
def generate_per_step_teacher(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    question_indices: torch.Tensor,
    config: PerStepTeacherConfig,
) -> PerStepTeacherResult:
    """Generate a full-context trajectory while retaining every pre-commit target."""
    from .trajectory_teacher import (
        build_step_target,
        pool_pre_step_context,
        select_candidates,
    )

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise TeacherGenerationError(
            "per-step teacher expects prompt_ids with shape [1, prompt]"
        )
    blocks = find_transformer_blocks(model)
    layer_count = len(blocks)
    prompt_length = int(prompt_ids.shape[1])
    prefill = _forward_model(model, prompt_ids)
    similarity_hidden = _layer_inputs(prefill, layer_count)

    suffix_ids = torch.full(
        (1, config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    collector = install_suffix_prompt_attention_collector(
        model,
        prompt_length=prompt_length,
        suffix_length=config.gen_length,
    )
    score_steps: list[torch.Tensor] = []
    context_steps: list[torch.Tensor] = []
    position_steps: list[torch.Tensor] = []
    confidence_steps: list[torch.Tensor] = []
    steps_per_block = config.steps // (config.gen_length // config.block_length)

    try:
        for block_id in range(config.gen_length // config.block_length):
            start = block_id * config.block_length
            end = (block_id + 1) * config.block_length
            block_masks = suffix_ids[:, start:end] == config.mask_id
            transfer_counts = _planned_transfer_counts(block_masks, steps_per_block)
            for step_id in range(steps_per_block):
                committed = suffix_ids[0] != config.mask_id
                output, attention = _full_sequence_forward(
                    model,
                    prompt_ids,
                    suffix_ids,
                    collector,
                )
                layer_inputs = _layer_inputs(output, layer_count)
                context_steps.append(
                    pool_pre_step_context(
                        layer_inputs,
                        prompt_length=prompt_length,
                        committed_suffix=committed,
                        question_indices=question_indices.to(prompt_ids.device),
                    )
                )
                logits = output.logits[:, prompt_length:].float()
                candidates, confidence = select_candidates(
                    logits,
                    suffix_ids,
                    suffix_ids == config.mask_id,
                    temperature=config.temperature,
                )
                confidence[:, end:] = -torch.inf
                transfer = _build_transfer_index(
                    confidence, transfer_counts[:, step_id]
                )
                positions = transfer[0].nonzero(as_tuple=False).flatten()
                selected_confidence = confidence[0].index_select(0, positions)
                score_steps.append(
                    build_step_target(
                        attention,
                        positions,
                        selected_confidence,
                        confidence_weight=config.confidence_weight,
                    )
                )
                position_steps.append(positions)
                confidence_steps.append(selected_confidence)
                suffix_ids[transfer] = candidates[transfer]
    finally:
        collector.restore()

    step_scores = torch.stack(score_steps)
    top_order, diverse_order, candidate_scores = rank_targets(
        step_scores,
        similarity_hidden,
        max_k=config.max_target_k,
        gamma=config.diversity_gamma,
    )
    commit_positions, commit_confidence = pad_commits(position_steps, confidence_steps)
    commit_counts = torch.tensor(
        [step.numel() for step in position_steps], dtype=torch.long
    )
    return PerStepTeacherResult(
        generated_ids=suffix_ids.detach().cpu().squeeze(0),
        valid_step_mask=(commit_counts > 0),
        commit_positions=commit_positions,
        commit_counts=commit_counts,
        commit_confidence=commit_confidence,
        context_pre=torch.stack(context_steps).detach().cpu().float(),
        step_scores=step_scores.detach().cpu().float(),
        top_order=top_order,
        diverse_order=diverse_order,
        candidate_scores=candidate_scores,
    )


def _forward_model(model: nn.Module, input_ids: torch.Tensor) -> TeacherForwardOutput:
    return model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )


def _full_sequence_forward(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    collector: SuffixPromptAttentionCollector,
) -> tuple[TeacherForwardOutput, torch.Tensor]:
    collector.clear()
    output = _forward_model(model, torch.cat([prompt_ids, suffix_ids], dim=1))
    return output, collector.stacked_attention()


def _layer_inputs(output: TeacherForwardOutput, layer_count: int) -> torch.Tensor:
    if len(output.hidden_states) < layer_count:
        raise TeacherGenerationError("model returned too few hidden states")
    layers = [
        output.hidden_states[layer_id].squeeze(0) for layer_id in range(layer_count)
    ]
    return torch.stack(layers)


def _planned_transfer_counts(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_count = mask_index.sum(dim=1, keepdim=True)
    base = mask_count // steps
    remainder = mask_count % steps
    counts = base.expand(-1, steps).clone()
    indices = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    counts += indices < remainder
    return counts.to(torch.long)


def _build_transfer_index(
    confidence: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    transfer = torch.zeros_like(confidence, dtype=torch.bool)
    for batch_id in range(confidence.shape[0]):
        count = int(counts[batch_id])
        if count > 0:
            selected = torch.topk(confidence[batch_id], k=count).indices
            transfer[batch_id, selected] = True
    return transfer
