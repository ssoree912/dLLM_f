from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dllm_cache.budget.attention_teacher import find_transformer_blocks
from dllm_cache.budget.full_dynamic_trajectory_teacher import (
    FullDynamicTrajectoryConfig,
    build_transfer_index,
    full_sequence_logits_and_prompt_attention,
    install_suffix_prompt_attention_collector,
    select_candidates,
)
from utils.generate_function import get_num_transfer_tokens

MASK_ID = 126336


@dataclass(frozen=True, slots=True)
class FuturePoolTeacherConfig:
    gen_length: int
    block_length: int
    steps: int
    active_top_k: int
    temperature: float
    confidence_weight: bool
    target_aggregation: str
    mask_id: int = MASK_ID


@dataclass(frozen=True, slots=True)
class FuturePoolTeacherResult:
    generated_ids: torch.Tensor
    teacher_raw: torch.Tensor
    teacher_norm: torch.Tensor
    union_mask: torch.Tensor
    union_size_by_layer: torch.Tensor
    commit_count: int
    weight_sum: float


@dataclass(slots=True)
class FuturePoolTrace:
    sum_scores: torch.Tensor
    max_scores: torch.Tensor
    union_mask: torch.Tensor
    confidence_weight: bool
    active_top_k: int
    commit_count: int = 0
    weight_sum: float = 0.0

    def aggregate(self, target_aggregation: str) -> torch.Tensor:
        match target_aggregation:
            case "max":
                raw = self.max_scores
            case "sum":
                raw = self.sum_scores
            case _:
                raise RuntimeError(f"unsupported target aggregation: {target_aggregation}")
        return raw * self.union_mask.float()


@torch.inference_mode()
def generate_with_future_pool_teacher(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    config: FuturePoolTeacherConfig,
) -> FuturePoolTeacherResult:
    validate_generation_shape(config)
    prompt_length = int(prompt_ids.shape[1])
    suffix_ids = torch.full(
        (prompt_ids.shape[0], config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    layer_count = len(find_transformer_blocks(model))
    trace = FuturePoolTrace(
        sum_scores=torch.zeros((layer_count, prompt_length), device=prompt_ids.device),
        max_scores=torch.zeros((layer_count, prompt_length), device=prompt_ids.device),
        union_mask=torch.zeros((layer_count, prompt_length), dtype=torch.bool, device=prompt_ids.device),
        confidence_weight=config.confidence_weight,
        active_top_k=config.active_top_k,
    )
    collector = install_suffix_prompt_attention_collector(model, prompt_length, config.gen_length)
    candidate_config = FullDynamicTrajectoryConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        max_weight=0.0,
        confidence_weight=config.confidence_weight,
        mask_id=config.mask_id,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    try:
        for block_id in range(num_blocks):
            end = (block_id + 1) * config.block_length
            block_mask = suffix_ids[:, block_id * config.block_length : end] == config.mask_id
            transfer_counts = get_num_transfer_tokens(block_mask, steps_per_block)
            for step_id in range(steps_per_block):
                mask_index = suffix_ids == config.mask_id
                logits, prompt_attention = full_sequence_logits_and_prompt_attention(
                    model,
                    prompt_ids,
                    suffix_ids,
                    collector,
                )
                x0, confidence = select_candidates(logits, suffix_ids, mask_index, candidate_config)
                confidence[:, end:] = -torch.inf
                transfer_index = build_transfer_index(confidence, transfer_counts[:, step_id])
                accumulate_future_pool(trace, prompt_attention, transfer_index, confidence)
                suffix_ids[transfer_index] = x0[transfer_index]
    finally:
        collector.restore()
    teacher_raw = trace.aggregate(config.target_aggregation).detach().cpu().float()
    teacher_norm = normalize_scores(teacher_raw)
    return FuturePoolTeacherResult(
        generated_ids=suffix_ids.detach().cpu().squeeze(0),
        teacher_raw=teacher_raw,
        teacher_norm=teacher_norm,
        union_mask=trace.union_mask.detach().cpu(),
        union_size_by_layer=trace.union_mask.sum(dim=-1).detach().cpu(),
        commit_count=trace.commit_count,
        weight_sum=trace.weight_sum,
    )


def validate_generation_shape(config: FuturePoolTeacherConfig) -> None:
    if config.gen_length % config.block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = config.gen_length // config.block_length
    if config.steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    if config.active_top_k <= 0:
        raise RuntimeError("active_top_k must be positive")


def accumulate_future_pool(
    trace: FuturePoolTrace,
    prompt_attention: dict[int, torch.Tensor],
    transfer_index: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    if transfer_index.shape[0] != 1:
        raise RuntimeError("future pool teacher expects batch size 1")
    selected = transfer_index[0].nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return
    weights = confidence[0].index_select(0, selected).clamp_min(0.0).float()
    if not trace.confidence_weight:
        weights = torch.ones_like(weights, dtype=torch.float32)
    for layer_id, layer_attention in prompt_attention.items():
        scores = (layer_attention.index_select(0, selected).float() * weights.unsqueeze(-1)).sum(dim=0)
        top_count = min(trace.active_top_k, int(scores.numel()))
        top_indices = torch.topk(scores, k=top_count, largest=True).indices
        trace.union_mask[layer_id].scatter_(dim=0, index=top_indices, value=True)
        trace.sum_scores[layer_id] += scores
        trace.max_scores[layer_id] = torch.maximum(trace.max_scores[layer_id], scores)
    trace.commit_count += int(selected.numel())
    trace.weight_sum += float(weights.sum().detach().cpu())


def normalize_scores(raw: torch.Tensor) -> torch.Tensor:
    totals = raw.sum(dim=-1, keepdim=True)
    return raw / totals.clamp_min(1e-6)
