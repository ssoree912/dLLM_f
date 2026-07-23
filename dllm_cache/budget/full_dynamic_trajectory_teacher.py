from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.attention_teacher import (
    NamedModuleModel,
    find_transformer_blocks,
    project_attention_heads,
    repeat_kv_heads,
)
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


MASK_ID = 126336


class DecodeTokenizer(Protocol):
    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class FullDynamicTrajectoryConfig:
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    max_weight: float
    confidence_weight: bool
    mask_id: int = MASK_ID


@dataclass(frozen=True, slots=True)
class FullDynamicTrajectoryResult:
    generated_ids: torch.Tensor
    teacher_raw: torch.Tensor
    teacher_norm: torch.Tensor
    commit_count: int
    weight_sum: float


@dataclass(slots=True)
class SuffixPromptAttentionCollector:
    prompt_length: int
    suffix_length: int
    layer_count: int
    attention_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def clear(self) -> None:
        self.attention_by_layer.clear()

    def capture(
        self,
        block: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> None:
        if q.shape[0] != 1:
            raise RuntimeError("full dynamic trajectory teacher expects batch size 1")
        sequence_length = self.prompt_length + self.suffix_length
        if q.shape[1] != sequence_length:
            raise RuntimeError(
                "trajectory collector must run on prompt+suffix input, got "
                f"{q.shape[1]} tokens for expected {sequence_length}"
            )
        q_heads, k_heads = project_attention_heads(block, q, k)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        suffix_q = q_heads[:, :, self.prompt_length :, :]
        raw_scores = torch.matmul(
            suffix_q.float(),
            k_heads.float().transpose(-1, -2),
        ) / math.sqrt(float(suffix_q.shape[-1]))
        if attention_bias is not None:
            raw_scores = raw_scores + attention_bias[
                :, :, self.prompt_length : sequence_length, :sequence_length
            ].float()
        attention = torch.softmax(raw_scores, dim=-1)
        prompt_attention = attention[:, :, :, : self.prompt_length].mean(dim=1).squeeze(0)
        layer_id = int(getattr(block, "layer_id"))
        self.attention_by_layer[layer_id] = prompt_attention.detach()

    def assert_complete(self) -> None:
        missing = sorted(set(range(self.layer_count)) - set(self.attention_by_layer))
        if missing:
            raise RuntimeError(f"trajectory attention missing layers: {missing}")

    def restore(self) -> None:
        for module, original in self.originals:
            module.attention = original
        self.originals.clear()


@dataclass(slots=True)
class TrajectoryTrace:
    sum_attention: torch.Tensor
    max_attention: torch.Tensor
    confidence_weight: bool
    commit_count: int = 0
    weight_sum: float = 0.0

    def aggregate(self, max_weight: float) -> torch.Tensor:
        if self.weight_sum <= 0.0:
            return self.sum_attention
        mean_attention = self.sum_attention / self.weight_sum
        if max_weight <= 0.0:
            return mean_attention
        if max_weight >= 1.0:
            return self.max_attention
        return max_weight * self.max_attention + (1.0 - max_weight) * mean_attention


def install_suffix_prompt_attention_collector(
    model: NamedModuleModel,
    prompt_length: int,
    suffix_length: int,
) -> SuffixPromptAttentionCollector:
    blocks = find_transformer_blocks(model)
    collector = SuffixPromptAttentionCollector(
        prompt_length=prompt_length,
        suffix_length=suffix_length,
        layer_count=len(blocks),
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        block.attention = types.MethodType(
            make_wrapped_attention(collector, original, accepts_block_mask),
            block,
        )
        collector.originals.append((block, original))
    return collector


def make_wrapped_attention(
    collector: SuffixPromptAttentionCollector,
    original: types.MethodType,
    accepts_block_mask: bool,
):
    def wrapped_attention(
        block_self: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        block_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        collector.capture(block_self, q, k, attention_bias)
        if accepts_block_mask:
            return original(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                block_mask=block_mask,
            )
        return original(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

    return wrapped_attention


@torch.inference_mode()
def generate_with_full_dynamic_trajectory_teacher(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    config: FullDynamicTrajectoryConfig,
) -> FullDynamicTrajectoryResult:
    if config.gen_length % config.block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = config.gen_length // config.block_length
    if config.steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    if not 0.0 <= config.max_weight <= 1.0:
        raise RuntimeError("max_weight must be in [0, 1]")
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    suffix_ids = torch.full(
        (prompt_ids.shape[0], config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    trace = TrajectoryTrace(
        sum_attention=torch.zeros((len(blocks), prompt_length), device=prompt_ids.device),
        max_attention=torch.zeros((len(blocks), prompt_length), device=prompt_ids.device),
        confidence_weight=config.confidence_weight,
    )
    collector = install_suffix_prompt_attention_collector(
        model,
        prompt_length=prompt_length,
        suffix_length=config.gen_length,
    )
    steps_per_block = config.steps // num_blocks
    try:
        for block_id in range(num_blocks):
            start = block_id * config.block_length
            end = (block_id + 1) * config.block_length
            block_mask_index = suffix_ids[:, start:end] == config.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for step_id in range(steps_per_block):
                mask_index = suffix_ids == config.mask_id
                logits, prompt_attention = full_sequence_logits_and_prompt_attention(
                    model,
                    prompt_ids,
                    suffix_ids,
                    collector,
                )
                x0, confidence = select_candidates(logits, suffix_ids, mask_index, config)
                confidence[:, end:] = -torch.inf
                transfer_index = build_transfer_index(confidence, num_transfer_tokens[:, step_id])
                accumulate_commits(trace, prompt_attention, transfer_index, confidence)
                suffix_ids[transfer_index] = x0[transfer_index]
    finally:
        collector.restore()
    teacher_raw = trace.aggregate(config.max_weight).detach().cpu().float()
    teacher_norm = teacher_raw / teacher_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return FullDynamicTrajectoryResult(
        generated_ids=suffix_ids.detach().cpu().squeeze(0),
        teacher_raw=teacher_raw,
        teacher_norm=teacher_norm,
        commit_count=trace.commit_count,
        weight_sum=trace.weight_sum,
    )


def full_sequence_logits_and_prompt_attention(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    collector: SuffixPromptAttentionCollector,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
    collector.clear()
    out = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    collector.assert_complete()
    logits = out.logits[:, collector.prompt_length :].float()
    return logits, dict(collector.attention_by_layer)


def select_candidates(
    logits: torch.Tensor,
    suffix_ids: torch.Tensor,
    mask_index: torch.Tensor,
    config: FullDynamicTrajectoryConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    probs = F.softmax(logits, dim=-1)
    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, suffix_ids)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -torch.inf))
    return x0, confidence


def build_transfer_index(
    confidence: torch.Tensor,
    transfer_counts: torch.Tensor,
) -> torch.Tensor:
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool, device=confidence.device)
    for batch_id in range(confidence.shape[0]):
        count = int(transfer_counts[batch_id].item())
        if count <= 0:
            continue
        select_index = torch.topk(confidence[batch_id], k=count).indices
        transfer_index[batch_id, select_index] = True
    return transfer_index


def accumulate_commits(
    trace: TrajectoryTrace,
    prompt_attention: dict[int, torch.Tensor],
    transfer_index: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    if transfer_index.shape[0] != 1:
        raise RuntimeError("full dynamic trajectory teacher expects batch size 1")
    selected = transfer_index[0].nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return
    weights = commit_weights(trace, confidence[0].index_select(0, selected))
    for layer_id, layer_attention in prompt_attention.items():
        rows = layer_attention.index_select(0, selected).float()
        weighted_rows = rows * weights.unsqueeze(-1)
        trace.sum_attention[layer_id] += weighted_rows.sum(dim=0)
        trace.max_attention[layer_id] = torch.maximum(
            trace.max_attention[layer_id],
            weighted_rows.max(dim=0).values,
        )
    trace.commit_count += int(selected.numel())
    trace.weight_sum += float(weights.sum().detach().cpu())


def commit_weights(trace: TrajectoryTrace, confidence: torch.Tensor) -> torch.Tensor:
    if trace.confidence_weight:
        return confidence.clamp_min(0.0).float()
    return torch.ones_like(confidence, dtype=torch.float32)


def decode_generated_answer(
    tokenizer: DecodeTokenizer,
    generated_ids: torch.Tensor,
) -> str:
    return tokenizer.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

