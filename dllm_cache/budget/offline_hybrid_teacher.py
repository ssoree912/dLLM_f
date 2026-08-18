"""Offline teacher that extracts reference attention and prompt K/V drift together.

The generation trajectory is identical to :mod:`future_pool_teacher`, but a
single attention wrapper observes both signals during each full-sequence
forward.  Only running ``[layer, prompt]`` accumulators and the immediately
previous prompt K/V are retained; no per-step tensors are returned or saved.
"""

from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.full_dynamic_trajectory_teacher import (
    FullDynamicTrajectoryConfig,
    build_transfer_index,
    select_candidates,
)
from dllm_cache.budget.future_pool_teacher import normalize_scores
from dllm_cache.budget.prompt_kv_cache import project_heads, repeat_heads
from utils.generate_function import get_num_transfer_tokens


MASK_ID = 126336


@dataclass(frozen=True, slots=True)
class OfflineHybridTeacherConfig:
    """Attention+Delta teacher defaults used by the prune-cache scorer.

    The primary path is dense, confidence weighted, max aggregated, and does
    not apply chat formatting.  Legacy callers may still override these fields
    while old experiments are being migrated.
    """

    gen_length: int = 128
    block_length: int = 8
    steps: int = 128
    active_top_k: int = 0
    temperature: float = 0.0
    confidence_weight: bool = True
    target_aggregation: str = "max"
    mask_id: int = MASK_ID


@dataclass(frozen=True, slots=True)
class OfflineHybridTeacherResult:
    generated_ids: torch.Tensor
    teacher_raw: torch.Tensor
    teacher_norm: torch.Tensor
    ref_union_mask: torch.Tensor
    ref_union_size_by_layer: torch.Tensor
    delta_raw: torch.Tensor
    delta_norm: torch.Tensor
    delta_step_max: torch.Tensor
    delta_observation_count: int
    reference_step_count: int
    commit_count: int
    weight_sum: float


@dataclass(slots=True)
class ReferenceTrace:
    """Running reference score and optional temporal-union state.

    ``active_top_k == 0`` means that no teacher-side support budget is applied:
    every prompt position remains in the support and the continuous aggregate is
    normalized directly.
    """

    sum_scores: torch.Tensor
    max_scores: torch.Tensor
    union_mask: torch.Tensor
    confidence_weight: bool
    active_top_k: int
    step_count: int = 0
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


@dataclass(slots=True)
class OfflineHybridCollector:
    """Observe suffix-to-prompt attention and stepwise prompt K/V movement."""

    prompt_length: int
    suffix_length: int
    layer_count: int
    attention_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    previous_key: dict[int, torch.Tensor] = field(default_factory=dict)
    previous_value: dict[int, torch.Tensor] = field(default_factory=dict)
    delta_sum: dict[int, torch.Tensor] = field(default_factory=dict)
    delta_max: dict[int, torch.Tensor] = field(default_factory=dict)
    observations: dict[int, int] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def clear_attention(self) -> None:
        self.attention_by_layer.clear()

    @torch.no_grad()
    def capture(
        self,
        block: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> None:
        if q.shape[0] != 1:
            raise RuntimeError("offline hybrid teacher expects batch size 1")
        sequence_length = self.prompt_length + self.suffix_length
        if q.shape[1] != sequence_length or k.shape[1] != sequence_length or v.shape[1] != sequence_length:
            raise RuntimeError(
                "offline hybrid collector must run on prompt+suffix input, got "
                f"q/k/v lengths {(q.shape[1], k.shape[1], v.shape[1])} for expected "
                f"{sequence_length}"
            )

        layer_id = int(getattr(block, "layer_id"))
        q_heads, k_heads, v_heads = project_heads(block, q, k, v, position_offset=0)
        self._capture_reference(layer_id, q_heads, k_heads, attention_bias)
        self._capture_delta(layer_id, k_heads, v_heads)

    def _capture_reference(
        self,
        layer_id: int,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> None:
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_heads(k_heads, q_heads.shape[1])
        sequence_length = self.prompt_length + self.suffix_length
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
        self.attention_by_layer[layer_id] = (
            attention[:, :, :, : self.prompt_length].mean(dim=1).squeeze(0).detach()
        )

    def _capture_delta(
        self,
        layer_id: int,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
    ) -> None:
        prompt_key = k_heads[:, :, : self.prompt_length, :].detach()
        prompt_value = v_heads[:, :, : self.prompt_length, :].detach()
        if layer_id not in self.previous_key:
            zeros = torch.zeros(self.prompt_length, dtype=torch.float32, device=prompt_key.device)
            self.previous_key[layer_id] = prompt_key.clone()
            self.previous_value[layer_id] = prompt_value.clone()
            self.delta_sum[layer_id] = zeros.clone()
            self.delta_max[layer_id] = zeros.clone()
            self.observations[layer_id] = 0
            return

        key_step = normalized_prompt_movement(prompt_key, self.previous_key[layer_id])
        value_step = normalized_prompt_movement(prompt_value, self.previous_value[layer_id])
        step_delta = (key_step + value_step) * 0.5
        self.delta_sum[layer_id] += step_delta
        self.delta_max[layer_id] = torch.maximum(self.delta_max[layer_id], step_delta)
        self.observations[layer_id] += 1
        self.previous_key[layer_id] = prompt_key.clone()
        self.previous_value[layer_id] = prompt_value.clone()

    def assert_forward_complete(self) -> None:
        expected = set(range(self.layer_count))
        missing_attention = sorted(expected - set(self.attention_by_layer))
        missing_delta = sorted(expected - set(self.delta_sum))
        if missing_attention:
            raise RuntimeError(f"offline hybrid attention missing layers: {missing_attention}")
        if missing_delta:
            raise RuntimeError(f"offline hybrid delta missing layers: {missing_delta}")

    def delta_result(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        missing = sorted(set(range(self.layer_count)) - set(self.delta_sum))
        if missing:
            raise RuntimeError(f"offline hybrid delta missing layers: {missing}")
        counts = {self.observations[layer_id] for layer_id in range(self.layer_count)}
        if len(counts) != 1:
            raise RuntimeError(f"offline hybrid delta observation counts differ by layer: {sorted(counts)}")
        delta_sum = torch.stack([self.delta_sum[layer_id] for layer_id in range(self.layer_count)])
        delta_max = torch.stack([self.delta_max[layer_id] for layer_id in range(self.layer_count)])
        return delta_sum, delta_max, counts.pop()

    def restore(self) -> None:
        for module, original in reversed(self.originals):
            module.attention = original
        self.originals.clear()


def normalized_prompt_movement(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Return mean-head relative L2 movement for every prompt position."""

    if current.shape != previous.shape:
        raise RuntimeError(
            f"prompt K/V shape changed between steps: {tuple(previous.shape)} -> {tuple(current.shape)}"
        )
    current_float = current.float()
    previous_float = previous.float()
    movement = (current_float - previous_float).norm(dim=-1).mean(dim=1).squeeze(0)
    norm = current_float.norm(dim=-1).mean(dim=1).squeeze(0).clamp_min(1e-6)
    return movement / norm


def install_offline_hybrid_collector(
    model: NamedModuleModel,
    prompt_length: int,
    suffix_length: int,
) -> OfflineHybridCollector:
    blocks = find_transformer_blocks(model)
    collector = OfflineHybridCollector(
        prompt_length=prompt_length,
        suffix_length=suffix_length,
        layer_count=len(blocks),
    )
    try:
        for block in blocks:
            original = block.attention
            if not isinstance(original, types.MethodType):
                raise TypeError("LLaDA block attention must be a bound method")
            accepts_block_mask = "block_mask" in inspect.signature(original).parameters
            block.attention = types.MethodType(
                make_offline_hybrid_attention_wrapper(collector, original, accepts_block_mask),
                block,
            )
            collector.originals.append((block, original))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        collector.restore()
        raise
    return collector


def make_offline_hybrid_attention_wrapper(
    collector: OfflineHybridCollector,
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
    ):
        collector.capture(block_self, q, k, v, attention_bias)
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
def generate_with_offline_hybrid_teacher(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    config: OfflineHybridTeacherConfig,
) -> OfflineHybridTeacherResult:
    validate_offline_hybrid_config(config)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise RuntimeError("offline hybrid teacher expects prompt_ids with shape [1, prompt]")

    prompt_length = int(prompt_ids.shape[1])
    blocks = find_transformer_blocks(model)
    layer_count = len(blocks)
    suffix_ids = torch.full(
        (1, config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    trace = ReferenceTrace(
        sum_scores=torch.zeros((layer_count, prompt_length), device=prompt_ids.device),
        max_scores=torch.zeros((layer_count, prompt_length), device=prompt_ids.device),
        union_mask=torch.full(
            (layer_count, prompt_length),
            fill_value=config.active_top_k == 0,
            dtype=torch.bool,
            device=prompt_ids.device,
        ),
        confidence_weight=config.confidence_weight,
        active_top_k=config.active_top_k,
    )
    collector = install_offline_hybrid_collector(model, prompt_length, config.gen_length)
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
            start = block_id * config.block_length
            end = (block_id + 1) * config.block_length
            block_mask = suffix_ids[:, start:end] == config.mask_id
            transfer_counts = get_num_transfer_tokens(block_mask, steps_per_block)
            for step_id in range(steps_per_block):
                mask_index = suffix_ids == config.mask_id
                logits, prompt_attention = full_sequence_logits_and_signals(
                    model,
                    prompt_ids,
                    suffix_ids,
                    collector,
                )
                x0, confidence = select_candidates(logits, suffix_ids, mask_index, candidate_config)
                confidence[:, end:] = -torch.inf
                transfer_index = build_transfer_index(confidence, transfer_counts[:, step_id])
                accumulate_reference(
                    trace,
                    prompt_attention,
                    transfer_index,
                    confidence,
                )
                suffix_ids[transfer_index] = x0[transfer_index]
    finally:
        collector.restore()

    teacher_raw = trace.aggregate(config.target_aggregation).detach().cpu().float()
    delta_raw, delta_step_max, delta_observation_count = collector.delta_result()
    delta_raw = delta_raw.detach().cpu().float()
    return OfflineHybridTeacherResult(
        generated_ids=suffix_ids.detach().cpu().squeeze(0),
        teacher_raw=teacher_raw,
        teacher_norm=normalize_scores(teacher_raw),
        ref_union_mask=trace.union_mask.detach().cpu(),
        ref_union_size_by_layer=trace.union_mask.sum(dim=-1).detach().cpu(),
        delta_raw=delta_raw,
        delta_norm=normalize_scores(delta_raw),
        delta_step_max=delta_step_max.detach().cpu().float(),
        delta_observation_count=delta_observation_count,
        reference_step_count=trace.step_count,
        commit_count=trace.commit_count,
        weight_sum=trace.weight_sum,
    )


def validate_offline_hybrid_config(config: OfflineHybridTeacherConfig) -> None:
    if config.gen_length % config.block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = config.gen_length // config.block_length
    if config.steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    if config.active_top_k < 0:
        raise RuntimeError("active_top_k must be non-negative; zero disables teacher-side top-k")
    if config.target_aggregation not in {"max", "sum"}:
        raise RuntimeError(f"unsupported target aggregation: {config.target_aggregation}")


def full_sequence_logits_and_signals(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    collector: OfflineHybridCollector,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
    collector.clear_attention()
    out = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    collector.assert_forward_complete()
    logits = out.logits[:, collector.prompt_length :].float()
    return logits, dict(collector.attention_by_layer)


def accumulate_reference(
    trace: ReferenceTrace,
    prompt_attention: dict[int, torch.Tensor],
    transfer_index: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    """Match ``future_pool_teacher.accumulate_future_pool`` without step tensors."""

    if transfer_index.shape[0] != 1:
        raise RuntimeError("offline hybrid teacher expects batch size 1")
    selected = transfer_index[0].nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return
    weights = confidence[0].index_select(0, selected).clamp_min(0.0).float()
    if not trace.confidence_weight:
        weights = torch.ones_like(weights, dtype=torch.float32)
    for layer_id, layer_attention in prompt_attention.items():
        scores = (layer_attention.index_select(0, selected).float() * weights.unsqueeze(-1)).sum(dim=0)
        if trace.active_top_k > 0:
            top_count = min(trace.active_top_k, int(scores.numel()))
            top_indices = torch.topk(scores, k=top_count, largest=True).indices
            trace.union_mask[layer_id].scatter_(dim=0, index=top_indices, value=True)
        trace.sum_scores[layer_id] += scores
        trace.max_scores[layer_id] = torch.maximum(trace.max_scores[layer_id], scores)
    trace.step_count += 1
    trace.commit_count += int(selected.numel())
    trace.weight_sum += float(weights.sum().detach().cpu())


__all__ = [
    "OfflineHybridCollector",
    "OfflineHybridTeacherConfig",
    "OfflineHybridTeacherResult",
    "ReferenceTrace",
    "accumulate_reference",
    "generate_with_offline_hybrid_teacher",
    "install_offline_hybrid_collector",
    "normalized_prompt_movement",
]
