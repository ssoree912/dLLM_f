"""Measure how much each prompt token's cached K/V moves across denoising steps.

LLaDA attends bidirectionally, so a prompt token's representation keeps absorbing
the suffix as generation fills it in. Freezing the prompt K/V at step 0 is therefore
lossy in a way an autoregressive cache never is, and the loss is not spread evenly:
some prompt positions barely move while others drift far.

This collector records, per (layer, prompt position):

  cumulative   mean_t ||v_t - v_0|| / ||v_0||        how wrong a permanent freeze is
  stepwise     mean_t ||v_t - v_{t-1}|| / ||v_t||    how fast a refreshed entry goes stale
  attention    mean_t sum_suffix softmax(q_s k_i)    how much the suffix actually reads it
  weighted     mean_t attention_t[i] * ||v_t - v_0||  the attention-weighted freeze error

`weighted` is the quantity that matters for a cache: drift on a position nothing
attends to costs nothing.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.prompt_kv_cache import project_heads, repeat_heads


@dataclass(slots=True)
class PromptDriftCollector:
    prompt_length: int
    layer_count: int
    cumulative: dict[int, torch.Tensor] = field(default_factory=dict)
    stepwise: dict[int, torch.Tensor] = field(default_factory=dict)
    attention: dict[int, torch.Tensor] = field(default_factory=dict)
    weighted: dict[int, torch.Tensor] = field(default_factory=dict)
    first_value: dict[int, torch.Tensor] = field(default_factory=dict)
    previous_value: dict[int, torch.Tensor] = field(default_factory=dict)
    observations: dict[int, int] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def restore(self) -> None:
        for module, original in reversed(self.originals):
            module.attention = original
        self.originals.clear()

    def result(self) -> dict[str, torch.Tensor]:
        """Stack the per-layer accumulators into [layer, prompt] tensors."""
        missing = sorted(set(range(self.layer_count)) - set(self.cumulative))
        if missing:
            raise RuntimeError(f"drift collector missing layers: {missing}")
        stacked: dict[str, torch.Tensor] = {}
        for name, source in (
            ("cumulative", self.cumulative),
            ("stepwise", self.stepwise),
            ("attention", self.attention),
            ("weighted", self.weighted),
        ):
            stacked[name] = torch.stack(
                [
                    source[layer_id] / max(1, self.observations[layer_id])
                    for layer_id in range(self.layer_count)
                ]
            )
        return stacked


def install_prompt_drift_collector(
    model: NamedModuleModel,
    prompt_length: int,
) -> PromptDriftCollector:
    blocks = find_transformer_blocks(model)
    collector = PromptDriftCollector(
        prompt_length=prompt_length,
        layer_count=len(blocks),
    )
    try:
        for block in blocks:
            original = block.attention
            if not isinstance(original, types.MethodType):
                raise TypeError("LLaDA block attention must be a bound method")
            wrapped = _make_wrapper(collector, original)
            block.attention = types.MethodType(wrapped, block)
            collector.originals.append((block, original))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        collector.restore()
        raise
    return collector


def _make_wrapper(collector: PromptDriftCollector, original: types.MethodType):
    def wrapped_attention(
        block_self: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        _observe(collector, block_self, q, k, v)
        return original(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            **kwargs,
        )

    return wrapped_attention


@torch.no_grad()
def _observe(
    collector: PromptDriftCollector,
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    prompt_length = collector.prompt_length
    if q.shape[1] <= prompt_length:
        return  # prompt-only prefill carries no suffix to read the prompt
    layer_id = int(block.layer_id)
    q_heads, k_heads, v_heads = project_heads(block, q, k, v, position_offset=0)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])

    prompt_value = v_heads[:, :, :prompt_length, :].float()
    norm = prompt_value.norm(dim=-1).mean(dim=1).squeeze(0).clamp_min(1e-6)

    device = prompt_value.device
    if layer_id not in collector.first_value:
        zeros = torch.zeros(prompt_length, device=device)
        collector.first_value[layer_id] = prompt_value.clone()
        collector.previous_value[layer_id] = prompt_value.clone()
        collector.cumulative[layer_id] = zeros.clone()
        collector.stepwise[layer_id] = zeros.clone()
        collector.attention[layer_id] = zeros.clone()
        collector.weighted[layer_id] = zeros.clone()
        collector.observations[layer_id] = 0

    from_first = (prompt_value - collector.first_value[layer_id]).norm(dim=-1)
    from_previous = (prompt_value - collector.previous_value[layer_id]).norm(dim=-1)
    cumulative = from_first.mean(dim=1).squeeze(0) / norm
    stepwise = from_previous.mean(dim=1).squeeze(0) / norm

    # How much the suffix queries actually read each prompt key at this step.
    suffix_query = q_heads[:, :, prompt_length:, :].float()
    prompt_key = k_heads[:, :, :prompt_length, :].float()
    scores = suffix_query @ prompt_key.transpose(-1, -2)
    scores = scores / math.sqrt(float(q_heads.shape[-1]))
    attention = F.softmax(scores, dim=-1).sum(dim=-2).mean(dim=1).squeeze(0)

    collector.cumulative[layer_id] += cumulative
    collector.stepwise[layer_id] += stepwise
    collector.attention[layer_id] += attention
    collector.weighted[layer_id] += attention * cumulative
    collector.observations[layer_id] += 1
    collector.previous_value[layer_id] = prompt_value.clone()


__all__ = ["PromptDriftCollector", "install_prompt_drift_collector"]
