from __future__ import annotations

import inspect
import math
import types
from typing import Protocol, cast

import torch
from torch import nn

from .model_hooks import (
    NamedModuleModel,
    find_transformer_blocks,
    project_attention_heads,
    repeat_kv_heads,
)

DISTILLATION_BUDGET = 960


class AttentionConfig(Protocol):
    n_heads: int
    effective_n_kv_heads: int


class AttentionBlock(Protocol):
    layer_id: int
    config: AttentionConfig
    attn_out: nn.Module


def _attention_method(block: nn.Module) -> types.MethodType:
    method = getattr(block, "attention", None)
    if not isinstance(method, types.MethodType):
        raise TypeError("LLaDA block attention must be a bound method")
    return method


def _set_attention(block: nn.Module, method: types.MethodType) -> None:
    object.__setattr__(block, "attention", method)


def _attention_block(block: nn.Module) -> AttentionBlock:
    return cast(AttentionBlock, cast(object, block))


class DistributionPruningController:
    """Own reversible attention wrappers and one differentiable step mask."""

    def __init__(self, prompt_length: int, layer_count: int) -> None:
        if prompt_length <= DISTILLATION_BUDGET:
            raise ValueError("B=960 training requires prompts longer than 960 tokens")
        if layer_count <= 0:
            raise ValueError("layer count must be positive")
        self.prompt_length = prompt_length
        self.layer_count = layer_count
        self.selector_scores: torch.Tensor | None = None
        self.current_layer_score: tuple[int, torch.Tensor] | None = None
        self.is_full_attention = False
        self.keep_indices: dict[int, torch.Tensor] = {}
        self.originals: list[tuple[nn.Module, types.MethodType]] = []

    def set_scores(self, selector_scores: torch.Tensor) -> None:
        expected = (self.layer_count, self.prompt_length)
        if selector_scores.shape != expected:
            raise ValueError(f"selector scores must have shape {expected}")
        self.selector_scores = selector_scores
        self.current_layer_score = None
        self.is_full_attention = False
        self.keep_indices.clear()

    def disable(self) -> None:
        self.selector_scores = None
        self.current_layer_score = None
        self.is_full_attention = False
        self.keep_indices.clear()

    def enable_full_attention(self) -> None:
        """Use the manual attention operator while keeping the full prompt."""
        self.selector_scores = None
        self.current_layer_score = None
        self.is_full_attention = True
        self.keep_indices.clear()

    def set_layer_score(self, layer_id: int, score: torch.Tensor) -> None:
        if self.selector_scores is None:
            raise ValueError("cannot set a layer score while pruning is disabled")
        if score.shape != (self.prompt_length,):
            raise ValueError("current layer score must have shape [prompt]")
        self.current_layer_score = (layer_id, score)

    def layer_scores(self, layer_id: int) -> torch.Tensor | None:
        if self.selector_scores is None:
            return None
        if self.current_layer_score is not None:
            current_layer, current_score = self.current_layer_score
            if current_layer == layer_id:
                return current_score.unsqueeze(0)
        return self.selector_scores[layer_id].unsqueeze(0)

    def restore(self) -> None:
        for block, original in reversed(self.originals):
            _set_attention(block, original)
        self.originals.clear()


def install_distribution_pruner(
    model: NamedModuleModel,
    controller: DistributionPruningController,
    *,
    gate_temperature: float,
) -> None:
    if gate_temperature <= 0.0:
        raise ValueError("gate temperature must be positive")
    blocks = find_transformer_blocks(model)
    if len(blocks) != controller.layer_count:
        raise ValueError("controller layer count does not match model")
    try:
        for block in blocks:
            original = _attention_method(block)
            accepts_block_mask = "block_mask" in inspect.signature(original).parameters
            wrapped = _make_attention_wrapper(
                controller,
                original,
                accepts_block_mask=accepts_block_mask,
                gate_temperature=gate_temperature,
            )
            _set_attention(block, types.MethodType(wrapped, block))
            controller.originals.append((block, original))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        controller.restore()
        raise


def _make_attention_wrapper(
    controller: DistributionPruningController,
    original: types.MethodType,
    *,
    accepts_block_mask: bool,
    gate_temperature: float,
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
        layer_id = _attention_block(block_self).layer_id
        scores = controller.layer_scores(layer_id)
        manual_attention = scores is not None or controller.is_full_attention
        if not manual_attention or layer_past is not None or use_cache or block_mask is not None:
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
            return original(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        keep_count = DISTILLATION_BUDGET
        if controller.is_full_attention:
            scores = torch.zeros(
                (q.shape[0], controller.prompt_length),
                device=q.device,
                dtype=q.dtype,
            )
            keep_count = controller.prompt_length
        if scores is None:
            raise RuntimeError("manual pruning requires prompt scores")
        output, keep = straight_through_pruned_attention(
            block_self,
            q,
            k,
            v,
            attention_bias,
            prompt_length=controller.prompt_length,
            selector_scores=scores,
            keep_count=keep_count,
            gate_temperature=gate_temperature,
        )
        controller.keep_indices[layer_id] = keep.detach()
        return output, None

    return wrapped_attention


def straight_through_pruned_attention(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: torch.Tensor | None,
    *,
    prompt_length: int,
    selector_scores: torch.Tensor,
    keep_count: int,
    gate_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact hard Top-K forward with a dense post-softmax ST gradient."""
    batch_size, query_length, channels = q.shape
    if selector_scores.shape != (batch_size, prompt_length):
        raise ValueError("selector scores must have shape [batch, prompt]")
    if query_length <= prompt_length or not 0 < keep_count <= prompt_length:
        raise ValueError("prompt/suffix lengths and keep count are incompatible")
    if gate_temperature <= 0.0:
        raise ValueError("gate temperature must be positive")

    q_heads, k_heads = project_attention_heads(block, q, k)
    typed_block = _attention_block(block)
    config = typed_block.config
    kv_heads = int(config.effective_n_kv_heads)
    head_dim = channels // int(config.n_heads)
    v_heads = v.view(batch_size, query_length, kv_heads, head_dim).transpose(1, 2)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_kv_heads(v_heads, q_heads.shape[1])

    attention_logits = q_heads.float() @ k_heads.float().transpose(-1, -2)
    attention_logits = attention_logits / math.sqrt(float(head_dim))
    if attention_bias is not None:
        bias = attention_bias[..., :query_length, :query_length]
        if bias.dtype == torch.bool:
            attention_logits = attention_logits.masked_fill(~bias, -torch.inf)
        else:
            attention_logits = attention_logits + bias.float()
    probabilities = torch.softmax(attention_logits, dim=-1)

    keep = torch.topk(selector_scores, k=keep_count, dim=-1).indices
    hard = torch.zeros_like(selector_scores).scatter(-1, keep, 1.0)
    threshold = selector_scores.detach().topk(keep_count, dim=-1).values[:, -1:]
    soft = torch.sigmoid((selector_scores - threshold) / gate_temperature)
    prompt_gate = soft + (hard - soft).detach()
    suffix_gate = torch.ones(
        (batch_size, query_length - prompt_length),
        dtype=prompt_gate.dtype,
        device=prompt_gate.device,
    )
    key_gate = torch.cat((prompt_gate, suffix_gate), dim=-1)[:, None, None, :]
    gated = probabilities * key_gate.float()
    normalized = gated / gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attended = normalized @ v_heads.float()
    merged = attended.to(v.dtype).transpose(1, 2).contiguous().view_as(q)
    output = typed_block.attn_out(merged)
    if not isinstance(output, torch.Tensor):
        raise TypeError("LLaDA attention output projection must return a tensor")
    return output, keep


__all__ = [
    "DISTILLATION_BUDGET",
    "DistributionPruningController",
    "install_distribution_pruner",
    "straight_through_pruned_attention",
]
