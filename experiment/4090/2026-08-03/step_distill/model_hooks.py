from __future__ import annotations

import inspect
import math
import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class NamedModuleModel(Protocol):
    def named_modules(self) -> Iterable[tuple[str, nn.Module]]: ...


@dataclass(frozen=True, slots=True)
class HookError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class SuffixPromptAttentionCollector:
    """Mutable per-forward buffer for suffix-to-prompt attention."""

    __slots__ = (
        "attention_by_layer",
        "layer_count",
        "originals",
        "prompt_length",
        "suffix_length",
    )

    def __init__(
        self, prompt_length: int, suffix_length: int, layer_count: int
    ) -> None:
        self.prompt_length = prompt_length
        self.suffix_length = suffix_length
        self.layer_count = layer_count
        self.attention_by_layer: dict[int, torch.Tensor] = {}
        self.originals: list[tuple[nn.Module, types.MethodType]] = []

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
            raise HookError("per-step teacher expects batch size one")
        sequence_length = self.prompt_length + self.suffix_length
        if q.shape[1] != sequence_length:
            raise HookError(
                f"collector expected {sequence_length} tokens, received {q.shape[1]}"
            )
        q_heads, k_heads = project_attention_heads(block, q, k)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        suffix_q = q_heads[:, :, self.prompt_length :, :]
        scores = suffix_q.float() @ k_heads.float().transpose(-1, -2)
        scores = scores / math.sqrt(float(suffix_q.shape[-1]))
        if attention_bias is not None:
            scores = (
                scores
                + attention_bias[
                    :, :, self.prompt_length : sequence_length, :sequence_length
                ].float()
            )
        attention = torch.softmax(scores, dim=-1)
        prompt_attention = attention[..., : self.prompt_length].mean(dim=1).squeeze(0)
        self.attention_by_layer[int(block.layer_id)] = prompt_attention.detach()

    def stacked_attention(self) -> torch.Tensor:
        missing = sorted(set(range(self.layer_count)) - set(self.attention_by_layer))
        if missing:
            raise HookError(f"trajectory attention missing layers: {missing}")
        return torch.stack(
            [self.attention_by_layer[layer_id] for layer_id in range(self.layer_count)],
            dim=0,
        )

    def restore(self) -> None:
        for module, original in reversed(self.originals):
            setattr(module, "attention", original)
        self.originals.clear()


def find_transformer_blocks(model: NamedModuleModel) -> nn.ModuleList:
    for name, module in model.named_modules():
        if name == "model.transformer.blocks":
            if not isinstance(module, nn.ModuleList):
                raise HookError("model.transformer.blocks is not a ModuleList")
            return module
    raise HookError("could not find model.transformer.blocks")


def project_attention_heads(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_length, channels = q.shape
    key_length = k.shape[1]
    key_dtype = k.dtype
    q_norm = getattr(block, "q_norm", None)
    k_norm = getattr(block, "k_norm", None)
    if q_norm is not None and k_norm is not None:
        q = q_norm(q).to(dtype=key_dtype)
        k = k_norm(k).to(dtype=key_dtype)
    config = block.config
    query_heads = int(config.n_heads)
    kv_heads = int(config.effective_n_kv_heads)
    head_dim = channels // query_heads
    q_heads = q.view(batch_size, query_length, query_heads, head_dim).transpose(1, 2)
    k_heads = k.view(batch_size, key_length, kv_heads, head_dim).transpose(1, 2)
    if bool(config.rope):
        q_heads, k_heads = block.rotary_emb(q_heads, k_heads)
    return q_heads, k_heads


def repeat_kv_heads(k_heads: torch.Tensor, query_head_count: int) -> torch.Tensor:
    kv_head_count = int(k_heads.shape[1])
    if query_head_count % kv_head_count != 0:
        raise HookError("query head count must be divisible by KV head count")
    return k_heads.repeat_interleave(query_head_count // kv_head_count, dim=1)


def install_suffix_prompt_attention_collector(
    model: NamedModuleModel,
    *,
    prompt_length: int,
    suffix_length: int,
) -> SuffixPromptAttentionCollector:
    """Install wrappers transactionally so partial installation cannot leak."""
    blocks = find_transformer_blocks(model)
    collector = SuffixPromptAttentionCollector(
        prompt_length, suffix_length, len(blocks)
    )
    try:
        for block in blocks:
            original = block.attention
            accepts_block_mask = "block_mask" in inspect.signature(original).parameters
            wrapped = _make_wrapped_attention(collector, original, accepts_block_mask)
            setattr(block, "attention", types.MethodType(wrapped, block))
            collector.originals.append((block, original))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        collector.restore()
        raise
    return collector


def _make_wrapped_attention(
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
    ):
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
        return original(
            q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
        )

    return wrapped_attention
