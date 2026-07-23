from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks


@dataclass(frozen=True, slots=True)
class LayerPromptKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class PromptKVCache:
    prompt_length: int
    budget: int
    teacher_scores: torch.Tensor
    layer_count: int
    layers: dict[int, LayerPromptKV] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def add_layer(self, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.layers[layer_id] = LayerPromptKV(key=key.detach(), value=value.detach())

    def layer(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> LayerPromptKV:
        cached = self.layers.get(layer_id)
        if cached is None:
            raise RuntimeError(f"prompt KV cache missing layer {layer_id}")
        return LayerPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
        )

    def restore(self) -> None:
        for module, original in self.originals:
            module.attention = original
        self.originals.clear()

    def assert_complete(self) -> None:
        missing = sorted(set(range(self.layer_count)) - set(self.layers))
        if missing:
            raise RuntimeError(f"prompt KV cache missing layers: {missing}")


def build_prompt_kv_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
    budget: int,
    teacher_scores: torch.Tensor,
) -> PromptKVCache:
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    cache = PromptKVCache(
        prompt_length=prompt_length,
        budget=budget,
        teacher_scores=teacher_scores.float().cpu(),
        layer_count=len(blocks),
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        block.attention = types.MethodType(
            make_collecting_attention(cache, original, accepts_block_mask),
            block,
        )
        cache.originals.append((block, original))
    try:
        with torch.inference_mode():
            model(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                use_cache=False,
                return_dict=True,
            )
    finally:
        cache.restore()
    cache.assert_complete()
    return cache


def make_collecting_attention(
    cache: PromptKVCache,
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
        capture_prompt_kv(block_self, q, k, v, cache)
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


def capture_prompt_kv(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PromptKVCache,
) -> None:
    if q.shape[1] != cache.prompt_length:
        raise RuntimeError("prompt KV precompute must run on prompt-only input")
    layer_id = int(getattr(block, "layer_id"))
    q_heads, k_heads, v_heads = project_heads(block, q, k, v, position_offset=0)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    keep = topk_prompt_indices(cache, layer_id, q.device)
    cache.add_layer(
        layer_id,
        k_heads.index_select(dim=2, index=keep),
        v_heads.index_select(dim=2, index=keep),
    )


def project_heads(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, q_len, channels = q.size()
    _, k_len, _ = k.size()
    dtype = k.dtype
    q_norm = getattr(block, "q_norm", None)
    k_norm = getattr(block, "k_norm", None)
    if q_norm is not None and k_norm is not None:
        q = q_norm(q).to(dtype=dtype)
        k = k_norm(k).to(dtype=dtype)
    config = getattr(block, "config")
    head_dim = channels // int(config.n_heads)
    q_heads = q.view(batch_size, q_len, int(config.n_heads), head_dim).transpose(1, 2)
    k_heads = k.view(batch_size, k_len, int(config.effective_n_kv_heads), head_dim).transpose(1, 2)
    v_heads = v.view(batch_size, k_len, int(config.effective_n_kv_heads), head_dim).transpose(1, 2)
    if bool(config.rope):
        q_heads, k_heads = apply_rope_at_offset(block, q_heads, k_heads, position_offset)
    return q_heads, k_heads, v_heads


def apply_rope_at_offset(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = getattr(block, "config")
    q_work = q.float() if bool(config.rope_full_precision) else q
    k_work = k.float() if bool(config.rope_full_precision) else k
    total_len = position_offset + k_work.shape[-2]
    with torch.autocast(q.device.type, enabled=False):
        pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(total_len, q_work.device)
        pos_sin = pos_sin.type_as(q_work)
        pos_cos = pos_cos.type_as(q_work)
        start = position_offset
        end = position_offset + q_work.shape[-2]
        q_work = block.rotary_emb.apply_rotary_pos_emb(
            pos_sin[:, :, start:end, :],
            pos_cos[:, :, start:end, :],
            q_work,
        )
        k_work = block.rotary_emb.apply_rotary_pos_emb(
            pos_sin[:, :, position_offset:total_len, :],
            pos_cos[:, :, position_offset:total_len, :],
            k_work,
        )
    return q_work.type_as(q), k_work.type_as(k)


def repeat_heads(states: torch.Tensor, head_count: int) -> torch.Tensor:
    state_heads = states.shape[1]
    if head_count % state_heads != 0:
        raise RuntimeError("query head count must be divisible by state head count")
    return states.repeat_interleave(head_count // state_heads, dim=1)


def topk_prompt_indices(cache: PromptKVCache, layer_id: int, device: torch.device) -> torch.Tensor:
    scores = cache.teacher_scores[layer_id]
    if scores.numel() != cache.prompt_length:
        raise RuntimeError("teacher score width does not match prompt length")
    budget = max(1, min(cache.budget, cache.prompt_length))
    return torch.topk(scores.to(device), k=budget, largest=True).indices
