from __future__ import annotations

import inspect
import types

import torch
import torch.nn as nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.prompt_kv_cache import PromptKVCache, project_heads, repeat_heads


def build_full_prompt_kv_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
) -> PromptKVCache:
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    cache = PromptKVCache(
        prompt_length=prompt_length,
        budget=prompt_length,
        teacher_scores=torch.ones((len(blocks), prompt_length), dtype=torch.float32),
        layer_count=len(blocks),
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        block.attention = types.MethodType(
            make_full_prompt_attention(cache, original, accepts_block_mask),
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


def make_full_prompt_attention(
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
        capture_full_prompt_kv(block_self, q, k, v, cache)
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


def capture_full_prompt_kv(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PromptKVCache,
) -> None:
    if q.shape[1] != cache.prompt_length:
        raise RuntimeError("full prompt KV cache must run on prompt-only input")
    layer_id = int(getattr(block, "layer_id"))
    q_heads, k_heads, v_heads = project_heads(block, q, k, v, position_offset=0)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    cache.add_layer(layer_id, k_heads, v_heads)
