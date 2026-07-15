from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from revealed_answer.prompt_kv_cache import PromptKVCache, project_heads, repeat_heads


def prompt_kv_suffix_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_cache: PromptKVCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("prompt KV suffix forward supports block_group_size=1 only")
    suffix_ids = input_ids[:, prompt_cache.prompt_length :]
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    for block in decoder.transformer.blocks:
        x = run_suffix_block(block, x, prompt_cache)
    x = decoder.transformer.ln_f(x)
    if bool(config.weight_tying):
        logits = F.linear(x, decoder.transformer.wte.weight, None)
    else:
        logits = decoder.transformer.ff_out(x)
    if bool(config.scale_logits):
        logits.mul_(1 / math.sqrt(float(config.d_model)))
    return logits.float()


def run_suffix_block(
    block: nn.Module,
    x: torch.Tensor,
    prompt_cache: PromptKVCache,
) -> torch.Tensor:
    x_normed = block.attn_norm(x)
    if hasattr(block, "att_proj"):
        q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
    else:
        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)
    q_heads, k_heads, v_heads = project_heads(
        block,
        q,
        k,
        v,
        position_offset=prompt_cache.prompt_length,
    )
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    layer_cache = prompt_cache.layer(int(getattr(block, "layer_id")), x.device, k_heads.dtype)
    key = torch.cat([layer_cache.key, k_heads], dim=2)
    value = torch.cat([layer_cache.value, v_heads], dim=2)
    att = F.scaled_dot_product_attention(q_heads, key, value, dropout_p=0.0, is_causal=False)
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_suffix_mlp(block, x)


def run_suffix_mlp(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = block.ff_norm(x)
    if hasattr(block, "up_proj"):
        x = block.act(block.ff_proj(x)) * block.up_proj(x)
    else:
        x = block.act(block.ff_proj(x))
    x = block.ff_out(x)
    return residual + block.dropout(x)
