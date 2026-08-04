from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional

from .refresh_cache import RefreshCacheError, RefreshPromptKVCache


class LLaDAModel(Protocol):
    model: nn.Module


@dataclass(frozen=True, slots=True)
class _BlockRequest:
    hidden: torch.Tensor
    positions: torch.Tensor
    cache: RefreshPromptKVCache


@dataclass(frozen=True, slots=True)
class _BlockAttention:
    hidden: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class LladaRefreshRunner:
    """Run LLaDA suffix decoding against periodically refreshed prompt KVs."""

    __slots__ = ("cache", "model")

    def __init__(self, model: LLaDAModel, cache: RefreshPromptKVCache) -> None:
        self.model = model
        self.cache = cache

    def start_step(self, step_id: int) -> bool:
        return self.cache.start_step(step_id)

    def refresh_logits(
        self,
        prompt_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_ids.shape[1] != self.cache.prompt_length:
            raise RefreshCacheError("prompt input does not match cache length")
        decoder = self.model.model
        cache_indices = self.cache.cached_prompt_indices(prompt_ids.device)
        cached_prompt_ids = prompt_ids.index_select(1, cache_indices)
        prompt_hidden = decoder.transformer.wte(cached_prompt_ids)
        suffix_hidden = decoder.transformer.wte(suffix_ids)
        hidden = torch.cat((prompt_hidden, suffix_hidden), dim=1)
        hidden = _prepare_embeddings(decoder, hidden)
        suffix_positions = torch.arange(
            self.cache.prompt_length,
            self.cache.prompt_length + suffix_ids.shape[1],
            device=hidden.device,
        )
        positions = torch.cat((cache_indices, suffix_positions))
        for block in decoder.transformer.blocks:
            request = _BlockRequest(hidden, positions, self.cache)
            hidden = _run_refresh_block(block, request)
        self.cache.assert_complete()
        return _suffix_logits(decoder, hidden[:, self.cache.cache_size :])

    def cached_logits(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        decoder = self.model.model
        hidden = _prepare_embeddings(decoder, decoder.transformer.wte(suffix_ids))
        positions = torch.arange(
            self.cache.prompt_length,
            self.cache.prompt_length + suffix_ids.shape[1],
            device=hidden.device,
        )
        for block in decoder.transformer.blocks:
            request = _BlockRequest(hidden, positions, self.cache)
            hidden = _run_cached_block(block, request)
        return _suffix_logits(decoder, hidden)


def _prepare_embeddings(decoder: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    config = decoder.config
    if bool(config.input_emb_norm):
        hidden = hidden * (float(config.d_model) ** 0.5)
    return decoder.transformer.emb_drop(hidden)


def _run_refresh_block(
    block: nn.Module,
    request: _BlockRequest,
) -> torch.Tensor:
    q_heads, k_heads, v_heads = _attention_heads(
        block,
        request.hidden,
        request.positions,
    )
    prompt_length = request.cache.cache_size
    layer_id = int(block.layer_id)
    request.cache.add_layer(
        layer_id,
        k_heads[:, :, :prompt_length],
        v_heads[:, :, :prompt_length],
    )
    prompt = request.cache.layer(layer_id, request.hidden.device, k_heads.dtype)
    key = torch.cat((prompt.key, k_heads[:, :, prompt_length:]), dim=2)
    value = torch.cat((prompt.value, v_heads[:, :, prompt_length:]), dim=2)
    attention = _BlockAttention(request.hidden, q_heads, key, value)
    return _finish_block(block, attention)


def _run_cached_block(
    block: nn.Module,
    request: _BlockRequest,
) -> torch.Tensor:
    q_heads, k_heads, v_heads = _attention_heads(
        block,
        request.hidden,
        request.positions,
    )
    prompt = request.cache.layer(
        int(block.layer_id),
        request.hidden.device,
        k_heads.dtype,
    )
    key = torch.cat((prompt.key, k_heads), dim=2)
    value = torch.cat((prompt.value, v_heads), dim=2)
    attention = _BlockAttention(request.hidden, q_heads, key, value)
    return _finish_block(block, attention)


def _attention_heads(
    block: nn.Module,
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = block.attn_norm(hidden)
    if hasattr(block, "att_proj"):
        q, k, v = block.att_proj(normalized).split(block.fused_dims, dim=-1)
    else:
        q = block.q_proj(normalized)
        k = block.k_proj(normalized)
        v = block.v_proj(normalized)
    q_norm = getattr(block, "q_norm", None)
    k_norm = getattr(block, "k_norm", None)
    if q_norm is not None and k_norm is not None:
        q = q_norm(q).to(dtype=k.dtype)
        k = k_norm(k).to(dtype=k.dtype)
    config = block.config
    query_heads = int(config.n_heads)
    kv_heads = int(config.effective_n_kv_heads)
    head_dim = q.shape[-1] // query_heads
    q_heads = q.view(q.shape[0], q.shape[1], query_heads, head_dim).transpose(1, 2)
    k_heads = k.view(k.shape[0], k.shape[1], kv_heads, head_dim).transpose(1, 2)
    v_heads = v.view(v.shape[0], v.shape[1], kv_heads, head_dim).transpose(1, 2)
    if bool(config.rope):
        q_heads, k_heads = _apply_rope(block, q_heads, k_heads, positions)
    if query_heads != kv_heads:
        if query_heads % kv_heads != 0:
            raise RefreshCacheError("query heads must be divisible by KV heads")
        repeats = query_heads // kv_heads
        k_heads = k_heads.repeat_interleave(repeats, dim=1)
        v_heads = v_heads.repeat_interleave(repeats, dim=1)
    return q_heads, k_heads, v_heads


def _apply_rope(
    block: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_precision = bool(block.config.rope_full_precision)
    query_work = query.float() if full_precision else query
    key_work = key.float() if full_precision else key
    total_length = int(positions.max().item()) + 1
    with torch.autocast(query.device.type, enabled=False):
        sine, cosine = block.rotary_emb.get_rotary_embedding(
            total_length,
            query_work.device,
        )
        sine = sine.type_as(query_work).index_select(2, positions)
        cosine = cosine.type_as(query_work).index_select(2, positions)
        query_work = block.rotary_emb.apply_rotary_pos_emb(sine, cosine, query_work)
        key_work = block.rotary_emb.apply_rotary_pos_emb(sine, cosine, key_work)
    return query_work.type_as(query), key_work.type_as(key)


def _finish_block(
    block: nn.Module,
    attention: _BlockAttention,
) -> torch.Tensor:
    attended = functional.scaled_dot_product_attention(
        attention.query,
        attention.key,
        attention.value,
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).contiguous().view_as(attention.hidden)
    hidden = attention.hidden + block.dropout(block.attn_out(attended))
    residual = hidden
    normalized = block.ff_norm(hidden)
    if hasattr(block, "up_proj"):
        normalized = block.act(block.ff_proj(normalized)) * block.up_proj(normalized)
    else:
        normalized = block.act(block.ff_proj(normalized))
    return residual + block.dropout(block.ff_out(normalized))


def _suffix_logits(decoder: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    normalized = decoder.transformer.ln_f(hidden)
    if bool(decoder.config.weight_tying):
        logits = functional.linear(normalized, decoder.transformer.wte.weight)
    else:
        logits = decoder.transformer.ff_out(normalized)
    if bool(decoder.config.scale_logits):
        logits.mul_(1 / math.sqrt(float(decoder.config.d_model)))
    return logits.float()
