from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.prompt_kv_cache import LayerPromptKV
from dllm_cache.budget.dynamic_prompt_kv import (
    build_keep_indices_by_layer,
    project_heads_at_positions,
    project_qkv,
    repeat_heads,
    run_block_mlp,
    suffix_logits_from_hidden,
)
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


@dataclass(slots=True)
class PoolActivePromptKVCache:
    prompt_length: int
    pool_budget: int
    active_budget: int
    teacher_scores: torch.Tensor
    layer_count: int
    selection_mode: str
    budget_mode: str
    min_pool_budget: int
    pool_budget_scale: float
    union_indices: torch.Tensor
    pool_indices_by_layer: dict[int, torch.Tensor]
    pool_offsets_by_layer: dict[int, torch.Tensor]
    layers: dict[int, LayerPromptKV] = field(default_factory=dict)

    @property
    def reduced_prompt_length(self) -> int:
        return int(self.union_indices.numel())

    def selected_prompt_indices(self, device: torch.device) -> torch.Tensor:
        return self.union_indices.to(device=device, dtype=torch.long)

    def sequence_positions(self, suffix_length: int, device: torch.device) -> torch.Tensor:
        prompt_positions = self.selected_prompt_indices(device)
        suffix_positions = torch.arange(
            self.prompt_length,
            self.prompt_length + suffix_length,
            device=device,
            dtype=torch.long,
        )
        return torch.cat([prompt_positions, suffix_positions], dim=0)

    def layer_pool_offsets(self, layer_id: int, device: torch.device) -> torch.Tensor:
        offsets = self.pool_offsets_by_layer.get(layer_id)
        if offsets is None:
            raise RuntimeError(f"missing prompt pool offsets for layer {layer_id}")
        return offsets.to(device=device, dtype=torch.long)

    def add_layer(self, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.layers[layer_id] = LayerPromptKV(key=key.detach(), value=value.detach())

    def layer(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> LayerPromptKV:
        cached = self.layers.get(layer_id)
        if cached is None:
            raise RuntimeError(f"pool-active prompt KV cache missing layer {layer_id}")
        return LayerPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
        )

    def has_all_layers(self) -> bool:
        return len(self.layers) == self.layer_count


def build_pool_active_prompt_kv_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
    pool_budget: int,
    active_budget: int,
    teacher_scores: torch.Tensor,
    selection_mode: str = "global",
    budget_mode: str = "fixed",
    min_pool_budget: int = 1,
    pool_budget_scale: float = 1.0,
) -> PoolActivePromptKVCache:
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    scores = teacher_scores.detach().float().cpu()
    if scores.shape != (len(blocks), prompt_length):
        raise RuntimeError(f"teacher score shape must be ({len(blocks)}, {prompt_length}), got {tuple(scores.shape)}")
    pool_indices_by_layer = build_keep_indices_by_layer(
        scores,
        prompt_length,
        pool_budget,
        len(blocks),
        selection_mode,
        budget_mode,
        min_pool_budget,
        pool_budget_scale,
    )
    union_indices = torch.unique(torch.cat(list(pool_indices_by_layer.values()), dim=0), sorted=True).to(dtype=torch.long)
    pool_offsets_by_layer = {
        layer_id: torch.searchsorted(union_indices, pool_indices)
        for layer_id, pool_indices in pool_indices_by_layer.items()
    }
    return PoolActivePromptKVCache(
        prompt_length=prompt_length,
        pool_budget=pool_budget,
        active_budget=max(1, min(active_budget, prompt_length)),
        teacher_scores=scores,
        layer_count=len(blocks),
        selection_mode=selection_mode,
        budget_mode=budget_mode,
        min_pool_budget=int(min_pool_budget),
        pool_budget_scale=float(pool_budget_scale),
        union_indices=union_indices,
        pool_indices_by_layer=pool_indices_by_layer,
        pool_offsets_by_layer=pool_offsets_by_layer,
    )


@torch.inference_mode()
def generate_with_pool_active_prompt_kv(
    input_ids: torch.Tensor,
    model: nn.Module,
    prompt_cache: PoolActivePromptKVCache,
    steps: int,
    gen_length: int,
    block_length: int,
    refresh_interval: int = 1,
    temperature: float = 0.0,
    mask_id: int = 126336,
) -> torch.Tensor:
    if refresh_interval <= 0:
        raise RuntimeError("refresh_interval must be positive")
    batch_size, prompt_length = input_ids.shape
    if prompt_length != prompt_cache.prompt_length:
        raise RuntimeError("input prompt length does not match pool-active prompt cache")
    x = torch.full((batch_size, prompt_length + gen_length), mask_id, dtype=torch.long, device=input_ids.device)
    x[:, :prompt_length] = input_ids
    validate_generation_shape(steps, gen_length, block_length)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    global_step = 0
    for block_id in range(num_blocks):
        start_idx = prompt_length + block_id * block_length
        end_idx = prompt_length + (block_id + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        transfer_counts = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            suffix_ids = x[:, prompt_length:]
            if not prompt_cache.has_all_layers() or global_step % refresh_interval == 0:
                logits = pool_active_suffix_logits(model, input_ids, suffix_ids, prompt_cache)
            else:
                logits = cached_pool_active_suffix_logits(model, suffix_ids, prompt_cache)
            x0, confidence = select_candidates(logits, x[:, prompt_length:], mask_index[:, prompt_length:], temperature)
            confidence[:, (block_id + 1) * block_length :] = -float("inf")
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                count = int(transfer_counts[batch_idx, step_idx].item())
                if count > 0:
                    transfer_index[batch_idx, torch.topk(confidence[batch_idx], k=count).indices] = True
            x[:, prompt_length:][transfer_index] = x0[transfer_index]
            global_step += 1
    return x[:, prompt_length:]


def validate_generation_shape(steps: int, gen_length: int, block_length: int) -> None:
    if gen_length % block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    if steps % (gen_length // block_length) != 0:
        raise RuntimeError("steps must be divisible by number of blocks")


def select_candidates(
    logits: torch.Tensor,
    suffix_ids: torch.Tensor,
    suffix_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    probs = F.softmax(logits, dim=-1)
    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(suffix_mask, x0, suffix_ids)
    confidence = torch.where(suffix_mask, confidence, torch.full_like(confidence, -torch.inf))
    return x0, confidence


def pool_active_suffix_logits(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    prompt_cache: PoolActivePromptKVCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    prompt_indices = prompt_cache.selected_prompt_indices(prompt_ids.device)
    x_prompt = decoder.transformer.wte(prompt_ids.index_select(dim=1, index=prompt_indices))
    x_suffix = decoder.transformer.wte(suffix_ids)
    x = torch.cat([x_prompt, x_suffix], dim=1)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    position_ids = prompt_cache.sequence_positions(suffix_ids.shape[1], x.device)
    for block in decoder.transformer.blocks:
        x = run_pool_active_block(block, x, position_ids, prompt_cache)
    return suffix_logits_from_hidden(decoder, x[:, prompt_cache.reduced_prompt_length :, :])


def cached_pool_active_suffix_logits(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    prompt_cache: PoolActivePromptKVCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    position_ids = torch.arange(prompt_cache.prompt_length, prompt_cache.prompt_length + suffix_ids.shape[1], device=x.device)
    for block in decoder.transformer.blocks:
        x = run_cached_pool_active_block(block, x, position_ids, prompt_cache)
    return suffix_logits_from_hidden(decoder, x)


def run_pool_active_block(
    block: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_cache: PoolActivePromptKVCache,
) -> torch.Tensor:
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, position_ids)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    layer_id = int(getattr(block, "layer_id"))
    pool_offsets = prompt_cache.layer_pool_offsets(layer_id, x.device)
    suffix_start = prompt_cache.reduced_prompt_length
    active_offsets = select_active_offsets(q_heads, k_heads, pool_offsets, suffix_start, prompt_cache.active_budget)
    pool_k, pool_v = k_heads.index_select(dim=2, index=pool_offsets), v_heads.index_select(dim=2, index=pool_offsets)
    active_k, active_v = k_heads.index_select(dim=2, index=active_offsets), v_heads.index_select(dim=2, index=active_offsets)
    suffix_k, suffix_v = k_heads[:, :, suffix_start:, :], v_heads[:, :, suffix_start:, :]
    prompt_key = torch.cat([pool_k, suffix_k], dim=2)
    prompt_value = torch.cat([pool_v, suffix_v], dim=2)
    suffix_key = torch.cat([active_k, suffix_k], dim=2)
    suffix_value = torch.cat([active_v, suffix_v], dim=2)
    prompt_att = F.scaled_dot_product_attention(q_heads[:, :, :suffix_start, :], prompt_key, prompt_value, dropout_p=0.0, is_causal=False)
    suffix_att = F.scaled_dot_product_attention(q_heads[:, :, suffix_start:, :], suffix_key, suffix_value, dropout_p=0.0, is_causal=False)
    att = torch.cat([prompt_att, suffix_att], dim=2)
    prompt_cache.add_layer(layer_id, active_k, active_v)
    x = x + block.dropout(block.attn_out(att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])))
    return run_block_mlp(block, x)


def select_active_offsets(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    pool_offsets: torch.Tensor,
    suffix_start: int,
    active_budget: int,
) -> torch.Tensor:
    active_count = min(max(1, active_budget), int(pool_offsets.numel()))
    if active_count == int(pool_offsets.numel()):
        return pool_offsets
    suffix_q = q_heads[:, :, suffix_start:, :]
    pool_k = k_heads.index_select(dim=2, index=pool_offsets)
    scores = torch.matmul(suffix_q.float(), pool_k.float().transpose(-1, -2)) / math.sqrt(float(q_heads.shape[-1]))
    pool_rank = scores.mean(dim=(0, 1, 2))
    active = torch.topk(pool_rank, k=active_count, largest=True).indices
    return pool_offsets.index_select(0, active).sort().values


def run_cached_pool_active_block(
    block: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_cache: PoolActivePromptKVCache,
) -> torch.Tensor:
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, position_ids)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    layer_id = int(getattr(block, "layer_id"))
    layer_cache = prompt_cache.layer(layer_id, x.device, k_heads.dtype)
    key = torch.cat([layer_cache.key, k_heads], dim=2)
    value = torch.cat([layer_cache.value, v_heads], dim=2)
    att = F.scaled_dot_product_attention(q_heads, key, value, dropout_p=0.0, is_causal=False)
    x = x + block.dropout(block.attn_out(att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])))
    return run_block_mlp(block, x)
