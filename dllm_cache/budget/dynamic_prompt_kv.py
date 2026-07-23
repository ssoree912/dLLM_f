from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


@dataclass(frozen=True, slots=True)
class LayerPromptKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class DynamicPromptKVCache:
    prompt_length: int
    budget: int
    teacher_scores: torch.Tensor
    layer_count: int
    selection_mode: str
    union_indices: torch.Tensor
    keep_indices_by_layer: dict[int, torch.Tensor]
    keep_offsets_by_layer: dict[int, torch.Tensor]
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

    def layer_prompt_offsets(self, layer_id: int, device: torch.device) -> torch.Tensor:
        offsets = self.keep_offsets_by_layer.get(layer_id)
        if offsets is None:
            raise RuntimeError(f"missing prompt keep offsets for layer {layer_id}")
        return offsets.to(device=device, dtype=torch.long)

    def add_layer(self, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.layers[layer_id] = LayerPromptKV(key=key.detach(), value=value.detach())

    def layer(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> LayerPromptKV:
        cached = self.layers.get(layer_id)
        if cached is None:
            raise RuntimeError(f"dynamic prompt KV cache missing layer {layer_id}")
        return LayerPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
        )

    def has_all_layers(self) -> bool:
        return len(self.layers) == self.layer_count


def build_dynamic_prompt_kv_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
    budget: int,
    teacher_scores: torch.Tensor,
    selection_mode: str = "layer_union",
    budget_mode: str = "fixed",
    min_budget: int = 1,
    budget_scale: float = 1.0,
) -> DynamicPromptKVCache:
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    scores = teacher_scores.detach().float().cpu()
    if scores.shape != (len(blocks), prompt_length):
        raise RuntimeError(
            "teacher score shape must be "
            f"({len(blocks)}, {prompt_length}), got {tuple(scores.shape)}"
        )
    keep_indices_by_layer = build_keep_indices_by_layer(
        scores,
        prompt_length,
        budget,
        len(blocks),
        selection_mode,
        budget_mode,
        min_budget,
        budget_scale,
    )
    union_indices = torch.unique(
        torch.cat(list(keep_indices_by_layer.values()), dim=0),
        sorted=True,
    ).to(dtype=torch.long)
    keep_offsets_by_layer = {
        layer_id: torch.searchsorted(union_indices, keep_indices)
        for layer_id, keep_indices in keep_indices_by_layer.items()
    }
    return DynamicPromptKVCache(
        prompt_length=prompt_length,
        budget=budget,
        teacher_scores=scores,
        layer_count=len(blocks),
        selection_mode=selection_mode,
        union_indices=union_indices,
        keep_indices_by_layer=keep_indices_by_layer,
        keep_offsets_by_layer=keep_offsets_by_layer,
    )


def build_keep_indices_by_layer(
    scores: torch.Tensor,
    prompt_length: int,
    budget: int,
    layer_count: int,
    selection_mode: str,
    budget_mode: str = "fixed",
    min_budget: int = 1,
    budget_scale: float = 1.0,
) -> dict[int, torch.Tensor]:
    if selection_mode == "layer_union":
        return {
            layer_id: topk_prompt_indices(scores[layer_id], prompt_length, budget, budget_mode, min_budget, budget_scale)
            for layer_id in range(layer_count)
        }
    if selection_mode == "global":
        keep = topk_prompt_indices(scores.mean(dim=0), prompt_length, budget, budget_mode, min_budget, budget_scale)
        return {layer_id: keep for layer_id in range(layer_count)}
    raise RuntimeError(f"unsupported prompt selection mode: {selection_mode}")


def topk_prompt_indices(
    scores: torch.Tensor,
    prompt_length: int,
    budget: int,
    budget_mode: str = "fixed",
    min_budget: int = 1,
    budget_scale: float = 1.0,
) -> torch.Tensor:
    if scores.numel() != prompt_length:
        raise RuntimeError("teacher score width does not match prompt length")
    keep_count = resolve_keep_count(scores, prompt_length, budget, budget_mode, min_budget, budget_scale)
    return torch.topk(scores, k=keep_count, largest=True).indices.sort().values.to(dtype=torch.long)


def resolve_keep_count(
    scores: torch.Tensor,
    prompt_length: int,
    budget: int,
    budget_mode: str,
    min_budget: int,
    budget_scale: float,
) -> int:
    max_budget = max(1, min(int(budget), prompt_length))
    if budget_mode == "fixed":
        return max_budget
    if budget_mode == "predicted_mass":
        if budget_scale <= 0.0:
            raise RuntimeError("budget_scale must be positive for predicted_mass budget mode")
        expected_size = float(scores.float().clamp(0.0, 1.0).sum().item()) * float(budget_scale)
        adaptive = int(math.ceil(expected_size))
        return max(1, min(max_budget, max(int(min_budget), adaptive)))
    raise RuntimeError(f"unsupported prompt budget mode: {budget_mode}")


@torch.inference_mode()
def generate_with_dynamic_prompt_kv(
    input_ids: torch.Tensor,
    model: nn.Module,
    prompt_cache: DynamicPromptKVCache,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    refresh_interval: int = 1,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    if cfg_scale > 0.0:
        raise RuntimeError("dynamic prompt KV generation does not support cfg_scale")
    if refresh_interval <= 0:
        raise RuntimeError("refresh_interval must be positive")
    batch_size, prompt_length = input_ids.shape
    if prompt_length != prompt_cache.prompt_length:
        raise RuntimeError("input prompt length does not match dynamic prompt cache")
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    x[:, :prompt_length] = input_ids
    if gen_length % block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    global_step = 0
    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            suffix_ids = x[:, prompt_length:]
            if should_refresh_prompt_cache(prompt_cache, global_step, refresh_interval):
                logits = dynamic_prompt_suffix_logits(model, input_ids, suffix_ids, prompt_cache)
            else:
                logits = cached_prompt_suffix_logits(model, suffix_ids, prompt_cache)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise RuntimeError(f"unsupported remasking: {remasking}")
            x0_p[:, (num_block + 1) * block_length :] = -float("inf")
            x0 = torch.where(mask_index[:, prompt_length:], x0, x[:, prompt_length:])
            confidence = torch.where(mask_index[:, prompt_length:], x0_p, -float("inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                count = int(num_transfer_tokens[batch_idx, step_idx].item())
                if count <= 0:
                    continue
                select_index = torch.topk(confidence[batch_idx], k=count).indices
                transfer_index[batch_idx, select_index] = True
            x[:, prompt_length:][transfer_index] = x0[transfer_index]
            global_step += 1
    return x[:, prompt_length:]


def should_refresh_prompt_cache(
    prompt_cache: DynamicPromptKVCache,
    global_step: int,
    refresh_interval: int,
) -> bool:
    return not prompt_cache.has_all_layers() or global_step % refresh_interval == 0


def dynamic_prompt_suffix_logits(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    prompt_cache: DynamicPromptKVCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("dynamic prompt KV forward supports block_group_size=1 only")
    prompt_indices = prompt_cache.selected_prompt_indices(prompt_ids.device)
    reduced_prompt_ids = prompt_ids.index_select(dim=1, index=prompt_indices)
    x_prompt = decoder.transformer.wte(reduced_prompt_ids)
    x_suffix = decoder.transformer.wte(suffix_ids)
    x = torch.cat([x_prompt, x_suffix], dim=1)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    position_ids = prompt_cache.sequence_positions(suffix_ids.shape[1], x.device)
    for block in decoder.transformer.blocks:
        x = run_dynamic_reduced_block(block, x, position_ids, prompt_cache)
    suffix_start = prompt_cache.reduced_prompt_length
    return suffix_logits_from_hidden(decoder, x[:, suffix_start:, :])


def cached_prompt_suffix_logits(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    prompt_cache: DynamicPromptKVCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("dynamic prompt KV forward supports block_group_size=1 only")
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    position_ids = torch.arange(
        prompt_cache.prompt_length,
        prompt_cache.prompt_length + suffix_ids.shape[1],
        device=x.device,
        dtype=torch.long,
    )
    for block in decoder.transformer.blocks:
        x = run_cached_suffix_block(block, x, position_ids, prompt_cache)
    return suffix_logits_from_hidden(decoder, x)


def suffix_logits_from_hidden(decoder: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    config = getattr(decoder, "config")
    x = decoder.transformer.ln_f(hidden_states)
    if bool(config.weight_tying):
        logits = F.linear(x, decoder.transformer.wte.weight, None)
    else:
        logits = decoder.transformer.ff_out(x)
    if bool(config.scale_logits):
        logits.mul_(1 / math.sqrt(float(config.d_model)))
    return logits.float()


def run_dynamic_reduced_block(
    block: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_cache: DynamicPromptKVCache,
) -> torch.Tensor:
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, position_ids)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    layer_id = int(getattr(block, "layer_id"))
    prompt_offsets = prompt_cache.layer_prompt_offsets(layer_id, x.device)
    prompt_k = k_heads.index_select(dim=2, index=prompt_offsets)
    prompt_v = v_heads.index_select(dim=2, index=prompt_offsets)
    suffix_start = prompt_cache.reduced_prompt_length
    key = torch.cat([prompt_k, k_heads[:, :, suffix_start:, :]], dim=2)
    value = torch.cat([prompt_v, v_heads[:, :, suffix_start:, :]], dim=2)
    att = F.scaled_dot_product_attention(q_heads, key, value, dropout_p=0.0, is_causal=False)
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    prompt_cache.add_layer(layer_id, prompt_k, prompt_v)
    x = x + block.dropout(block.attn_out(att))
    return run_block_mlp(block, x)


def run_cached_suffix_block(
    block: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_cache: DynamicPromptKVCache,
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
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_block_mlp(block, x)


def project_qkv(block: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_normed = block.attn_norm(x)
    if hasattr(block, "att_proj"):
        return block.att_proj(x_normed).split(block.fused_dims, dim=-1)
    return block.q_proj(x_normed), block.k_proj(x_normed), block.v_proj(x_normed)


def project_heads_at_positions(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, q_len, channels = q.size()
    _, k_len, _ = k.size()
    if q_len != k_len or q_len != int(position_ids.numel()):
        raise RuntimeError("position ids must match Q/K sequence length")
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
        q_heads, k_heads = apply_rope_at_positions(block, q_heads, k_heads, position_ids)
    return q_heads, k_heads, v_heads


def apply_rope_at_positions(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = getattr(block, "config")
    q_work = q.float() if bool(config.rope_full_precision) else q
    k_work = k.float() if bool(config.rope_full_precision) else k
    position_ids = position_ids.to(device=q.device, dtype=torch.long)
    total_len = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
    with torch.autocast(q.device.type, enabled=False):
        pos_sin, pos_cos = block.rotary_emb.get_rotary_embedding(total_len, q_work.device)
        pos_sin = pos_sin.type_as(q_work).index_select(dim=2, index=position_ids)
        pos_cos = pos_cos.type_as(q_work).index_select(dim=2, index=position_ids)
        q_work = block.rotary_emb.apply_rotary_pos_emb(pos_sin, pos_cos, q_work)
        k_work = block.rotary_emb.apply_rotary_pos_emb(pos_sin, pos_cos, k_work)
    return q_work.type_as(q), k_work.type_as(k)


def repeat_heads(states: torch.Tensor, head_count: int) -> torch.Tensor:
    state_heads = states.shape[1]
    if head_count % state_heads != 0:
        raise RuntimeError("query head count must be divisible by state head count")
    return states.repeat_interleave(head_count // state_heads, dim=1)


def run_block_mlp(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = block.ff_norm(x)
    if hasattr(block, "up_proj"):
        x = block.act(block.ff_proj(x)) * block.up_proj(x)
    else:
        x = block.act(block.ff_proj(x))
    x = block.ff_out(x)
    return residual + block.dropout(x)
