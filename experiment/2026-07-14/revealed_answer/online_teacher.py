from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from revealed_answer.attention_teacher import find_transformer_blocks
from revealed_answer.full_prompt_kv_cache import build_full_prompt_kv_cache
from revealed_answer.prompt_kv_cache import PromptKVCache, project_heads, repeat_heads
from revealed_answer.prompt_kv_forward import run_suffix_mlp
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


MASK_ID = 126336


@dataclass(frozen=True, slots=True)
class OnlineTeacherConfig:
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    confidence_weight: bool
    mask_id: int = MASK_ID


@dataclass(frozen=True, slots=True)
class OnlineTeacherResult:
    generated_ids: torch.Tensor
    teacher_raw: torch.Tensor
    teacher_norm: torch.Tensor
    commit_count: int
    confidence_weight_sum: float


@dataclass(slots=True)
class OnlineTeacherTrace:
    raw: torch.Tensor
    confidence_weight: bool
    commit_count: int = 0
    confidence_weight_sum: float = 0.0


@torch.inference_mode()
def generate_with_online_teacher(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    config: OnlineTeacherConfig,
) -> OnlineTeacherResult:
    if config.gen_length % config.block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    if config.steps % (config.gen_length // config.block_length) != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    prompt_cache = build_full_prompt_kv_cache(model, prompt_ids)
    blocks = find_transformer_blocks(model)
    suffix_ids = torch.full(
        (prompt_ids.shape[0], config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    trace = OnlineTeacherTrace(
        raw=torch.zeros((len(blocks), prompt_cache.prompt_length), device=prompt_ids.device),
        confidence_weight=config.confidence_weight,
    )
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    for block_id in range(num_blocks):
        start = block_id * config.block_length
        end = (block_id + 1) * config.block_length
        block_mask_index = suffix_ids[:, start:end] == config.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_id in range(steps_per_block):
            mask_index = suffix_ids == config.mask_id
            logits, prompt_attention = suffix_logits_and_prompt_attention(
                model,
                suffix_ids,
                prompt_cache,
            )
            x0, confidence = select_candidates(logits, suffix_ids, mask_index, config)
            confidence[:, end:] = -torch.inf
            transfer_index = build_transfer_index(confidence, num_transfer_tokens[:, step_id])
            accumulate_commits(trace, prompt_attention, transfer_index, confidence)
            suffix_ids[transfer_index] = x0[transfer_index]
    teacher_raw = trace.raw.detach().cpu().float()
    teacher_norm = teacher_raw / teacher_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return OnlineTeacherResult(
        generated_ids=suffix_ids.detach().cpu().squeeze(0),
        teacher_raw=teacher_raw,
        teacher_norm=teacher_norm,
        commit_count=trace.commit_count,
        confidence_weight_sum=trace.confidence_weight_sum,
    )


def suffix_logits_and_prompt_attention(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    prompt_cache: PromptKVCache,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("online teacher supports block_group_size=1 only")
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    prompt_attention: dict[int, torch.Tensor] = {}
    for block in decoder.transformer.blocks:
        x, layer_attention = run_suffix_block_with_prompt_attention(block, x, prompt_cache)
        prompt_attention[int(getattr(block, "layer_id"))] = layer_attention
    x = decoder.transformer.ln_f(x)
    if bool(config.weight_tying):
        logits = F.linear(x, decoder.transformer.wte.weight, None)
    else:
        logits = decoder.transformer.ff_out(x)
    if bool(config.scale_logits):
        logits.mul_(1 / math.sqrt(float(config.d_model)))
    return logits.float(), prompt_attention


def run_suffix_block_with_prompt_attention(
    block: nn.Module,
    x: torch.Tensor,
    prompt_cache: PromptKVCache,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    scores = torch.matmul(q_heads.float(), key.float().transpose(-1, -2))
    scores = scores / math.sqrt(float(q_heads.shape[-1]))
    attention = torch.softmax(scores, dim=-1)
    prompt_attention = attention[:, :, :, : prompt_cache.prompt_length].mean(dim=1).squeeze(0)
    att = torch.matmul(attention.to(dtype=value.dtype), value)
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_suffix_mlp(block, x), prompt_attention


def select_candidates(
    logits: torch.Tensor,
    suffix_ids: torch.Tensor,
    mask_index: torch.Tensor,
    config: OnlineTeacherConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    probs = F.softmax(logits, dim=-1)
    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, suffix_ids)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -torch.inf))
    return x0, confidence


def build_transfer_index(
    confidence: torch.Tensor,
    transfer_counts: torch.Tensor,
) -> torch.Tensor:
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool, device=confidence.device)
    for batch_id in range(confidence.shape[0]):
        count = int(transfer_counts[batch_id].item())
        if count <= 0:
            continue
        select_index = torch.topk(confidence[batch_id], k=count).indices
        transfer_index[batch_id, select_index] = True
    return transfer_index


def accumulate_commits(
    trace: OnlineTeacherTrace,
    prompt_attention: dict[int, torch.Tensor],
    transfer_index: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    if transfer_index.shape[0] != 1:
        raise RuntimeError("online teacher extraction expects batch size 1")
    selected = transfer_index[0].nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return
    weights = commit_weights(trace, confidence[0].index_select(0, selected))
    for layer_id, layer_attention in prompt_attention.items():
        rows = layer_attention.index_select(0, selected)
        trace.raw[layer_id] += (rows * weights.unsqueeze(-1)).sum(dim=0)
    trace.commit_count += int(selected.numel())
    trace.confidence_weight_sum += float(weights.sum().detach().cpu())


def commit_weights(trace: OnlineTeacherTrace, confidence: torch.Tensor) -> torch.Tensor:
    if trace.confidence_weight:
        return confidence.clamp_min(0.0).float()
    return torch.ones_like(confidence, dtype=torch.float32)
