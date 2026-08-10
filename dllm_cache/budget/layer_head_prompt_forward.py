"""Suffix (response-region) forward pass and generation loop for the layer+head
prompt-KV cache -- same structure as prompt_kv_forward.py/prompt_kv_generate.py,
extended to fold each layer's per-head keep_bias into the SDPA attn_mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.layer_head_prompt_kv import LayerHeadPromptCache
from dllm_cache.budget.prompt_kv_cache import project_heads, repeat_heads
from dllm_cache.budget.prompt_kv_forward import run_suffix_mlp
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


def layer_head_prompt_kv_suffix_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_cache: LayerHeadPromptCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("layer-head prompt suffix forward supports block_group_size=1 only")
    suffix_ids = input_ids[:, prompt_cache.prompt_length :]
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    for block in decoder.transformer.blocks:
        x = run_layer_head_suffix_block(block, x, prompt_cache)
    x = decoder.transformer.ln_f(x)
    if bool(config.weight_tying):
        logits = F.linear(x, decoder.transformer.wte.weight, None)
    else:
        logits = decoder.transformer.ff_out(x)
    if bool(config.scale_logits):
        logits.mul_(1 / math.sqrt(float(config.d_model)))
    return logits.float()


def run_layer_head_suffix_block(
    block: nn.Module,
    x: torch.Tensor,
    prompt_cache: LayerHeadPromptCache,
) -> torch.Tensor:
    x_normed = block.attn_norm(x)
    if hasattr(block, "att_proj"):
        q, k, v = block.att_proj(x_normed).split(block.fused_dims, dim=-1)
    else:
        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)
    q_heads, k_heads, v_heads = project_heads(
        block, q, k, v, position_offset=prompt_cache.prompt_length
    )
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    layer_cache = prompt_cache.layer(int(getattr(block, "layer_id")), x.device, k_heads.dtype)
    key = torch.cat([layer_cache.key, k_heads], dim=2)
    value = torch.cat([layer_cache.value, v_heads], dim=2)

    head_count = key.shape[1]
    suffix_len = k_heads.shape[2]
    suffix_bias = torch.zeros((head_count, suffix_len), dtype=torch.float32, device=x.device)
    attn_bias = torch.cat([layer_cache.keep_bias, suffix_bias], dim=-1)
    attn_bias = attn_bias.to(q_heads.dtype).view(1, head_count, 1, key.shape[2])

    att = F.scaled_dot_product_attention(
        q_heads, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
    )
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_suffix_mlp(block, x)


@torch.inference_mode()
def generate_with_layer_head_prompt_kv(
    input_ids: torch.Tensor,
    model: nn.Module,
    prompt_cache: LayerHeadPromptCache,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    if cfg_scale > 0.0:
        raise RuntimeError("layer-head prompt KV cache generation does not support cfg_scale")
    batch_size, prompt_length = input_ids.shape
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

    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            logits = layer_head_prompt_kv_suffix_logits(model, x, prompt_cache)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise RuntimeError(f"unsupported remasking: {remasking}")
            x0_p[:, (num_block + 1) * block_length :] = -float("inf")
            x0 = torch.where(mask_index[:, prompt_length:], x0, x[:, prompt_length:])
            confidence = torch.where(mask_index[:, prompt_length:], x0_p, -float("inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                select_index = torch.topk(
                    confidence[batch_idx], k=num_transfer_tokens[batch_idx, step_idx]
                ).indices
                transfer_index[batch_idx, select_index] = True
            x[:, prompt_length:][transfer_index] = x0[transfer_index]
    return x[:, prompt_length:]
