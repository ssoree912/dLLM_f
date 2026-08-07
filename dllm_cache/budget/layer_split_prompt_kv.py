"""Freeze the prompt through the shallow layers and keep refreshing it deeper.

Measured drift on SAMSum (see experiment/report/2026-08-07): a prompt position's
value vector moves 0.00 at layer 0, 0.04 by layer 8, 0.13 by layer 16 and 0.37 by
layer 24. Almost all of the movement that a frozen cache gets wrong happens in the
deep half, so refreshing every layer every step pays full price for a problem that
only exists at the top.

This splits the stack at `frozen_layers`:

  layers <  frozen_layers   prompt is not forwarded at all; its K/V come from the
                            prefill cache, and only the suffix flows through
  layers >= frozen_layers   the reduced prompt is spliced back in at its cached
                            hidden state and forwarded normally, so those layers
                            keep absorbing the suffix

Per-step cost drops from `frozen_layers` worth of full-sequence blocks to the same
count of suffix-only blocks. With 960 kept prompt tokens, a 128-token canvas and a
split at layer 16, that is 44% fewer token-blocks than refreshing everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.dynamic_prompt_kv import (
    project_heads_at_positions,
    project_qkv,
    run_block_mlp,
    suffix_logits_from_hidden,
)
from dllm_cache.budget.prompt_kv_cache import LayerPromptKV, repeat_heads
from utils.generate_function import add_gumbel_noise, get_num_transfer_tokens


@dataclass(slots=True)
class LayerSplitPromptCache:
    prompt_length: int
    budget: int
    frozen_layers: int
    keep_indices: torch.Tensor
    refresh_indices: torch.Tensor
    stale_indices: torch.Tensor
    frozen_kv: dict[int, LayerPromptKV] = field(default_factory=dict)
    stale_kv: dict[int, LayerPromptKV] = field(default_factory=dict)
    split_hidden: torch.Tensor | None = None

    @property
    def reduced_prompt_length(self) -> int:
        return int(self.refresh_indices.numel())

    @property
    def refreshes_every_token(self) -> bool:
        return int(self.stale_indices.numel()) == 0

    def stale_layer(
        self,
        layer_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LayerPromptKV | None:
        cached = self.stale_kv.get(layer_id)
        if cached is None:
            return None
        return LayerPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
        )

    def sequence_positions(
        self,
        suffix_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Original positions of the refreshed prompt tokens, then the suffix."""
        suffix = torch.arange(
            self.prompt_length,
            self.prompt_length + suffix_length,
            device=device,
            dtype=torch.long,
        )
        return torch.cat([self.refresh_indices.to(device), suffix])

    def frozen_layer(
        self,
        layer_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LayerPromptKV:
        cached = self.frozen_kv.get(layer_id)
        if cached is None:
            raise RuntimeError(f"layer-split cache missing frozen layer {layer_id}")
        return LayerPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
        )


@torch.inference_mode()
def build_layer_split_prompt_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
    budget: int,
    teacher_scores: torch.Tensor,
    frozen_layers: int,
    refresh_tokens: int = 0,
) -> LayerSplitPromptCache:
    """Prefill the whole prompt once, then keep only what the split needs.

    `refresh_tokens` narrows the deep layers further: only that many of the kept
    positions are re-forwarded there, and the rest serve their prefill K/V. Zero
    means every kept position is refreshed.
    """
    blocks = find_transformer_blocks(model)
    prompt_length = int(prompt_ids.shape[1])
    if not 0 <= frozen_layers <= len(blocks):
        raise RuntimeError("frozen_layers must fall inside the decoder stack")
    scores = teacher_scores.detach().float().cpu()
    if scores.shape != (len(blocks), prompt_length):
        raise RuntimeError("teacher scores must have shape [layer, prompt]")
    keep_count = max(1, min(budget, prompt_length))
    # One shared token set, so the surviving prompt stays a contiguous sequence.
    pooled = scores.mean(dim=0)
    keep = torch.topk(pooled, k=keep_count, largest=True).indices.sort().values
    if refresh_tokens <= 0 or refresh_tokens >= keep_count:
        refresh = keep
        stale = torch.empty(0, dtype=torch.long)
    else:
        # Attention mass concentrates hard, so the top slice carries most of the
        # error a frozen entry would introduce.
        ranked = torch.topk(pooled[keep], k=refresh_tokens, largest=True).indices
        refresh_mask = torch.zeros(keep_count, dtype=torch.bool)
        refresh_mask[ranked] = True
        refresh = keep[refresh_mask]
        stale = keep[~refresh_mask]
    cache = LayerSplitPromptCache(
        prompt_length=prompt_length,
        budget=keep_count,
        frozen_layers=frozen_layers,
        keep_indices=keep,
        refresh_indices=refresh,
        stale_indices=stale,
    )

    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    if int(config.block_group_size) != 1:
        raise RuntimeError("layer-split prompt cache supports block_group_size=1 only")
    x = decoder.transformer.wte(prompt_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    positions = torch.arange(prompt_length, device=x.device, dtype=torch.long)
    keep_device = keep.to(x.device)
    refresh_device = refresh.to(x.device)
    stale_device = stale.to(x.device)

    for layer_id, block in enumerate(blocks):
        if layer_id == frozen_layers:
            cache.split_hidden = x.index_select(dim=1, index=refresh_device).detach()
        q, k, v = project_qkv(block, x)
        q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, positions)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_heads(k_heads, q_heads.shape[1])
            v_heads = repeat_heads(v_heads, q_heads.shape[1])
        if layer_id < frozen_layers:
            cache.frozen_kv[layer_id] = LayerPromptKV(
                key=k_heads.index_select(dim=2, index=keep_device).detach(),
                value=v_heads.index_select(dim=2, index=keep_device).detach(),
            )
        elif stale_device.numel():
            cache.stale_kv[layer_id] = LayerPromptKV(
                key=k_heads.index_select(dim=2, index=stale_device).detach(),
                value=v_heads.index_select(dim=2, index=stale_device).detach(),
            )
        att = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=False
        )
        att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
        x = x + block.dropout(block.attn_out(att))
        x = run_block_mlp(block, x)

    if frozen_layers == len(blocks):
        cache.split_hidden = x.index_select(dim=1, index=refresh_device).detach()
    if cache.split_hidden is None:
        raise RuntimeError("layer-split cache never captured the split hidden state")
    return cache


def layer_split_suffix_logits(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    x = decoder.transformer.wte(suffix_ids)
    if bool(config.input_emb_norm):
        x = x * (float(config.d_model) ** 0.5)
    x = decoder.transformer.emb_drop(x)
    suffix_length = int(suffix_ids.shape[1])
    suffix_positions = torch.arange(
        cache.prompt_length,
        cache.prompt_length + suffix_length,
        device=x.device,
        dtype=torch.long,
    )
    full_positions = cache.sequence_positions(suffix_length, x.device)
    reduced = cache.reduced_prompt_length

    for layer_id, block in enumerate(decoder.transformer.blocks):
        if layer_id == cache.frozen_layers:
            split = cache.split_hidden
            if split is None:
                raise RuntimeError("layer-split cache is missing its hidden state")
            x = torch.cat([split.to(device=x.device, dtype=x.dtype), x], dim=1)
        if layer_id < cache.frozen_layers:
            x = _run_frozen_block(block, x, suffix_positions, cache)
        else:
            x = _run_refreshed_block(block, x, full_positions, cache)
    return suffix_logits_from_hidden(decoder, x[:, reduced:, :])


def _run_frozen_block(
    block: nn.Module,
    x: torch.Tensor,
    suffix_positions: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    """Only the suffix flows; the prompt contributes its prefill K/V unchanged."""
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(
        block, q, k, v, suffix_positions
    )
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    frozen = cache.frozen_layer(int(block.layer_id), x.device, k_heads.dtype)
    key = torch.cat([frozen.key, k_heads], dim=2)
    value = torch.cat([frozen.value, v_heads], dim=2)
    att = F.scaled_dot_product_attention(
        q_heads, key, value, dropout_p=0.0, is_causal=False
    )
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_block_mlp(block, x)


def _run_refreshed_block(
    block: nn.Module,
    x: torch.Tensor,
    full_positions: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    """Refreshed prompt tokens flow; any held-back ones contribute prefill K/V.

    Attention is order-independent once RoPE has been applied at each token's own
    position, so the held-back keys can simply be concatenated on.
    """
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, full_positions)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    key, value = k_heads, v_heads
    stale = cache.stale_layer(int(block.layer_id), x.device, k_heads.dtype)
    if stale is not None:
        key = torch.cat([k_heads, stale.key], dim=2)
        value = torch.cat([v_heads, stale.value], dim=2)
    att = F.scaled_dot_product_attention(
        q_heads, key, value, dropout_p=0.0, is_causal=False
    )
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_block_mlp(block, x)


@torch.inference_mode()
def generate_with_layer_split_prompt_kv(
    input_ids: torch.Tensor,
    model: nn.Module,
    prompt_cache: LayerSplitPromptCache,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    if cfg_scale > 0.0:
        raise RuntimeError("layer-split generation does not support cfg_scale")
    batch_size, prompt_length = input_ids.shape
    if prompt_length != prompt_cache.prompt_length:
        raise RuntimeError("input prompt length does not match layer-split cache")
    if gen_length % block_length != 0:
        raise RuntimeError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise RuntimeError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    x[:, :prompt_length] = input_ids

    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            logits = layer_split_suffix_logits(
                model, x[:, prompt_length:], prompt_cache
            )
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
    return x[:, prompt_length:]


__all__ = [
    "LayerSplitPromptCache",
    "build_layer_split_prompt_cache",
    "generate_with_layer_split_prompt_kv",
    "layer_split_suffix_logits",
]
