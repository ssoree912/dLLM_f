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


def build_rotation_schedule(
    weights: torch.Tensor,
    refresh_tokens: int,
    steps: int,
) -> list[torch.Tensor]:
    """Spread refreshes over time so no position stays stale for long.

    A fixed refresh set leaves the rest frozen for the whole trajectory, which is
    the same failure mode as never refreshing at all -- just applied to fewer
    positions. Rotating instead bounds every position's staleness while spending
    the identical budget.

    Weights bias the rotation: a position whose value vector drifts fast comes
    round more often than one that barely moves. Uniform weights degenerate to
    plain round-robin, where each position is refreshed every
    `len(weights) / refresh_tokens` steps.

    Deficit round-robin keeps the per-step count exactly `refresh_tokens`.
    """
    count = int(weights.numel())
    if not 0 < refresh_tokens <= count:
        raise RuntimeError("refresh_tokens must fall inside the kept prompt")
    positive = weights.clamp_min(0.0).double()
    if float(positive.sum()) <= 0.0:
        positive = torch.ones(count, dtype=torch.float64)
    # Per-step share of a refresh slot, summing to refresh_tokens.
    rate = positive / positive.sum() * float(refresh_tokens)
    rate = rate.clamp(max=1.0)
    if float(rate.sum()) > 0.0:
        rate = rate / rate.sum() * float(refresh_tokens)
        rate = rate.clamp(max=1.0)
    credit = torch.zeros(count, dtype=torch.float64)
    schedule: list[torch.Tensor] = []
    for _ in range(steps):
        credit += rate
        chosen = torch.topk(credit, k=refresh_tokens, largest=True).indices
        credit[chosen] -= 1.0
        schedule.append(chosen.sort().values.to(torch.long))
    return schedule


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
    # Rotation mode: every kept position holds a slot in a mutable deep-layer cache
    # that refreshed positions write back into.
    schedule: list[torch.Tensor] = field(default_factory=list)
    deep_kv: dict[int, LayerPromptKV] = field(default_factory=dict)
    split_hidden_all: torch.Tensor | None = None
    # Measured mode: per deep layer, the prompt's block input and its last attention
    # output, so movement can be observed rather than predicted.
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    attn: dict[int, torch.Tensor] = field(default_factory=dict)
    measured_tokens: int = 0

    @property
    def measures(self) -> bool:
        return self.measured_tokens > 0

    @property
    def rotates(self) -> bool:
        return bool(self.schedule)

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
    refresh_scores: torch.Tensor | None = None,
    rotate_steps: int = 0,
    measured_tokens: int = 0,
) -> LayerSplitPromptCache:
    """Prefill the whole prompt once, then keep only what the split needs.

    `refresh_tokens` narrows the deep layers further: only that many of the kept
    positions are re-forwarded there, and the rest serve their prefill K/V. Zero
    means every kept position is refreshed.

    The two decisions want different signals. Which positions to keep is a question
    about importance; which of them to keep refreshing is a question about drift,
    and on SAMSum the two rank positions almost independently (Spearman 0.096).
    Pass `refresh_scores` to rank the refresh set separately from the keep set.
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
    measuring = 0 < measured_tokens < keep_count
    rotating = (
        not measuring and rotate_steps > 0 and 0 < refresh_tokens < keep_count
    )
    if measuring or refresh_tokens <= 0 or refresh_tokens >= keep_count:
        refresh = keep
        stale = torch.empty(0, dtype=torch.long)
    elif rotating:
        # Every kept position stays in play; the schedule decides when each is due.
        refresh = keep
        stale = torch.empty(0, dtype=torch.long)
    else:
        if refresh_scores is None:
            refresh_rank = pooled
        else:
            refresh_rank = refresh_scores.detach().float().cpu()
            if refresh_rank.shape != (len(blocks), prompt_length):
                raise RuntimeError("refresh scores must have shape [layer, prompt]")
            refresh_rank = refresh_rank.mean(dim=0)
        ranked = torch.topk(refresh_rank[keep], k=refresh_tokens, largest=True).indices
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
    if measuring:
        cache.measured_tokens = measured_tokens
    if rotating:
        if refresh_scores is None:
            rotation_weight = torch.ones(keep_count)
        else:
            weights = refresh_scores.detach().float().cpu()
            if weights.shape != (len(blocks), prompt_length):
                raise RuntimeError("refresh scores must have shape [layer, prompt]")
            rotation_weight = weights.mean(dim=0)[keep]
        cache.schedule = build_rotation_schedule(
            rotation_weight, refresh_tokens, rotate_steps
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
            if rotating:
                cache.split_hidden_all = x.index_select(
                    dim=1, index=keep_device
                ).detach()
        q, k, v = project_qkv(block, x)
        q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, positions)
        if q_heads.shape[1] != k_heads.shape[1]:
            k_heads = repeat_heads(k_heads, q_heads.shape[1])
            v_heads = repeat_heads(v_heads, q_heads.shape[1])
        if measuring and layer_id >= frozen_layers:
            cache.hidden[layer_id] = x.index_select(
                dim=1, index=keep_device
            ).detach().clone()
        if layer_id < frozen_layers:
            cache.frozen_kv[layer_id] = LayerPromptKV(
                key=k_heads.index_select(dim=2, index=keep_device).detach(),
                value=v_heads.index_select(dim=2, index=keep_device).detach(),
            )
        elif rotating:
            cache.deep_kv[layer_id] = LayerPromptKV(
                key=k_heads.index_select(dim=2, index=keep_device).detach().clone(),
                value=v_heads.index_select(dim=2, index=keep_device).detach().clone(),
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
        if measuring and layer_id >= frozen_layers:
            cache.attn[layer_id] = att.index_select(
                dim=1, index=keep_device
            ).detach().clone()
        x = x + block.dropout(block.attn_out(att))
        x = run_block_mlp(block, x)

    if frozen_layers == len(blocks):
        cache.split_hidden = x.index_select(dim=1, index=refresh_device).detach()
        if rotating:
            cache.split_hidden_all = x.index_select(dim=1, index=keep_device).detach()
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


def rotating_suffix_logits(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    cache: LayerSplitPromptCache,
    step: int,
) -> torch.Tensor:
    """Refresh this step's slice of the prompt and write it back into the cache."""
    decoder = getattr(model, "model")
    config = getattr(decoder, "config")
    due = cache.schedule[step % len(cache.schedule)].to(suffix_ids.device)
    keep = cache.keep_indices.to(suffix_ids.device)

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
    due_positions = torch.cat([keep.index_select(0, due), suffix_positions])
    due_count = int(due.numel())

    for layer_id, block in enumerate(decoder.transformer.blocks):
        if layer_id == cache.frozen_layers:
            split = cache.split_hidden_all
            if split is None:
                raise RuntimeError("rotation needs the full split hidden state")
            head = split.index_select(1, due).to(device=x.device, dtype=x.dtype)
            x = torch.cat([head, x], dim=1)
        if layer_id < cache.frozen_layers:
            x = _run_frozen_block(block, x, suffix_positions, cache)
        else:
            x = _run_rotating_block(block, x, due_positions, due, cache)
    return suffix_logits_from_hidden(decoder, x[:, due_count:, :])


def _run_rotating_block(
    block: nn.Module,
    x: torch.Tensor,
    due_positions: torch.Tensor,
    due: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    layer_id = int(block.layer_id)
    cached = cache.deep_kv.get(layer_id)
    if cached is None:
        raise RuntimeError(f"rotation cache missing deep layer {layer_id}")
    q, k, v = project_qkv(block, x)
    q_heads, k_heads, v_heads = project_heads_at_positions(block, q, k, v, due_positions)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    due_count = int(due.numel())
    # Write this step's recomputed prompt entries back, so a position stays fresh
    # until its next turn rather than reverting to the prefill value.
    slots = due.view(1, 1, -1, 1).expand(
        cached.key.shape[0], cached.key.shape[1], -1, cached.key.shape[3]
    )
    cached.key.scatter_(2, slots, k_heads[:, :, :due_count, :].to(cached.key.dtype))
    cached.value.scatter_(2, slots, v_heads[:, :, :due_count, :].to(cached.value.dtype))
    key = torch.cat(
        [cached.key.to(k_heads.dtype), k_heads[:, :, due_count:, :]], dim=2
    )
    value = torch.cat(
        [cached.value.to(v_heads.dtype), v_heads[:, :, due_count:, :]], dim=2
    )
    att = F.scaled_dot_product_attention(
        q_heads, key, value, dropout_p=0.0, is_causal=False
    )
    att = att.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], x.shape[2])
    x = x + block.dropout(block.attn_out(att))
    return run_block_mlp(block, x)


def measured_suffix_logits(
    model: nn.Module,
    suffix_ids: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    """Pick the refresh set from measured change instead of a predicted schedule.

    dLLM-Cache ranks generation positions by how far their value vector moved from
    the cached one. That test does not carry over to the prompt: prompt token ids
    never change, so a position whose hidden state is served from cache reprojects
    to exactly its cached value and would report zero movement forever.

    What does move a prompt position is its attention to the suffix, so that is
    what we compare. Query, key and value are reprojected for every kept position
    each step -- which also keeps the keys the suffix reads as fresh as the cached
    hidden states allow -- and only the positions whose attention output moved most
    pay for `attn_out` and the MLP.
    """
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
    prompt_positions = cache.keep_indices.to(x.device)

    for layer_id, block in enumerate(decoder.transformer.blocks):
        if layer_id < cache.frozen_layers:
            x = _run_frozen_block(block, x, suffix_positions, cache)
        else:
            x = _run_measured_block(
                block, x, prompt_positions, suffix_positions, cache
            )
    return suffix_logits_from_hidden(decoder, x)


def _run_measured_block(
    block: nn.Module,
    x_suffix: torch.Tensor,
    prompt_positions: torch.Tensor,
    suffix_positions: torch.Tensor,
    cache: LayerSplitPromptCache,
) -> torch.Tensor:
    layer_id = int(block.layer_id)
    hidden = cache.hidden[layer_id]
    prompt_hidden = hidden.to(device=x_suffix.device, dtype=x_suffix.dtype)

    q_p, k_p, v_p = project_qkv(block, prompt_hidden)
    q_ph, k_ph, v_ph = project_heads_at_positions(block, q_p, k_p, v_p, prompt_positions)
    q_s, k_s, v_s = project_qkv(block, x_suffix)
    q_sh, k_sh, v_sh = project_heads_at_positions(block, q_s, k_s, v_s, suffix_positions)
    if q_ph.shape[1] != k_ph.shape[1]:
        k_ph = repeat_heads(k_ph, q_ph.shape[1])
        v_ph = repeat_heads(v_ph, q_ph.shape[1])
        k_sh = repeat_heads(k_sh, q_sh.shape[1])
        v_sh = repeat_heads(v_sh, q_sh.shape[1])

    key = torch.cat([k_ph, k_sh], dim=2)
    value = torch.cat([v_ph, v_sh], dim=2)
    att_p = F.scaled_dot_product_attention(q_ph, key, value, dropout_p=0.0, is_causal=False)
    att_s = F.scaled_dot_product_attention(q_sh, key, value, dropout_p=0.0, is_causal=False)
    att_p = att_p.transpose(1, 2).contiguous().view_as(prompt_hidden)
    att_s = att_s.transpose(1, 2).contiguous().view_as(x_suffix)

    # Movement since this position was last refreshed, not since the last step.
    # What the cache gets wrong is the hidden state it is still serving, and that
    # error is whatever has accumulated since it was last written. Drift saturates
    # -- 2-3% per step but only 0.13-0.37 in total -- so a per-step delta looks
    # nearly uniform across positions and would never surface a long-stale one.
    previous = cache.attn[layer_id].to(device=att_p.device, dtype=att_p.dtype)
    moved = 1.0 - F.cosine_similarity(att_p.float(), previous.float(), dim=-1)
    due = torch.topk(moved.squeeze(0), k=cache.measured_tokens, largest=True).indices

    slots = due.view(1, -1, 1).expand(1, -1, prompt_hidden.shape[-1])
    cache.attn[layer_id].scatter_(
        1, slots, torch.gather(att_p, 1, slots).detach().to(cache.attn[layer_id].dtype)
    )
    selected_hidden = torch.gather(prompt_hidden, 1, slots)
    selected_att = torch.gather(att_p, 1, slots)
    updated = selected_hidden + block.dropout(block.attn_out(selected_att))
    updated = run_block_mlp(block, updated)

    following = cache.hidden.get(layer_id + 1)
    if following is None:
        cache.hidden[layer_id + 1] = prompt_hidden.detach().clone()
        following = cache.hidden[layer_id + 1]
    following.scatter_(1, slots, updated.detach().to(following.dtype))

    x_suffix = x_suffix + block.dropout(block.attn_out(att_s))
    return run_block_mlp(block, x_suffix)


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

    global_step = 0
    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for step_idx in range(steps_per_block):
            mask_index = x == mask_id
            suffix_ids = x[:, prompt_length:]
            if prompt_cache.measures:
                logits = measured_suffix_logits(model, suffix_ids, prompt_cache)
            elif prompt_cache.rotates:
                logits = rotating_suffix_logits(
                    model, suffix_ids, prompt_cache, global_step
                )
            else:
                logits = layer_split_suffix_logits(model, suffix_ids, prompt_cache)
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


__all__ = [
    "LayerSplitPromptCache",
    "build_layer_split_prompt_cache",
    "generate_with_layer_split_prompt_kv",
    "build_rotation_schedule",
    "layer_split_suffix_logits",
    "measured_suffix_logits",
    "rotating_suffix_logits",
]
