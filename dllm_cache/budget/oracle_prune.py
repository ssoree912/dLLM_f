from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks


@dataclass(slots=True)
class OraclePruneController:
    prompt_length: int
    budget: int
    teacher_scores: torch.Tensor
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)
    pruned_attention_calls: int = 0

    def restore(self) -> None:
        for module, original in self.originals:
            module.attention = original
            if hasattr(module, "_old_oracle_attention"):
                delattr(module, "_old_oracle_attention")


def install_oracle_pruner(
    model: NamedModuleModel,
    prompt_length: int,
    budget: int,
    teacher_scores: torch.Tensor,
) -> OraclePruneController:
    blocks = find_transformer_blocks(model)
    controller = OraclePruneController(
        prompt_length=prompt_length,
        budget=budget,
        teacher_scores=teacher_scores.float().cpu(),
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        setattr(block, "_old_oracle_attention", original)
        block.attention = types.MethodType(
            make_oracle_attention(controller, original, accepts_block_mask),
            block,
        )
        controller.originals.append((block, original))
    return controller


def make_oracle_attention(
    controller: OraclePruneController,
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
        if layer_past is not None or use_cache or block_mask is not None:
            return call_original(
                original,
                accepts_block_mask,
                q,
                k,
                v,
                attention_bias,
                layer_past,
                use_cache,
                block_mask,
            )
        output = oracle_pruned_attention(block_self, q, k, v, attention_bias, controller)
        controller.pruned_attention_calls += 1
        return output, None

    return wrapped_attention


def call_original(
    original: types.MethodType,
    accepts_block_mask: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: torch.Tensor | None,
    layer_past: tuple[torch.Tensor, torch.Tensor] | None,
    use_cache: bool,
    block_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
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


def oracle_pruned_attention(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: torch.Tensor | None,
    controller: OraclePruneController,
) -> torch.Tensor:
    batch_size, q_len, channels = q.size()
    prompt_length = controller.prompt_length
    layer_id = int(getattr(block, "layer_id"))
    if q_len <= prompt_length or controller.budget >= prompt_length:
        return call_full_attention(block, q, k, v)
    q_heads, k_heads, v_heads = project_qkv_heads(block, q, k, v)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])
    prompt_att = F.scaled_dot_product_attention(
        q_heads[:, :, :prompt_length, :],
        k_heads,
        v_heads,
        attn_mask=slice_query_attention_bias(attention_bias, 0, prompt_length),
        dropout_p=0.0,
        is_causal=False,
    )
    keep = topk_prompt_indices(controller, layer_id, q.device)
    prompt_k = k_heads.index_select(dim=2, index=keep)
    prompt_v = v_heads.index_select(dim=2, index=keep)
    key = torch.cat([prompt_k, k_heads[:, :, prompt_length:, :]], dim=2)
    value = torch.cat([prompt_v, v_heads[:, :, prompt_length:, :]], dim=2)
    suffix_attention_bias = prune_attention_bias(
        slice_query_attention_bias(attention_bias, prompt_length, q_len),
        keep,
        prompt_length,
    )
    suffix_att = F.scaled_dot_product_attention(
        q_heads[:, :, prompt_length:, :],
        key,
        value,
        attn_mask=suffix_attention_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    att = torch.cat([prompt_att, suffix_att], dim=2)
    att = att.transpose(1, 2).contiguous().view(batch_size, q_len, channels)
    return block.attn_out(att)


def slice_query_attention_bias(
    attention_bias: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if attention_bias is None:
        return None
    return attention_bias[:, :, start:end, :]


def prune_attention_bias(
    attention_bias: torch.Tensor | None,
    keep: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor | None:
    if attention_bias is None:
        return None
    prompt_bias = attention_bias[:, :, :, :prompt_length].index_select(dim=-1, index=keep)
    suffix_bias = attention_bias[:, :, :, prompt_length:]
    return torch.cat([prompt_bias, suffix_bias], dim=-1)


def call_full_attention(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    att, _present = block._old_oracle_attention(q, k, v, None, layer_past=None, use_cache=False)
    return att


def project_qkv_heads(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
        q_heads, k_heads = block.rotary_emb(q_heads, k_heads)
    return q_heads, k_heads, v_heads


def repeat_heads(states: torch.Tensor, head_count: int) -> torch.Tensor:
    state_heads = states.shape[1]
    if head_count % state_heads != 0:
        raise RuntimeError("query head count must be divisible by state head count")
    return states.repeat_interleave(head_count // state_heads, dim=1)


def topk_prompt_indices(
    controller: OraclePruneController,
    layer_id: int,
    device: torch.device,
) -> torch.Tensor:
    scores = controller.teacher_scores[layer_id]
    if scores.numel() != controller.prompt_length:
        raise RuntimeError("teacher score width does not match prompt length")
    budget = max(1, min(controller.budget, controller.prompt_length))
    return torch.topk(scores.to(device), k=budget, largest=True).indices
