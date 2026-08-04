from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import assert_never

from .model_hooks import (
    NamedModuleModel,
    find_transformer_blocks,
    repeat_kv_heads,
)


class ReplayMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True, slots=True)
class ReplayPruningError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class OfflineReplayController:
    """Mutable step cursor plus reversible attention wrappers for one replay."""

    __slots__ = (
        "budget",
        "mode",
        "order",
        "originals",
        "prompt_length",
        "step_id",
    )

    def __init__(
        self,
        prompt_length: int,
        budget: int,
        order: torch.Tensor,
        mode: ReplayMode,
    ) -> None:
        if order.ndim != 3:
            raise ReplayPruningError(
                "replay order must have shape [step, layer, rank]"
            )
        if (
            prompt_length <= 0
            or budget <= 0
            or min(budget, prompt_length) > order.shape[2]
        ):
            raise ReplayPruningError(
                "prompt length and replay budget are incompatible"
            )
        self.prompt_length = prompt_length
        self.budget = min(budget, prompt_length)
        self.order = order.detach().cpu().to(torch.long)
        self.mode = mode
        self.step_id = 0
        self.originals: list[tuple[nn.Module, types.MethodType]] = []

    def set_step(self, step_id: int) -> None:
        if step_id < 0 or step_id >= self.order.shape[0]:
            raise ReplayPruningError(
                f"replay step {step_id} falls outside stored order"
            )
        self.step_id = step_id

    def prompt_indices(self, layer_id: int, device: torch.device) -> torch.Tensor:
        if layer_id < 0 or layer_id >= self.order.shape[1]:
            raise ReplayPruningError(
                f"replay layer {layer_id} falls outside stored order"
            )
        match self.mode:
            case ReplayMode.STATIC:
                order_step = 0
            case ReplayMode.DYNAMIC:
                order_step = self.step_id
            case unreachable:
                assert_never(unreachable)
        return self.order[order_step, layer_id, : self.budget].to(device)

    def restore(self) -> None:
        for module, original in reversed(self.originals):
            setattr(module, "attention", original)
        self.originals.clear()


def install_offline_replay_pruner(
    model: NamedModuleModel,
    controller: OfflineReplayController,
) -> None:
    """Install layer-local prompt pruning wrappers transactionally."""
    blocks = find_transformer_blocks(model)
    if len(blocks) != controller.order.shape[1]:
        raise ReplayPruningError("stored replay layers do not match model layers")
    try:
        for block in blocks:
            original = block.attention
            accepts_block_mask = "block_mask" in inspect.signature(original).parameters
            wrapped = _make_replay_attention(
                controller,
                original,
                accepts_block_mask=accepts_block_mask,
            )
            setattr(block, "attention", types.MethodType(wrapped, block))
            controller.originals.append((block, original))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        controller.restore()
        raise


def _make_replay_attention(
    controller: OfflineReplayController,
    original: types.MethodType,
    *,
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
    ):
        if layer_past is not None or use_cache or block_mask is not None:
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
            return original(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        return (
            oracle_pruned_attention(
                block_self,
                q,
                k,
                v,
                attention_bias,
                controller,
            ),
            None,
        )

    return wrapped_attention


def oracle_pruned_attention(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: torch.Tensor | None,
    controller: OfflineReplayController,
) -> torch.Tensor:
    """Attend from every query to selected prompt keys and every suffix key."""
    batch_size, query_length, channels = q.shape
    prompt_length = controller.prompt_length
    if query_length <= prompt_length:
        raise ReplayPruningError(
            "offline replay requires a prompt-plus-suffix forward"
        )

    key_dtype = k.dtype
    q_norm = getattr(block, "q_norm", None)
    k_norm = getattr(block, "k_norm", None)
    if q_norm is not None and k_norm is not None:
        q = q_norm(q).to(dtype=key_dtype)
        k = k_norm(k).to(dtype=key_dtype)
    config = block.config
    query_heads = int(config.n_heads)
    kv_heads = int(config.effective_n_kv_heads)
    head_dim = channels // query_heads
    q_heads = q.view(batch_size, query_length, query_heads, head_dim).transpose(1, 2)
    k_heads = k.view(batch_size, query_length, kv_heads, head_dim).transpose(1, 2)
    v_heads = v.view(batch_size, query_length, kv_heads, head_dim).transpose(1, 2)
    if bool(config.rope):
        q_heads, k_heads = block.rotary_emb(q_heads, k_heads)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_kv_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_kv_heads(v_heads, q_heads.shape[1])

    keep_prompt = controller.prompt_indices(int(block.layer_id), q.device)
    suffix = torch.arange(prompt_length, query_length, device=q.device)
    keep_keys = torch.cat((keep_prompt, suffix))
    selected_k = k_heads.index_select(2, keep_keys)
    selected_v = v_heads.index_select(2, keep_keys)
    selected_bias = (
        None
        if attention_bias is None
        else attention_bias[..., :query_length, :query_length].index_select(
            -1, keep_keys
        )
    )
    attended = functional.scaled_dot_product_attention(
        q_heads,
        selected_k,
        selected_v,
        attn_mask=selected_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    merged = attended.transpose(1, 2).contiguous().view(
        batch_size, query_length, channels
    )
    return block.attn_out(merged)
