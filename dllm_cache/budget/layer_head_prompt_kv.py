"""Layer-axis + head-axis prompt-KV cache.

Token *ranking* within a layer is still the student's per-(layer, token) score --
same signal `prompt_kv_cache.py` uses, no per-head ranking is trained or computed.
What differs per head is only the *budget*: `layer_head_budget.head_budgets(...)`
gives each head h in layer l a count k_l_h, derived offline from the calibration
profile (`head_pref_profile.py`) via `layer_head_budget.layer_budgets(...)`.

Because every head in a layer shares one ranking, top-k_l_h for a small-budget head
is always a *subset* of top-k_l_h' for a larger-budget head in the same layer
(nested top-k). So instead of building ragged per-head K/V tensors, we cache the
UNION (top max_h(k_l_h) positions, same shape as the existing single-budget path)
and enforce each head's smaller budget with an additive attention-bias mask that
-inf's out the tail of the union for that head. Attention semantics match a true
per-head KV-truncation exactly; only physical memory isn't reduced below the union
size (see FLOP/memory accounting on the step_distill branch for that follow-up).
"""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from dllm_cache.budget.attention_teacher import NamedModuleModel, find_transformer_blocks
from dllm_cache.budget.layer_head_budget import head_budgets, layer_budgets
from dllm_cache.budget.prompt_kv_cache import project_heads, repeat_heads


@dataclass(frozen=True, slots=True)
class LayerHeadPromptKV:
    key: torch.Tensor
    value: torch.Tensor
    keep_bias: torch.Tensor  # [n_heads, union_size] additive bias: 0 kept, -inf dropped


@dataclass(slots=True)
class LayerHeadPromptCache:
    prompt_length: int
    teacher_scores: torch.Tensor
    layer_budget: list[int]
    head_budgets_by_layer: dict[int, list[int]]
    layer_count: int
    layers: dict[int, LayerHeadPromptKV] = field(default_factory=dict)
    originals: list[tuple[nn.Module, types.MethodType]] = field(default_factory=list)

    def add_layer(self, layer_id: int, key: torch.Tensor, value: torch.Tensor, keep_bias: torch.Tensor) -> None:
        self.layers[layer_id] = LayerHeadPromptKV(key=key.detach(), value=value.detach(), keep_bias=keep_bias.detach())

    def layer(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> LayerHeadPromptKV:
        cached = self.layers.get(layer_id)
        if cached is None:
            raise RuntimeError(f"layer-head prompt cache missing layer {layer_id}")
        return LayerHeadPromptKV(
            key=cached.key.to(device=device, dtype=dtype),
            value=cached.value.to(device=device, dtype=dtype),
            keep_bias=cached.keep_bias.to(device=device, dtype=torch.float32),
        )

    def restore(self) -> None:
        for module, original in self.originals:
            module.attention = original
        self.originals.clear()

    def assert_complete(self) -> None:
        missing = sorted(set(range(self.layer_count)) - set(self.layers))
        if missing:
            raise RuntimeError(f"layer-head prompt cache missing layers: {missing}")


def build_layer_head_prompt_cache(
    model: NamedModuleModel,
    prompt_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
    profile: dict,
    budget_per_head: int,
    boundary_layers: list[int],
    beta: float = 0.4,
    alpha: float = 0.1,
) -> LayerHeadPromptCache:
    """`teacher_scores` is [layer_count, prompt_length] -- the student's existing
    per-token importance, reused unchanged as the within-layer ranking. `profile`
    is a loaded head_pref_profile.py calibration JSON (`layer_importance`,
    `head_preference`). `budget_per_head` is B: the per-layer-per-head average
    keep count (k_p = layer_count * B), matching the paper's 128/256/512 knob --
    NOT the same unit as the flat `student_budget` the single-axis path uses.
    """
    blocks = find_transformer_blocks(model)
    layer_count = len(blocks)
    prompt_length = int(prompt_ids.shape[1])

    importance = profile["layer_importance"]
    preference = profile["head_preference"]
    if len(importance) != layer_count:
        raise RuntimeError(
            f"calibration profile has {len(importance)} layers, model has {layer_count}"
        )
    if len(preference) != layer_count:
        raise RuntimeError(
            f"calibration profile has {len(preference)} head-preference layers, "
            f"model has {layer_count}"
        )

    k_l = layer_budgets(importance, budget_per_head, boundary_layers, beta=beta)
    head_budgets_by_layer = {
        layer_id: head_budgets(k_l[layer_id], preference[layer_id], alpha=alpha)
        for layer_id in range(layer_count)
    }

    cache = LayerHeadPromptCache(
        prompt_length=prompt_length,
        teacher_scores=teacher_scores.float().cpu(),
        layer_budget=k_l,
        head_budgets_by_layer=head_budgets_by_layer,
        layer_count=layer_count,
    )
    for block in blocks:
        original = block.attention
        accepts_block_mask = "block_mask" in inspect.signature(original).parameters
        block.attention = types.MethodType(
            _make_collecting_attention(cache, original, accepts_block_mask), block
        )
        cache.originals.append((block, original))
    try:
        with torch.inference_mode():
            model(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                use_cache=False,
                return_dict=True,
            )
    finally:
        cache.restore()
    cache.assert_complete()
    return cache


def _make_collecting_attention(cache: LayerHeadPromptCache, original, accepts_block_mask: bool):
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
        _capture_layer_head_kv(block_self, q, k, v, cache)
        if accepts_block_mask:
            return original(
                q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, block_mask=block_mask
            )
        return original(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

    return wrapped_attention


def _capture_layer_head_kv(
    block: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: LayerHeadPromptCache,
) -> None:
    if q.shape[1] != cache.prompt_length:
        raise RuntimeError("layer-head prompt precompute must run on prompt-only input")
    layer_id = int(getattr(block, "layer_id"))
    q_heads, k_heads, v_heads = project_heads(block, q, k, v, position_offset=0)
    if q_heads.shape[1] != k_heads.shape[1]:
        k_heads = repeat_heads(k_heads, q_heads.shape[1])
        v_heads = repeat_heads(v_heads, q_heads.shape[1])

    head_count = k_heads.shape[1]
    per_head_budget = cache.head_budgets_by_layer[layer_id]
    if len(per_head_budget) != head_count:
        raise RuntimeError(
            f"layer {layer_id}: calibration profile has {len(per_head_budget)} heads, "
            f"model has {head_count} heads"
        )

    union_size = max(1, min(max(per_head_budget), cache.prompt_length))
    scores = cache.teacher_scores[layer_id]
    if scores.numel() != cache.prompt_length:
        raise RuntimeError("teacher score width does not match prompt length")
    ranked = torch.topk(scores.to(q.device), k=union_size, largest=True).indices

    keep_bias = torch.zeros((head_count, union_size), dtype=torch.float32, device=q.device)
    for head_id, budget in enumerate(per_head_budget):
        budget = max(0, min(budget, union_size))
        if budget < union_size:
            keep_bias[head_id, budget:] = -float("inf")

    cache.add_layer(
        layer_id,
        k_heads.index_select(dim=2, index=ranked),
        v_heads.index_select(dim=2, index=ranked),
        keep_bias,
    )
