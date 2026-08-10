import os
from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class MaskKVAttentionRequest:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    mask_index: torch.Tensor | None
    prompt_length: int
    layer_id: int
    layer_count: int
    budget: int
    layer_base_rate: float
    head_base_rate: float
    student_scores: torch.Tensor | None = None


def maskkv_scaled_dot_product_attention(
    request: MaskKVAttentionRequest,
) -> torch.Tensor | None:
    if not _can_prune(request):
        return None

    prompt_length = request.prompt_length
    mask_index = request.mask_index
    if mask_index is None:
        return None

    layer_budget = _layer_budget(request)
    if layer_budget >= prompt_length:
        return None

    mask_queries = _gather_mask_queries(request.q, mask_index)
    if mask_queries is None:
        return None

    scale = 1.0 / sqrt(request.q.shape[-1])
    scores = torch.matmul(mask_queries, request.k.transpose(-1, -2)) * scale
    attention = torch.softmax(scores.float(), dim=-1).to(dtype=request.q.dtype)
    prompt_scores = _prompt_scores(request, attention, prompt_length)
    head_budgets = _head_budgets(attention, request, layer_budget)
    return _attention_with_selected_prompt_tokens(request, prompt_scores, head_budgets)


def _prompt_scores(
    request: MaskKVAttentionRequest,
    attention: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Rank prompt tokens by the utility student when it has spoken, else by attention.

    MaskKV's own ranking is this step's mask->prompt attention mass. The student instead
    predicts how much a position matters over the whole trajectory, which is what the
    keep decision actually needs; the budget split per layer and per head stays MaskKV's.
    Student scores are per layer, so every head in a layer ranks alike and the head
    budgets decide how deep into that ranking each head keeps.
    """
    scores = request.student_scores
    if scores is None:
        return attention[..., :prompt_length].sum(dim=-2)
    layer_scores = scores[request.layer_id].to(
        device=attention.device, dtype=attention.dtype
    )
    if layer_scores.shape[-1] < prompt_length:
        return attention[..., :prompt_length].sum(dim=-2)
    layer_scores = layer_scores[:prompt_length]
    batch, heads = attention.shape[0], attention.shape[1]
    return layer_scores.view(1, 1, -1).expand(batch, heads, -1)


def _can_prune(request: MaskKVAttentionRequest) -> bool:
    same_heads = request.q.shape[1] == request.k.shape[1] == request.v.shape[1]
    positive_budget = request.budget > 0 and request.prompt_length > request.budget
    compatible_shape = request.q.shape[-2] == request.k.shape[-2]
    return same_heads and positive_budget and compatible_shape


def _gather_mask_queries(q: torch.Tensor, mask_index: torch.Tensor) -> torch.Tensor | None:
    if mask_index.ndim != 2 or mask_index.shape[1] != q.shape[-2]:
        return None
    mask_counts = mask_index.sum(dim=1)
    if torch.any(mask_counts == 0):
        return None
    mask_count = int(mask_counts.min().item())
    indices = torch.stack(
        [
            torch.nonzero(row, as_tuple=False).flatten()[:mask_count]
            for row in mask_index
        ],
        dim=0,
    ).to(device=q.device)
    expanded = indices[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[-1])
    return torch.gather(q, dim=2, index=expanded)


def _layer_budget(request: MaskKVAttentionRequest) -> int:
    """Split a fixed pool across layers: flat floor first, the rest by importance.

    `budget` is the per-layer *average*, so the pool is L x budget and it is preserved:
    every layer gets k_base = floor(beta x budget) unconditionally, and the remaining
    L x (budget - k_base) is handed out in proportion to the layer profile. Scaling each
    layer by a rate instead (the earlier form) shrinks the pool -- with beta=0.4 it spent
    only 71% of it, so a run was never comparable to a uniform baseline at the same
    nominal budget.
    """
    layer_count = request.layer_count
    if layer_count <= 1:
        return min(request.budget, request.prompt_length)
    base = int(request.layer_base_rate * request.budget)
    pool = layer_count * (request.budget - base)
    weights = _layer_profile(layer_count)
    share = weights[request.layer_id] / sum(weights)
    budget = base + int(pool * share)
    return max(1, min(request.prompt_length, budget))


def _layer_profile(layer_count: int) -> list[float]:
    """Relative importance per layer, used to hand out the non-flat part of the pool.

    Defaults to distance from the middle of the stack, i.e. the assumption that the
    middle layers are the redundant ones. MASKKV_LAYER_PROFILE overrides it with a
    comma-separated measured profile (one value per layer) so the split can follow what
    the drift measurement actually shows on this model rather than the assumption.
    """
    override = os.getenv("MASKKV_LAYER_PROFILE", "")
    if override:
        values = [float(v) for v in override.split(",") if v.strip()]
        if len(values) == layer_count and sum(values) > 0:
            return [max(0.0, v) for v in values]
    center = (layer_count - 1) / 2
    return [abs(layer_id - center) / center for layer_id in range(layer_count)]


def _head_budgets(
    attention: torch.Tensor,
    request: MaskKVAttentionRequest,
    layer_budget: int,
) -> torch.Tensor:
    prompt_mass = attention[..., : request.prompt_length].sum(dim=(-2, -1))
    response_mass = attention[..., request.prompt_length :].sum(dim=(-2, -1))
    preference = prompt_mass / (prompt_mass + response_mass + 1e-6)
    preference = preference.mean(dim=0)
    preference = preference / preference.sum().clamp_min(1e-6)
    heads = attention.shape[1]
    adaptive = heads * layer_budget * preference
    budget = request.head_base_rate * layer_budget
    budget = budget + (1.0 - request.head_base_rate) * adaptive
    return budget.round().to(dtype=torch.long).clamp(min=1, max=request.prompt_length)


def _attention_with_selected_prompt_tokens(
    request: MaskKVAttentionRequest,
    prompt_scores: torch.Tensor,
    head_budgets: torch.Tensor,
) -> torch.Tensor:
    head_outputs: list[torch.Tensor] = []
    response_k = request.k[:, :, request.prompt_length :, :]
    response_v = request.v[:, :, request.prompt_length :, :]
    for head_index in range(request.q.shape[1]):
        budget = int(head_budgets[head_index].item())
        keep = torch.topk(prompt_scores[:, head_index, :], k=budget, dim=-1).indices
        expanded = keep[:, :, None].expand(-1, -1, request.k.shape[-1])
        prompt_k = torch.gather(
            request.k[:, head_index, : request.prompt_length, :],
            dim=1,
            index=expanded,
        )
        prompt_v = torch.gather(
            request.v[:, head_index, : request.prompt_length, :],
            dim=1,
            index=expanded,
        )
        key = torch.cat([prompt_k, response_k[:, head_index, :, :]], dim=1)
        value = torch.cat([prompt_v, response_v[:, head_index, :, :]], dim=1)
        head_output = F.scaled_dot_product_attention(
            request.q[:, head_index : head_index + 1, :, :],
            key[:, None, :, :],
            value[:, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        )
        head_outputs.append(head_output)
    return torch.cat(head_outputs, dim=1)
