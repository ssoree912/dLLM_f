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
    prompt_scores = attention[..., :prompt_length].sum(dim=-2)
    head_budgets = _head_budgets(attention, request, layer_budget)
    return _attention_with_selected_prompt_tokens(request, prompt_scores, head_budgets)


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
    if request.layer_count <= 1:
        return min(request.budget, request.prompt_length)
    center = (request.layer_count - 1) / 2
    distance = abs(request.layer_id - center) / center
    rate = request.layer_base_rate + (1.0 - request.layer_base_rate) * distance
    budget = round(request.budget * rate)
    return max(1, min(request.prompt_length, budget))


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
