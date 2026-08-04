from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from typing_extensions import assert_never

from .selection import stable_top_order


class AttentionPolicy(str, Enum):
    ALL_CACHE = "all_cache"
    DYNAMIC_TOPK = "dynamic_topk"


@dataclass(frozen=True, slots=True)
class RefreshCacheError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class RefreshCacheConfig:
    cache_budget: int
    attention_budget: int
    refresh_interval: int
    attention_policy: AttentionPolicy

    def __post_init__(self) -> None:
        if min(self.cache_budget, self.attention_budget, self.refresh_interval) <= 0:
            raise RefreshCacheError(
                "cache budgets and refresh interval must be positive"
            )
        if self.attention_budget > self.cache_budget:
            raise RefreshCacheError("attention budget cannot exceed cache budget")


@dataclass(frozen=True, slots=True)
class LayerPromptKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class RefreshTeacherTargets:
    order: torch.Tensor
    scores: torch.Tensor


def build_refresh_targets(
    order: torch.Tensor,
    ranked_scores: torch.Tensor,
) -> RefreshTeacherTargets:
    """Restore dense per-position scores from a complete ranked teacher shard."""
    if order.ndim != 3 or ranked_scores.shape != order.shape:
        raise RefreshCacheError("ranked teacher tensors must share [step, layer, rank]")
    width = int(order.shape[2])
    order_cpu = order.detach().cpu().to(torch.long)
    expected = torch.arange(width).expand_as(order_cpu)
    if not torch.equal(torch.sort(order_cpu, dim=-1).values, expected):
        raise RefreshCacheError("refresh teacher order must be a full permutation")
    scores_cpu = ranked_scores.detach().cpu().float()
    if not torch.isfinite(scores_cpu).all():
        raise RefreshCacheError("refresh teacher scores must be finite")
    dense_scores = torch.empty_like(scores_cpu).scatter_(-1, order_cpu, scores_cpu)
    return RefreshTeacherTargets(order=order_cpu, scores=dense_scores)


class RefreshPromptKVCache:
    """Mutable per-layer prompt KV state driven by full per-step teacher orders."""

    __slots__ = (
        "_cached_indices",
        "_layers",
        "config",
        "layer_count",
        "order",
        "prompt_length",
        "scores",
        "step_id",
    )

    def __init__(
        self,
        prompt_length: int,
        targets: RefreshTeacherTargets,
        config: RefreshCacheConfig,
    ) -> None:
        layer_count = int(targets.order.shape[1])
        if prompt_length <= 0 or layer_count <= 0:
            raise RefreshCacheError("prompt length and layer count must be positive")
        if targets.order.shape != targets.scores.shape:
            raise RefreshCacheError("teacher score and order shapes must match")
        if targets.order.shape[2] != prompt_length:
            raise RefreshCacheError(
                "exact cache-restricted ranking requires a full prompt order"
            )
        self.prompt_length = prompt_length
        self.layer_count = layer_count
        self.order = targets.order
        self.scores = targets.scores
        self.config = config
        self.step_id = 0
        self._cached_indices: torch.Tensor | None = None
        self._layers: dict[int, LayerPromptKV] = {}

    @property
    def cache_size(self) -> int:
        return min(self.config.cache_budget, self.prompt_length)

    @property
    def attention_size(self) -> int:
        return min(self.config.attention_budget, self.cache_size)

    def start_step(self, step_id: int) -> bool:
        if step_id < 0 or step_id >= self.order.shape[0]:
            raise RefreshCacheError(f"step {step_id} falls outside teacher order")
        self.step_id = step_id
        refresh = (
            self._cached_indices is None or step_id % self.config.refresh_interval == 0
        )
        if refresh:
            global_scores = self.scores[step_id].mean(dim=0)
            self._cached_indices = stable_top_order(global_scores, self.cache_size)
            self._layers.clear()
        return refresh

    def cached_prompt_indices(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        indices = self._cached_indices
        if indices is None:
            raise RefreshCacheError(
                "cache indices are unavailable before the first step"
            )
        return indices.to(device)

    def attention_prompt_indices(
        self,
        layer_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = self.cached_prompt_indices(torch.device("cpu"))
        match self.config.attention_policy:
            case AttentionPolicy.ALL_CACHE:
                selected = cached
            case AttentionPolicy.DYNAMIC_TOPK:
                membership = torch.zeros(self.prompt_length, dtype=torch.bool)
                membership[cached] = True
                current_order = self.order[self.step_id, layer_id]
                selected = current_order[membership[current_order]][
                    : self.attention_size
                ]
            case unreachable:
                assert_never(unreachable)
        if selected.numel() != self.attention_size and (
            self.config.attention_policy is AttentionPolicy.DYNAMIC_TOPK
        ):
            raise RefreshCacheError(
                "teacher order did not cover the requested attention set"
            )
        return selected.to(device)

    def add_layer(
        self,
        layer_id: int,
        prompt_key: torch.Tensor,
        prompt_value: torch.Tensor,
    ) -> None:
        if (
            prompt_key.shape[2] != self.cache_size
            or prompt_value.shape != prompt_key.shape
        ):
            raise RefreshCacheError("refreshed prompt KV tensors must match cache size")
        self._layers[layer_id] = LayerPromptKV(
            key=prompt_key.detach(),
            value=prompt_value.detach(),
        )

    def layer(
        self,
        layer_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LayerPromptKV:
        layer = self._layers.get(layer_id)
        if layer is None:
            raise RefreshCacheError(f"prompt KV cache missing layer {layer_id}")
        cached = self.cached_prompt_indices(torch.device("cpu"))
        selected = self.attention_prompt_indices(layer_id, torch.device("cpu"))
        offset_by_position = torch.full((self.prompt_length,), -1, dtype=torch.long)
        offset_by_position[cached] = torch.arange(cached.numel())
        offsets = offset_by_position[selected].to(device)
        return LayerPromptKV(
            key=layer.key.to(device=device, dtype=dtype).index_select(2, offsets),
            value=layer.value.to(device=device, dtype=dtype).index_select(2, offsets),
        )

    def assert_complete(self) -> None:
        missing = sorted(set(range(self.layer_count)) - set(self._layers))
        if missing:
            raise RefreshCacheError(f"prompt KV cache missing layers: {missing}")
