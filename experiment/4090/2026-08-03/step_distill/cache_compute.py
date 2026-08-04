from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheComputeError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ModelShape:
    layers: int
    hidden_size: int
    mlp_hidden_size: int
    vocabulary_size: int


@dataclass(frozen=True, slots=True)
class CacheSchedule:
    steps: int
    refresh_interval: int
    suffix_length: int
    cache_budget: int
    attention_budget: int


@dataclass(frozen=True, slots=True)
class CacheFlopRequest:
    shape: ModelShape
    schedule: CacheSchedule
    prompt_length: int


@dataclass(frozen=True, slots=True)
class CacheFlops:
    refresh_count: int
    effective_cache_tokens: int
    effective_attention_tokens: int
    attention_flops: int
    dense_flops: int
    logit_flops: int
    total_flops: int


def estimate_cache_flops(request: CacheFlopRequest) -> CacheFlops:
    """Estimate one sample's dense-model FLOPs under a prompt-KV refresh schedule."""
    shape = request.shape
    schedule = request.schedule
    values = (
        shape.layers,
        shape.hidden_size,
        shape.mlp_hidden_size,
        shape.vocabulary_size,
        schedule.steps,
        schedule.refresh_interval,
        schedule.suffix_length,
        schedule.cache_budget,
        schedule.attention_budget,
        request.prompt_length,
    )
    if min(values) <= 0:
        raise CacheComputeError(
            "model dimensions, budgets, and lengths must be positive"
        )
    if schedule.attention_budget > schedule.cache_budget:
        raise CacheComputeError("attention budget cannot exceed cache budget")

    cache_tokens = min(request.prompt_length, schedule.cache_budget)
    attention_tokens = min(cache_tokens, schedule.attention_budget)
    refresh_count = (schedule.steps + schedule.refresh_interval - 1) // (
        schedule.refresh_interval
    )
    cached_steps = schedule.steps - refresh_count
    query_token_steps = (
        refresh_count * (cache_tokens + schedule.suffix_length)
        + cached_steps * schedule.suffix_length
    )
    key_tokens = attention_tokens + schedule.suffix_length
    attention_flops = (
        4 * shape.layers * shape.hidden_size * query_token_steps * key_tokens
    )
    dense_per_token_layer = (
        8 * shape.hidden_size**2 + 6 * shape.hidden_size * shape.mlp_hidden_size
    )
    dense_flops = shape.layers * query_token_steps * dense_per_token_layer
    logit_flops = (
        2
        * schedule.steps
        * schedule.suffix_length
        * shape.hidden_size
        * shape.vocabulary_size
    )
    return CacheFlops(
        refresh_count=refresh_count,
        effective_cache_tokens=cache_tokens,
        effective_attention_tokens=attention_tokens,
        attention_flops=attention_flops,
        dense_flops=dense_flops,
        logit_flops=logit_flops,
        total_flops=attention_flops + dense_flops + logit_flops,
    )
