"""Fixed policy for the refactored prompt-pruning cache path.

The refactored path intentionally has no experimental budget knobs.  It keeps
half of the actual prompt and refreshes half of that retained set.  The ratios
therefore remain correct when generation reserves part of the 2048-token
context (for example, a 1920-token prompt becomes 960 kept / 480 refreshed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


MAX_CONTEXT_TOKENS: Final = 2048
ATTENTION_HEAD: Final = "attention"
DELTA_HEAD: Final = "delta"
PRUNE_CACHE_HEADS: Final = (ATTENTION_HEAD, DELTA_HEAD)


@dataclass(frozen=True, slots=True)
class PruneCacheBudget:
    prompt_tokens: int
    kept_tokens: int
    updated_tokens: int


def split_prune_cache_budget(prompt_tokens: int) -> PruneCacheBudget:
    """Keep half the prompt and update half of the kept tokens.

    Odd sizes round up so a non-empty prompt always keeps and updates at least
    one token.  The update set is chosen from the retained set, not the original
    prompt.
    """

    if prompt_tokens <= 0:
        raise RuntimeError("prompt_tokens must be positive")
    kept_tokens = max(1, (prompt_tokens + 1) // 2)
    updated_tokens = max(1, (kept_tokens + 1) // 2)
    return PruneCacheBudget(
        prompt_tokens=prompt_tokens,
        kept_tokens=kept_tokens,
        updated_tokens=updated_tokens,
    )


def resolve_generation_kwargs(gen_kwargs: dict) -> dict:
    """Fill dLLM generation fields from each task's official output length."""

    resolved = dict(gen_kwargs)
    gen_length = int(
        resolved.get("gen_length", resolved.get("max_gen_toks", 128))
    )
    resolved.setdefault("gen_length", gen_length)
    resolved.setdefault("steps", gen_length)
    resolved.setdefault("block_length", 32)
    return resolved


def validate_prune_cache_heads(heads: tuple[str, ...] | list[str]) -> None:
    actual = tuple(heads)
    if actual != PRUNE_CACHE_HEADS:
        raise RuntimeError(
            "prune_cache requires one joint student with attention and delta "
            f"heads, got {actual}"
        )
