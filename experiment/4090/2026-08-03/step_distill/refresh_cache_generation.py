from __future__ import annotations

from typing import Protocol

import torch

from .oracle_generation import (
    ReplayGenerationConfig,
    _build_transfer_index,
    _planned_transfer_counts,
)
from .trajectory_teacher import select_candidates


class RefreshLogitRunner(Protocol):
    def start_step(self, step_id: int) -> bool: ...

    def refresh_logits(
        self,
        prompt_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor: ...

    def cached_logits(self, suffix_ids: torch.Tensor) -> torch.Tensor: ...


@torch.inference_mode()
def generate_with_refresh_cache(
    runner: RefreshLogitRunner,
    prompt_ids: torch.Tensor,
    config: ReplayGenerationConfig,
) -> torch.Tensor:
    """Generate with prompt KV refreshes and a step-conditioned attention set."""
    suffix_ids = torch.full(
        (1, config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    block_count = config.gen_length // config.block_length
    steps_per_block = config.steps // block_count
    global_step = 0
    for block_id in range(block_count):
        start = block_id * config.block_length
        end = (block_id + 1) * config.block_length
        transfer_counts = _planned_transfer_counts(
            suffix_ids[:, start:end] == config.mask_id,
            steps_per_block,
        )
        for block_step in range(steps_per_block):
            refresh = runner.start_step(global_step)
            if refresh:
                logits = runner.refresh_logits(prompt_ids, suffix_ids)
            else:
                logits = runner.cached_logits(suffix_ids)
            candidates, confidence = select_candidates(
                logits.float(),
                suffix_ids,
                suffix_ids == config.mask_id,
                temperature=config.temperature,
            )
            confidence[:, end:] = -torch.inf
            transfer = _build_transfer_index(
                confidence,
                transfer_counts[:, block_step],
            )
            suffix_ids[transfer] = candidates[transfer]
            global_step += 1
    return suffix_ids
