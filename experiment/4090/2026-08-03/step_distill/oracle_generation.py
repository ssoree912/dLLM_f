from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .trajectory_teacher import select_candidates


class StepController(Protocol):
    def set_step(self, step_id: int) -> None: ...


class LogitOutput(Protocol):
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayGenerationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ReplayGenerationConfig:
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    mask_id: int = 126336

    def __post_init__(self) -> None:
        if self.gen_length <= 0 or self.block_length <= 0 or self.steps <= 0:
            raise ReplayGenerationError(
                "generation lengths and steps must be positive"
            )
        if self.gen_length % self.block_length != 0:
            raise ReplayGenerationError(
                "gen_length must be divisible by block_length"
            )
        block_count = self.gen_length // self.block_length
        if self.steps % block_count != 0:
            raise ReplayGenerationError("steps must be divisible by block count")
        if self.temperature < 0.0:
            raise ReplayGenerationError("temperature must be non-negative")


@torch.inference_mode()
def generate_offline_replay(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    config: ReplayGenerationConfig,
    *,
    step_controller: StepController | None = None,
) -> torch.Tensor:
    """Generate with the exact teacher schedule and optional stored step masks."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ReplayGenerationError(
            "offline replay expects prompt_ids with shape [1, prompt]"
        )
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
            if step_controller is not None:
                step_controller.set_step(global_step)
            sequence = torch.cat((prompt_ids, suffix_ids), dim=1)
            output: LogitOutput = model(
                sequence,
                attention_mask=torch.ones_like(sequence),
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[:, prompt_ids.shape[1] :].float()
            candidates, confidence = select_candidates(
                logits,
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


def _planned_transfer_counts(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_count = mask_index.sum(dim=1, keepdim=True)
    base = mask_count // steps
    remainder = mask_count % steps
    counts = base.expand(-1, steps).clone()
    step_ids = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    counts += step_ids < remainder
    return counts.to(torch.long)


def _build_transfer_index(
    confidence: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    transfer = torch.zeros_like(confidence, dtype=torch.bool)
    for batch_id in range(confidence.shape[0]):
        count = int(counts[batch_id])
        if count > 0:
            selected = torch.topk(confidence[batch_id], k=count).indices
            transfer[batch_id, selected] = True
    return transfer
