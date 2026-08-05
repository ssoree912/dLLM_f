from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .distribution_forward import decoder_embedding, extract_prompt_features
from .distribution_pruning import (
    DistributionPruningController,
    install_distribution_pruner,
)
from .distribution_step import (
    DistributionRolloutConfig,
    DistributionStepReport,
    pool_causal_state,
    run_distribution_step,
)
from .distribution_student import StateConditionedSelector
from .oracle_generation import _planned_transfer_counts


@dataclass(frozen=True, slots=True)
class DistributionRolloutResult:
    generated_ids: torch.Tensor
    reports: tuple[DistributionStepReport, ...]
    complete: bool


def run_distribution_rollout(
    model: nn.Module,
    selector: StateConditionedSelector,
    prompt_ids: torch.Tensor,
    initial_state_indices: torch.Tensor,
    config: DistributionRolloutConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int = 0,
) -> DistributionRolloutResult:
    """Roll out x_t^pi while distilling a same-state full-cache teacher."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("online rollout expects prompt ids with shape [1, prompt]")
    prompt_length = int(prompt_ids.shape[1])
    prompt_features = extract_prompt_features(model, prompt_ids)
    layer_count = int(prompt_features.shape[0])
    controller = DistributionPruningController(prompt_length, layer_count)
    install_distribution_pruner(
        model,
        controller,
        gate_temperature=config.gate_temperature,
    )
    suffix_ids = torch.full(
        (1, config.gen_length),
        config.mask_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    reports: list[DistributionStepReport] = []
    try:
        _rollout_blocks(
            model,
            selector,
            prompt_ids,
            initial_state_indices,
            suffix_ids,
            prompt_features,
            controller,
            config,
            optimizer,
            reports,
            max_steps,
        )
    finally:
        controller.restore()
    complete = len(reports) == config.steps
    return DistributionRolloutResult(
        suffix_ids.detach().cpu(),
        tuple(reports),
        complete,
    )


def _rollout_blocks(
    model: nn.Module,
    selector: StateConditionedSelector,
    prompt_ids: torch.Tensor,
    initial_indices: torch.Tensor,
    suffix_ids: torch.Tensor,
    prompt_features: torch.Tensor,
    controller: DistributionPruningController,
    config: DistributionRolloutConfig,
    optimizer: torch.optim.Optimizer | None,
    reports: list[DistributionStepReport],
    max_steps: int,
) -> None:
    block_count = config.gen_length // config.block_length
    steps_per_block = config.steps // block_count
    embedding = decoder_embedding(model)
    for block_id in range(block_count):
        start = block_id * config.block_length
        end = start + config.block_length
        counts = _planned_transfer_counts(
            suffix_ids[:, start:end] == config.mask_id,
            steps_per_block,
        )
        for block_step in range(steps_per_block):
            if max_steps > 0 and len(reports) >= max_steps:
                return
            report, transfer, candidates = run_distribution_step(
                model,
                selector,
                prompt_ids,
                initial_indices,
                suffix_ids,
                prompt_features,
                controller,
                config,
                optimizer,
                embedding,
                start,
                end,
                counts[:, block_step],
                len(reports),
            )
            suffix_ids[transfer] = candidates[transfer]
            reports.append(report)


__all__ = [
    "DistributionRolloutConfig",
    "DistributionRolloutResult",
    "DistributionStepReport",
    "pool_causal_state",
    "run_distribution_rollout",
]
