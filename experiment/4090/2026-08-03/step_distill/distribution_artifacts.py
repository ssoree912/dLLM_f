from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch

from .distribution_pruning import DISTILLATION_BUDGET
from .distribution_rollout import (
    DistributionRolloutConfig,
    DistributionRolloutResult,
)
from .distribution_student import SelectorConfig, StateConditionedSelector
from .generation_output import (
    LLadaDecodeTokenizer,
    decode_generation,
    llada_stop_token_ids,
)
from .summary_metrics import rouge_l_recall, rouge_l_score


class MetricsWriter(Protocol):
    def write(self, text: str, /) -> int: ...

    def flush(self) -> None: ...


def result_record(
    result: DistributionRolloutResult,
    *,
    split: str,
    sample_id: str,
    epoch: int,
    gold: str,
    tokenizer: LLadaDecodeTokenizer,
) -> dict[str, object]:
    if not result.reports:
        raise RuntimeError("online rollout produced no step reports")
    count = len(result.reports)
    record: dict[str, object] = {
        "split": split,
        "sample_id": sample_id,
        "epoch": epoch,
        "budget": DISTILLATION_BUDGET,
        "steps": count,
        "complete": result.complete,
        "step_reports": [asdict(report) for report in result.reports],
    }
    for field in (
        "loss",
        "kd_loss",
        "diagnostic_kl",
        "commit_loss",
        "selector_grad_norm",
        "token_top1_agreement",
        "commit_jaccard",
    ):
        record[field] = sum(getattr(report, field) for report in result.reports) / count
    if result.complete:
        decoded = decode_generation(
            tokenizer,
            result.generated_ids,
            stop_token_ids=llada_stop_token_ids(tokenizer),
        )
        record.update(
            prediction=decoded.prediction,
            raw_prediction=decoded.raw_prediction,
            rouge_l_f1=rouge_l_score(decoded.prediction, gold),
            raw_rouge_l_f1=rouge_l_score(decoded.raw_prediction, gold),
            lcs_recall=rouge_l_recall(decoded.prediction, gold),
            raw_lcs_recall=rouge_l_recall(decoded.raw_prediction, gold),
            output_tokens=decoded.tokens_before_stop,
            tokens_before_stop=decoded.tokens_before_stop,
            canvas_token_count=decoded.canvas_token_count,
            first_stop_position=decoded.first_stop_position,
            first_stop_token_id=decoded.first_stop_token_id,
            trailing_token_count=decoded.trailing_token_count,
        )
    return record


def write_record(handle: MetricsWriter, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def save_checkpoint(
    path: Path,
    selector: StateConditionedSelector,
    optimizer: torch.optim.Optimizer,
    selector_config: SelectorConfig,
    rollout_config: DistributionRolloutConfig,
    train_config: dict[str, object],
    train_ids: list[str],
    validation_ids: list[str],
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "budget": DISTILLATION_BUDGET,
            "selector_config": asdict(selector_config),
            "rollout_config": asdict(rollout_config),
            "train_config": train_config,
            "selector": selector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_sample_ids": train_ids,
            "validation_sample_ids": validation_ids,
            "rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


__all__ = ["MetricsWriter", "result_record", "save_checkpoint", "write_record"]
