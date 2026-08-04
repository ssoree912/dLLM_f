from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .generation_output import BatchDecodeTokenizer, decode_generation
from .summary_metrics import rouge_l_score


@dataclass(frozen=True, slots=True)
class SummaryOracleResult:
    sample_id: str
    dataset: str
    method: str
    budget: int | None
    prompt_length: int
    gold: str
    prediction: str
    raw_prediction: str
    rouge_l: float
    raw_rouge_l: float
    tokens_before_stop: int
    first_stop_position: int | None
    trailing_token_count: int
    generated_ids: list[int]
    matches_teacher_trajectory: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SummaryOracleAggregate:
    method: str
    budget: int | None
    samples: int
    rouge_l: float
    raw_rouge_l: float
    stop_rate: float
    mean_tokens_before_stop: float
    teacher_trajectory_match_rate: float


def build_summary_result(
    tokenizer: BatchDecodeTokenizer,
    generated: torch.Tensor,
    *,
    stop_token_ids: frozenset[int],
    sample_id: str,
    dataset: str,
    method: str,
    budget: int | None,
    prompt_length: int,
    gold: str,
    matches_teacher_trajectory: bool,
    elapsed_seconds: float,
) -> SummaryOracleResult:
    decoded = decode_generation(
        tokenizer,
        generated,
        stop_token_ids=stop_token_ids,
    )
    return SummaryOracleResult(
        sample_id=sample_id,
        dataset=dataset,
        method=method,
        budget=budget,
        prompt_length=prompt_length,
        gold=gold,
        prediction=decoded.prediction,
        raw_prediction=decoded.raw_prediction,
        rouge_l=rouge_l_score(decoded.prediction, gold),
        raw_rouge_l=rouge_l_score(decoded.raw_prediction, gold),
        tokens_before_stop=decoded.tokens_before_stop,
        first_stop_position=decoded.first_stop_position,
        trailing_token_count=decoded.trailing_token_count,
        generated_ids=[int(token_id) for token_id in generated[0].tolist()],
        matches_teacher_trajectory=matches_teacher_trajectory,
        elapsed_seconds=elapsed_seconds,
    )


def summary_result_path(
    output_root: Path,
    method: str,
    budget: int | None,
    shard_path: Path,
) -> Path:
    label = "full" if budget is None else f"b{budget}"
    return output_root / "records" / method / label / f"{shard_path.stem}.json"


def save_summary_result_atomic(result: SummaryOracleResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_summary_results(output_root: Path) -> list[SummaryOracleResult]:
    results: list[SummaryOracleResult] = []
    for path in sorted((output_root / "records").rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            results.append(SummaryOracleResult(**json.load(handle)))
    return results


def aggregate_summary_results(
    results: list[SummaryOracleResult],
) -> list[SummaryOracleAggregate]:
    groups: dict[tuple[str, int | None], list[SummaryOracleResult]] = {}
    for result in results:
        groups.setdefault((result.method, result.budget), []).append(result)
    aggregates: list[SummaryOracleAggregate] = []
    for (method, budget), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1] or 0),
    ):
        count = len(group)
        aggregates.append(
            SummaryOracleAggregate(
                method=method,
                budget=budget,
                samples=count,
                rouge_l=sum(item.rouge_l for item in group) / count,
                raw_rouge_l=sum(item.raw_rouge_l for item in group) / count,
                stop_rate=sum(
                    item.first_stop_position is not None for item in group
                )
                / count,
                mean_tokens_before_stop=sum(
                    item.tokens_before_stop for item in group
                )
                / count,
                teacher_trajectory_match_rate=sum(
                    item.matches_teacher_trajectory for item in group
                )
                / count,
            )
        )
    return aggregates


def save_summary_aggregates_atomic(
    aggregates: list[SummaryOracleAggregate],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [asdict(aggregate) for aggregate in aggregates],
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
