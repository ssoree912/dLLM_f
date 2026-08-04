from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .generation_output import BatchDecodeTokenizer, decode_generation
from .qa_metrics import qa_scores


@dataclass(frozen=True, slots=True)
class OracleResult:
    sample_id: str
    dataset: str
    method: str
    budget: int | None
    prompt_length: int
    gold: str
    prediction: str
    raw_prediction: str
    f1: float
    recall: float
    exact_match: float
    raw_f1: float
    raw_recall: float
    raw_exact_match: float
    tokens_before_stop: int
    first_stop_position: int | None
    first_stop_token_id: int | None
    trailing_token_count: int
    generated_ids: list[int]
    matches_teacher_trajectory: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OracleSummary:
    method: str
    budget: int | None
    samples: int
    f1: float
    recall: float
    exact_match: float
    raw_f1: float
    raw_recall: float
    stop_rate: float
    mean_tokens_before_stop: float
    teacher_trajectory_match_rate: float


def build_oracle_result(
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
) -> OracleResult:
    decoded = decode_generation(
        tokenizer,
        generated,
        stop_token_ids=stop_token_ids,
    )
    scores = qa_scores(decoded.prediction, gold)
    raw_scores = qa_scores(decoded.raw_prediction, gold)
    return OracleResult(
        sample_id=sample_id,
        dataset=dataset,
        method=method,
        budget=budget,
        prompt_length=prompt_length,
        gold=gold,
        prediction=decoded.prediction,
        raw_prediction=decoded.raw_prediction,
        f1=scores.f1,
        recall=scores.recall,
        exact_match=scores.exact_match,
        raw_f1=raw_scores.f1,
        raw_recall=raw_scores.recall,
        raw_exact_match=raw_scores.exact_match,
        tokens_before_stop=decoded.tokens_before_stop,
        first_stop_position=decoded.first_stop_position,
        first_stop_token_id=decoded.first_stop_token_id,
        trailing_token_count=decoded.trailing_token_count,
        generated_ids=[int(token_id) for token_id in generated[0].tolist()],
        matches_teacher_trajectory=matches_teacher_trajectory,
        elapsed_seconds=elapsed_seconds,
    )


def result_path(
    output_root: Path,
    method: str,
    budget: int | None,
    shard_path: Path,
) -> Path:
    label = "full" if budget is None else f"b{budget}"
    return output_root / "records" / method / label / f"{shard_path.stem}.json"


def save_result_atomic(result: OracleResult, path: Path) -> None:
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


def load_results(output_root: Path) -> list[OracleResult]:
    results: list[OracleResult] = []
    for path in sorted((output_root / "records").rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            results.append(OracleResult(**json.load(handle)))
    return results


def summarize_results(results: list[OracleResult]) -> list[OracleSummary]:
    groups: dict[tuple[str, int | None], list[OracleResult]] = {}
    for result in results:
        groups.setdefault((result.method, result.budget), []).append(result)
    summaries: list[OracleSummary] = []
    for (method, budget), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1] or 0),
    ):
        count = len(group)
        summaries.append(
            OracleSummary(
                method=method,
                budget=budget,
                samples=count,
                f1=sum(item.f1 for item in group) / count,
                recall=sum(item.recall for item in group) / count,
                exact_match=sum(item.exact_match for item in group) / count,
                raw_f1=sum(item.raw_f1 for item in group) / count,
                raw_recall=sum(item.raw_recall for item in group) / count,
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
    return summaries


def save_summary_atomic(summaries: list[OracleSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(summary) for summary in summaries]
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
