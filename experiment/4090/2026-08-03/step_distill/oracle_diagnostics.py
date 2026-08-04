from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import torch

from .schema import load_teacher_shard


@dataclass(frozen=True, slots=True)
class DiagnosticError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class TemporalMetrics:
    adjacent_jaccard: float
    early_late_jaccard: float
    union_ratio: float
    step_churn: float


@dataclass(frozen=True, slots=True)
class DiagnosticSummary:
    shard_count: int
    budget: int
    adjacent_jaccard: float
    early_late_jaccard: float
    union_ratio: float
    step_churn: float


def effective_budget(
    *,
    requested: int,
    prompt_length: int,
    order_width: int,
) -> int:
    """Apply the selection contract B_eff=min(B, P) at diagnostics time."""
    if requested <= 0 or prompt_length <= 0 or order_width <= 0:
        raise DiagnosticError("diagnostic budgets and widths must be positive")
    selected = min(requested, prompt_length)
    if selected > order_width:
        raise DiagnosticError("stored order is shorter than the effective budget")
    return selected


def temporal_metrics(
    order: torch.Tensor,
    valid_step_mask: torch.Tensor,
    *,
    budget: int,
) -> TemporalMetrics:
    """Summarize temporal mask change, averaging independently over layers."""
    if order.ndim != 3:
        raise DiagnosticError("order must have shape [step, layer, k]")
    if valid_step_mask.shape != (order.shape[0],):
        raise DiagnosticError("valid_step_mask does not match the step axis")
    if budget <= 0 or budget > order.shape[2]:
        raise DiagnosticError("budget must be within the stored order width")
    valid_steps = valid_step_mask.nonzero(as_tuple=False).flatten().tolist()
    if not valid_steps:
        raise DiagnosticError("at least one valid step is required")

    adjacent: list[float] = []
    early_late: list[float] = []
    union_ratios: list[float] = []
    churn: list[float] = []
    for layer_id in range(order.shape[1]):
        selections = [
            set(order[step_id, layer_id, :budget].tolist()) for step_id in valid_steps
        ]
        adjacent.extend(
            _jaccard(previous, current) for previous, current in pairwise(selections)
        )
        churn.extend(
            len(previous - current) / float(budget)
            for previous, current in pairwise(selections)
        )
        early_late.append(_jaccard(selections[0], selections[-1]))
        union_ratios.append(len(set().union(*selections)) / float(budget))

    return TemporalMetrics(
        adjacent_jaccard=_mean_or_identity(adjacent, identity=1.0),
        early_late_jaccard=sum(early_late) / len(early_late),
        union_ratio=sum(union_ratios) / len(union_ratios),
        step_churn=_mean_or_identity(churn, identity=0.0),
    )


def summarize_shards(
    input_root: Path,
    *,
    budget: int,
) -> DiagnosticSummary:
    """Load every valid shard and average plain top-order diagnostics."""
    paths = sorted(input_root.rglob("*.pt"))
    if not paths:
        raise DiagnosticError(f"no .pt shards found below {input_root}")
    metrics: list[TemporalMetrics] = []
    for path in paths:
        shard = load_teacher_shard(path)
        order = shard.top_order
        selected_budget = effective_budget(
            requested=budget,
            prompt_length=int(shard.prompt_input_ids.numel()),
            order_width=int(order.shape[2]),
        )
        metrics.append(
            temporal_metrics(
                order,
                shard.valid_step_mask,
                budget=selected_budget,
            )
        )
    return DiagnosticSummary(
        shard_count=len(metrics),
        budget=budget,
        adjacent_jaccard=sum(item.adjacent_jaccard for item in metrics) / len(metrics),
        early_late_jaccard=sum(item.early_late_jaccard for item in metrics)
        / len(metrics),
        union_ratio=sum(item.union_ratio for item in metrics) / len(metrics),
        step_churn=sum(item.step_churn for item in metrics) / len(metrics),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure temporal change in plain per-step teacher masks."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_shards(
        args.input_root,
        budget=args.budget,
    )
    payload = json.dumps(asdict(summary), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _mean_or_identity(values: list[float], *, identity: float) -> float:
    if not values:
        return identity
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
