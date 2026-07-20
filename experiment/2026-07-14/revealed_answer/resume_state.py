from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class TrainingRecord(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float | None
    best_metric: float
    elapsed: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    best_metric: float | None


def read_resume_state(log_path: Path) -> ResumeState:
    if not log_path.exists():
        return ResumeState(best_metric=None)
    best_metric: float | None = None
    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if not line.strip():
                continue
            record: TrainingRecord = json.loads(line)
            metric = record["val_loss"] if record["val_loss"] is not None else record["train_loss"]
            if best_metric is None or metric < best_metric:
                best_metric = metric
    return ResumeState(best_metric=best_metric)
