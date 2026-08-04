from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QaScores:
    f1: float
    recall: float
    exact_match: float


def normalize_answer(text: str) -> str:
    """Apply the LongBench QA normalization used by its token F1 metric."""
    lowered = text.lower()
    without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
    alphanumeric = re.sub(r"[^0-9a-z]+", " ", without_articles)
    return " ".join(alphanumeric.split())


def qa_scores(prediction: str, ground_truth: str) -> QaScores:
    """Return F1 and recall separately so trailing noise stays observable."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not ground_truth_tokens:
        exact = float(not prediction_tokens)
        return QaScores(f1=exact, recall=exact, exact_match=exact)
    overlap = sum(
        (Counter(prediction_tokens) & Counter(ground_truth_tokens)).values()
    )
    recall = overlap / len(ground_truth_tokens)
    if not prediction_tokens or overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(prediction_tokens)
        f1 = 2.0 * precision * recall / (precision + recall)
    exact_match = float(
        " ".join(prediction_tokens) == " ".join(ground_truth_tokens)
    )
    return QaScores(f1=f1, recall=recall, exact_match=exact_match)
