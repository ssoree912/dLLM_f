from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TopKMetric:
    recall: float
    ndcg: float
    teacher_mass: float

    def to_dict(self) -> dict[str, float]:
        return {
            "recall": self.recall,
            "ndcg": self.ndcg,
            "teacher_mass": self.teacher_mass,
        }


def score_topk_metrics(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    k: int,
) -> TopKMetric:
    if student_scores.shape != teacher_scores.shape or student_scores.ndim != 1:
        raise RuntimeError("student and teacher scores must have matching [prompt] shape")
    top_count = min(max(1, k), teacher_scores.numel())
    student_top = torch.topk(student_scores.float(), top_count, largest=True).indices
    teacher_top = torch.topk(teacher_scores.float(), top_count, largest=True).indices
    hits = torch.isin(student_top, teacher_top).float().mean()
    return TopKMetric(
        recall=float(hits.item()),
        ndcg=compute_ndcg(student_top, teacher_scores.float(), top_count),
        teacher_mass=float(teacher_scores.index_select(0, student_top).sum().item()),
    )


def compute_ndcg(student_top: torch.Tensor, teacher_scores: torch.Tensor, k: int) -> float:
    gains = teacher_scores.index_select(0, student_top)
    ideal = torch.topk(teacher_scores, min(k, teacher_scores.numel()), largest=True).values
    discounts = torch.log2(
        torch.arange(2, gains.numel() + 2, dtype=torch.float32, device=gains.device)
    )
    dcg = (gains / discounts).sum()
    ideal_dcg = (ideal / discounts[: ideal.numel()]).sum()
    if float(ideal_dcg.item()) <= 0.0:
        return 0.0
    return float((dcg / ideal_dcg).item())
