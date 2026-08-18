from __future__ import annotations

from types import SimpleNamespace

import torch

from dllm_cache.budget.drift_refresh_kv import select_delta_student_topk


class HiddenScoreStudent:
    def __init__(self) -> None:
        self.question_indices: list[torch.Tensor] = []

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        prompt_indices: torch.Tensor,
        question_indices: torch.Tensor,
    ) -> torch.Tensor:
        del layer_id
        self.question_indices.append(question_indices.cpu())
        return hidden_states.index_select(1, prompt_indices)[..., 0]


class TwoHeadStudent(HiddenScoreStudent):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(heads=("attention", "delta"))
        self.selected_heads: list[str | None] = []

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        prompt_indices: torch.Tensor,
        question_indices: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor:
        self.selected_heads.append(head)
        scores = super().forward_layer(
            layer_id,
            hidden_states,
            prompt_indices,
            question_indices,
        )
        return -scores if head == "attention" else scores


def test_delta_student_topk_is_recomputed_from_current_hidden_states() -> None:
    student = HiddenScoreStudent()
    keep_positions = torch.tensor([2, 7, 8, 9])
    first = [torch.tensor([[[4.0], [3.0], [2.0], [1.0]]]) for _ in range(2)]
    second = [torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]) for _ in range(2)]

    first_due = select_delta_student_topk(student, first, keep_positions, 10, 3, 2)
    second_due = select_delta_student_topk(student, second, keep_positions, 10, 3, 2)

    assert first_due.tolist() == [0, 1]
    assert second_due.tolist() == [2, 3]
    assert all(indices.tolist() == [1, 2, 3] for indices in student.question_indices)


def test_delta_student_topk_falls_back_when_no_question_token_is_kept() -> None:
    student = HiddenScoreStudent()
    hidden = [torch.tensor([[[1.0], [2.0]]])]

    due = select_delta_student_topk(
        student,
        hidden,
        keep_positions=torch.tensor([0, 1]),
        prompt_length=10,
        question_window=2,
        refresh_tokens=1,
    )

    assert due.tolist() == [1]
    assert student.question_indices[0].tolist() == [1]


def test_delta_student_selects_delta_head_from_joint_student() -> None:
    student = TwoHeadStudent()
    hidden = [torch.tensor([[[1.0], [2.0], [3.0]]])]

    due = select_delta_student_topk(
        student,
        hidden,
        keep_positions=torch.tensor([0, 1, 2]),
        prompt_length=3,
        question_window=3,
        refresh_tokens=1,
    )

    assert due.tolist() == [2]
    assert student.selected_heads == ["delta"]
