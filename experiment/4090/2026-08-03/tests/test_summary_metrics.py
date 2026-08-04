from __future__ import annotations

from step_distill.summary_metrics import rouge_l_score


def test_rouge_l_is_one_for_identical_summary() -> None:
    # Given / When
    score = rouge_l_score("Alice met Bob.", "Alice met Bob.")

    # Then
    assert score == 1.0


def test_rouge_l_penalizes_trailing_unrelated_words() -> None:
    # Given / When
    score = rouge_l_score("Alice met Bob and unrelated noise", "Alice met Bob")

    # Then
    assert score == 2 * 3 / (6 + 3)
