from __future__ import annotations

import re


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute word-level ROUGE-L F1 from the longest common subsequence."""
    predicted = re.findall(r"\w+", prediction.lower())
    reference = re.findall(r"\w+", ground_truth.lower())
    if not predicted or not reference:
        return float(predicted == reference)
    lcs = _lcs_length(predicted, reference)
    if lcs == 0:
        return 0.0
    precision = lcs / len(predicted)
    recall = lcs / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_recall(prediction: str, ground_truth: str) -> float:
    """Compute word-level LCS recall without a generated-length penalty."""
    predicted = re.findall(r"\w+", prediction.lower())
    reference = re.findall(r"\w+", ground_truth.lower())
    if not reference:
        return float(not predicted)
    return _lcs_length(predicted, reference) / len(reference)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]
