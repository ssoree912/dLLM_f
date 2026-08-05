from __future__ import annotations

import torch
from step_distill.distribution_artifacts import result_record
from step_distill.distribution_rollout import DistributionRolloutResult
from step_distill.distribution_step import DistributionStepReport


class FakeTokenizer:
    eos_token_id = 90

    def convert_tokens_to_ids(self, token: str) -> int:
        return 91 if token == "<|eot_id|>" else -1

    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        del skip_special_tokens
        special = {90, 91}
        return [
            " ".join(str(value) for value in row.tolist() if value not in special)
            for row in sequences
        ]


def test_complete_result_records_raw_and_stop_diagnostics() -> None:
    # Given
    report = DistributionStepReport(0, 1.0, 0.8, 0.7, 2.0, 3.0, 0.5, 1.0)
    result = DistributionRolloutResult(
        generated_ids=torch.tensor([[10, 90, 12, 13]]),
        reports=(report,),
        complete=True,
    )

    # When
    record = result_record(
        result,
        split="validation",
        sample_id="sample-1",
        epoch=1,
        gold="10",
        tokenizer=FakeTokenizer(),
    )

    # Then
    assert record["prediction"] == "10"
    assert record["raw_prediction"] == "10 12 13"
    assert record["rouge_l_f1"] == 1.0
    assert record["raw_rouge_l_f1"] == 0.5
    assert record["first_stop_position"] == 1
    assert record["tokens_before_stop"] == 1
    assert record["trailing_token_count"] == 2
    assert record["canvas_token_count"] == 4
