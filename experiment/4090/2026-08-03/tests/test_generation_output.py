from __future__ import annotations

import torch

from step_distill.generation_output import decode_generation, llada_stop_token_ids


class TokenIdTokenizer:
    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in sequences]


class LLadaTokenizer(TokenIdTokenizer):
    eos_token_id = 91

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|eot_id|>"
        return 90


def test_llada_stop_token_ids_include_eos_and_end_of_turn() -> None:
    # Given / When
    stop_token_ids = llada_stop_token_ids(LLadaTokenizer())

    # Then
    assert stop_token_ids == frozenset({90, 91})


def test_decode_generation_removes_the_first_stop_and_all_trailing_tokens() -> None:
    # Given
    generated = torch.tensor([[10, 11, 90, 12, 91, 13]], dtype=torch.long)

    # When
    decoded = decode_generation(
        TokenIdTokenizer(),
        generated,
        stop_token_ids=frozenset({90, 91}),
    )

    # Then
    assert decoded.prediction == "10 11"
    assert decoded.raw_prediction == "10 11 90 12 91 13"
    assert decoded.first_stop_position == 2
    assert decoded.first_stop_token_id == 90
    assert decoded.tokens_before_stop == 2
    assert decoded.trailing_token_count == 3


def test_decode_generation_preserves_the_full_canvas_without_a_stop() -> None:
    # Given
    generated = torch.tensor([[10, 11, 12]], dtype=torch.long)

    # When
    decoded = decode_generation(
        TokenIdTokenizer(),
        generated,
        stop_token_ids=frozenset({90, 91}),
    )

    # Then
    assert decoded.prediction == "10 11 12"
    assert decoded.raw_prediction == decoded.prediction
    assert decoded.first_stop_position is None
    assert decoded.first_stop_token_id is None
    assert decoded.tokens_before_stop == 3
    assert decoded.trailing_token_count == 0


def test_decode_generation_allows_an_empty_answer_before_the_first_stop() -> None:
    # Given
    generated = torch.tensor([[90, 10, 11]], dtype=torch.long)

    # When
    decoded = decode_generation(
        TokenIdTokenizer(),
        generated,
        stop_token_ids=frozenset({90}),
    )

    # Then
    assert decoded.prediction == ""
    assert decoded.first_stop_position == 0
    assert decoded.tokens_before_stop == 0
    assert decoded.trailing_token_count == 2
