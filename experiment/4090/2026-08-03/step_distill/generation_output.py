from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class BatchDecodeTokenizer(Protocol):
    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]: ...


class LLadaDecodeTokenizer(BatchDecodeTokenizer, Protocol):
    eos_token_id: int | None

    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass(frozen=True, slots=True)
class GenerationOutputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class DecodedGeneration:
    prediction: str
    raw_prediction: str
    canvas_token_count: int
    tokens_before_stop: int
    first_stop_position: int | None
    first_stop_token_id: int | None
    trailing_token_count: int


def llada_stop_token_ids(tokenizer: LLadaDecodeTokenizer) -> frozenset[int]:
    """Resolve both sequence-level and chat-turn stop IDs for LLaDA-Instruct."""
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    candidates = {end_of_turn_id}
    if tokenizer.eos_token_id is not None:
        candidates.add(tokenizer.eos_token_id)
    valid = frozenset(token_id for token_id in candidates if token_id >= 0)
    if not valid:
        raise GenerationOutputError("LLaDA tokenizer has no usable stop token IDs")
    return valid


def decode_generation(
    tokenizer: BatchDecodeTokenizer,
    generated: torch.Tensor,
    *,
    stop_token_ids: frozenset[int],
) -> DecodedGeneration:
    """Decode one fixed canvas after removing its first stop and trailing positions."""
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise GenerationOutputError("generated must have shape [1, generation_length]")
    if generated.shape[1] == 0:
        raise GenerationOutputError("generated canvas must not be empty")
    if not stop_token_ids:
        raise GenerationOutputError("stop_token_ids must not be empty")

    row = generated[0].detach().cpu()
    stop_position = next(
        (
            position
            for position, token_id in enumerate(row.tolist())
            if token_id in stop_token_ids
        ),
        None,
    )
    canvas_token_count = int(row.numel())
    tokens_before_stop = (
        canvas_token_count if stop_position is None else stop_position
    )
    truncated = generated[:, :tokens_before_stop]
    raw_prediction = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()
    prediction = tokenizer.batch_decode(
        truncated,
        skip_special_tokens=True,
    )[0].strip()
    stop_token_id = (
        None if stop_position is None else int(row[stop_position].item())
    )
    trailing_token_count = (
        0
        if stop_position is None
        else canvas_token_count - stop_position - 1
    )
    return DecodedGeneration(
        prediction=prediction,
        raw_prediction=raw_prediction,
        canvas_token_count=canvas_token_count,
        tokens_before_stop=tokens_before_stop,
        first_stop_position=stop_position,
        first_stop_token_id=stop_token_id,
        trailing_token_count=trailing_token_count,
    )
