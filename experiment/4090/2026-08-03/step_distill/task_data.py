from __future__ import annotations

import json
import lzma
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict

import torch
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

INSTRUCTION = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words."
)
SAMSUM_INSTRUCTION = (
    "Summarize the dialogue into a few short sentences. "
    "The following are some examples."
)
AnswerValue: TypeAlias = str | list[str]
ANSWER_ADAPTER = TypeAdapter(AnswerValue)


class OffsetEncoding(TypedDict):
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> OffsetEncoding: ...


@dataclass(frozen=True, slots=True)
class DataError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class TeacherSample:
    sample_id: str
    dataset: str
    question: str
    context: str
    answer: str


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    prompt_text: str
    prompt_ids: list[int]
    question_indices: torch.Tensor
    truncation_offset: int


class SamsumStateSpan(str, Enum):
    SHARED_INSTRUCTION = "shared_instruction"
    TARGET_REQUEST = "target_request"


def load_teacher_samples(path: Path, *, limit: int = 0) -> list[TeacherSample]:
    """Load original-task training rows represented in the local unified JSONL schema."""
    samples: list[TeacherSample] = []
    with (
        lzma.open(path, mode="rt", encoding="utf-8")
        if path.suffix == ".xz"
        else path.open("r", encoding="utf-8")
    ) as handle:
        for index, line in enumerate(handle):
            if limit > 0 and len(samples) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(
                    f"row {index} is not valid JSON: {error.msg}"
                ) from error
            samples.append(_parse_sample(row, index))
    return samples


def tokenize_2wiki_prompt(
    tokenizer: OffsetTokenizer,
    sample: TeacherSample,
    *,
    max_length: int,
    reserve_length: int,
) -> TokenizedPrompt:
    """Tokenize once and map the exact query character span through left truncation."""
    if max_length <= reserve_length:
        raise DataError("max_length must exceed reserve_length")
    question = _normalize_question(sample.question)
    prefix = (
        f"{INSTRUCTION}\n\n"
        f"The following are given passages.\n{sample.context.strip()}\n\n"
        f"{INSTRUCTION}\n\n"
        "Question: "
    )
    suffix = "\nAnswer:"
    prompt_text = f"{prefix}{question}{suffix}"
    question_start = len(prefix)
    question_end = question_start + len(question)
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids_full = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(prompt_ids_full):
        raise DataError("tokenizer offset mapping does not align with input ids")
    prompt_cap = max_length - reserve_length
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    prompt_ids = prompt_ids_full[truncation_offset:]
    retained_question = [
        token_index - truncation_offset
        for token_index, (start, end) in enumerate(offsets)
        if token_index >= truncation_offset
        and start < question_end
        and end > question_start
    ]
    if not retained_question:
        raise DataError("left truncation removed the complete question span")
    return TokenizedPrompt(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        question_indices=torch.tensor(retained_question, dtype=torch.long),
        truncation_offset=truncation_offset,
    )


def tokenize_samsum_prompt(
    tokenizer: OffsetTokenizer,
    sample: TeacherSample,
    *,
    max_length: int,
    reserve_length: int,
    state_span: SamsumStateSpan = SamsumStateSpan.SHARED_INSTRUCTION,
) -> TokenizedPrompt:
    """Use the LongBench template and explicitly choose the static state span."""
    if max_length <= reserve_length:
        raise DataError("max_length must exceed reserve_length")
    dialogue = sample.context.strip()
    request = sample.question.strip()
    prompt_text = f"{SAMSUM_INSTRUCTION}\n\n{dialogue}\n\n{request}"
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids_full = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(prompt_ids_full):
        raise DataError("tokenizer offset mapping does not align with input ids")
    prompt_cap = max_length - reserve_length
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    prompt_ids = prompt_ids_full[truncation_offset:]
    match state_span:
        case SamsumStateSpan.SHARED_INSTRUCTION:
            span_indices = _retained_instruction_indices(
                offsets,
                truncation_offset,
                prompt_text,
                request,
            )
        case SamsumStateSpan.TARGET_REQUEST:
            request_start = len(prompt_text) - len(request)
            span_indices = [
                token_index - truncation_offset
                for token_index, (start, end) in enumerate(offsets)
                if token_index >= truncation_offset
                and start < len(prompt_text)
                and end > request_start
            ]
        case unreachable:
            assert_never(unreachable)
    if not span_indices:
        raise DataError("left truncation removed the selected SAMSum state span")
    return TokenizedPrompt(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        question_indices=torch.tensor(span_indices, dtype=torch.long),
        truncation_offset=truncation_offset,
    )


def _retained_instruction_indices(
    offsets: list[tuple[int, int]],
    truncation_offset: int,
    prompt_text: str,
    request: str,
) -> list[int]:
    instruction_end = len(SAMSUM_INSTRUCTION)
    retained = [
        token_index - truncation_offset
        for token_index, (start, end) in enumerate(offsets)
        if token_index >= truncation_offset and start < instruction_end and end > 0
    ]
    if retained:
        return retained
    request_start = len(prompt_text) - len(request)
    return [
        token_index - truncation_offset
        for token_index, (start, end) in enumerate(offsets)
        if token_index >= truncation_offset
        and start < len(prompt_text)
        and end > request_start
    ]


def _normalize_question(question: str) -> str:
    normalized = question.strip()
    prefix = "Question:"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :].strip()
    return normalized


def _parse_sample(row, index: int) -> TeacherSample:
    try:
        sample_id = row.get("_id", f"sample_{index}")
        dataset = row.get("dataset", "2wikimultihopqa_train")
        question = row.get("question", row.get("input"))
        context = row["context"]
        raw_answers = row["answers"]
    except (AttributeError, KeyError) as error:
        raise DataError(f"row {index} is missing required fields") from error
    try:
        answers = ANSWER_ADAPTER.validate_python(raw_answers)
    except ValidationError as error:
        raise DataError(f"row {index} has invalid answers") from error
    match answers:
        case [first, *_]:
            answer = first
        case str() as value:
            answer = value
        case []:
            raise DataError(f"row {index} has no answers")
        case unreachable:
            assert_never(unreachable)
    if not all(
        isinstance(value, str) for value in (sample_id, dataset, question, context)
    ):
        raise DataError(f"row {index} contains a non-string required field")
    return TeacherSample(sample_id, dataset, question, context, answer)
