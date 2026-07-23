from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import torch

INSTRUCTION: Final = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words."
)
DEFAULT_DATA_PATH: Final = Path("/home/M2026107/dllm/data/longbench/2wikimqa.jsonl")
DEFAULT_MODEL_PATH: Final = Path("/home/M2026107/dllm/model/LLaDA-8B-Instruct")
DEFAULT_TEACHER_ROOT: Final = Path("results/budget/revealed_answer_teacher")


class TextTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        ...


@dataclass(frozen=True, slots=True)
class LongBenchSample:
    sample_id: str
    question: str
    context: str
    answer: str
    dataset: str


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    prompt_text: str
    answer_text: str
    prompt_ids: list[int]
    answer_ids: list[int]
    question_indices: torch.Tensor
    truncation_offset: int

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_ids) + len(self.answer_ids)


class SampleParseError(RuntimeError):
    pass


def build_2wikimqa_prompt(sample: LongBenchSample) -> str:
    return (
        f"{INSTRUCTION}\n\n"
        f"The following are given passages.\n{sample.context}\n\n"
        f"{INSTRUCTION}\n\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )


def load_longbench_samples(path: Path, limit: int) -> list[LongBenchSample]:
    samples: list[LongBenchSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit > 0 and len(samples) >= limit:
                break
            samples.append(parse_longbench_sample(json.loads(line), index))
    return samples


def parse_longbench_sample(row: dict[str, object], index: int) -> LongBenchSample:
    answer = parse_answer(row.get("answers"))
    sample_id = row.get("_id", f"sample_{index}")
    question = row.get("question")
    context = row.get("context")
    dataset = row.get("dataset", "2wikimqa")
    if not isinstance(sample_id, str):
        raise SampleParseError(f"row {index} has non-string _id")
    if not isinstance(question, str):
        raise SampleParseError(f"row {index} has non-string question")
    if not isinstance(context, str):
        raise SampleParseError(f"row {index} has non-string context")
    if not isinstance(dataset, str):
        raise SampleParseError(f"row {index} has non-string dataset")
    return LongBenchSample(
        sample_id=sample_id,
        question=question,
        context=context,
        answer=answer,
        dataset=dataset,
    )


def parse_answer(value: object) -> str:
    match value:
        case [str() as first, *_]:
            return first
        case str() as answer:
            return answer
        case _:
            raise SampleParseError("answers must be a string or a non-empty string list")


def encode_text(tokenizer: TextTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def tokenize_revealed_answer(
    tokenizer: TextTokenizer,
    sample: LongBenchSample,
    max_length: int,
    question_window: int,
    reserve_length: int = 0,
) -> TokenizedExample:
    prompt_text = build_2wikimqa_prompt(sample)
    answer_text = " " + sample.answer.strip()
    answer_ids = encode_text(tokenizer, answer_text)
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[: max(1, max_length // 4)]
    prompt_ids_full = encode_text(tokenizer, prompt_text)
    prompt_cap = max(1, max_length - max(len(answer_ids), reserve_length))
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    prompt_ids = prompt_ids_full[truncation_offset:]
    question_count = min(max(1, question_window), len(prompt_ids))
    question_start = len(prompt_ids) - question_count
    question_indices = torch.arange(question_start, len(prompt_ids), dtype=torch.long)
    return TokenizedExample(
        prompt_text=prompt_text,
        answer_text=answer_text,
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        question_indices=question_indices,
        truncation_offset=truncation_offset,
    )


def record_output_path(output_root: Path, sample: LongBenchSample) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample.sample_id)
    return output_root / sample.dataset / f"{safe_id[:160]}.pt"
