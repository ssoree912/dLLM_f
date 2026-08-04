from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from .task_data import OffsetTokenizer, TeacherSample, tokenize_samsum_prompt


@dataclass(frozen=True, slots=True)
class FewshotDataError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class FewshotBuildSpec:
    max_length: int
    reserve_length: int
    train_count: int
    validation_count: int
    seed: int

    def __post_init__(self) -> None:
        if self.max_length <= self.reserve_length:
            raise FewshotDataError("max_length must exceed reserve_length")
        if self.train_count <= 0 or self.validation_count < 0:
            raise FewshotDataError("split counts must include positive training data")

    @property
    def prompt_cap(self) -> int:
        return self.max_length - self.reserve_length


@dataclass(frozen=True, slots=True)
class SamsumTargetSplit:
    train: tuple[TeacherSample, ...]
    validation: tuple[TeacherSample, ...]
    demo_pool: tuple[TeacherSample, ...]


@dataclass(frozen=True, slots=True)
class FewshotPackRequest:
    tokenizer: OffsetTokenizer
    target: TeacherSample
    demo_pool: tuple[TeacherSample, ...]
    spec: FewshotBuildSpec


@dataclass(frozen=True, slots=True)
class PackedSamsumSample:
    teacher_sample: TeacherSample
    source_target_id: str
    demo_ids: tuple[str, ...]
    prompt_length: int
    full_prompt_length: int
    truncation_offset: int


class FewshotSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class SamsumFewshotRecord(TypedDict):
    _id: str
    dataset: str
    context: str
    question: str
    answers: list[str]
    all_classes: None
    answer_prefix: str
    length: int
    language: str
    task: str
    max_new_tokens: int
    source_dataset: str
    source_target_id: str
    source_split: str
    demo_ids: list[str]
    prompt_token_length: int
    full_prompt_token_length: int
    truncation_offset: int
    construction_seed: int


def select_samsum_targets(
    samples: Sequence[TeacherSample],
    spec: FewshotBuildSpec,
) -> SamsumTargetSplit:
    """Select disjoint deterministic targets and a train-only demonstration pool."""
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise FewshotDataError("source sample IDs must be unique")
    target_count = spec.train_count + spec.validation_count
    if len(samples) <= target_count:
        raise FewshotDataError("source data must leave at least one demonstration")
    ordered = list(samples)
    random.Random(spec.seed).shuffle(ordered)
    return SamsumTargetSplit(
        train=tuple(ordered[: spec.train_count]),
        validation=tuple(ordered[spec.train_count : target_count]),
        demo_pool=tuple(ordered[target_count:]),
    )


def pack_samsum_target(request: FewshotPackRequest) -> PackedSamsumSample:
    """Pack train-only demonstrations until the tokenized prompt reaches its cap."""
    target = request.target
    if not request.demo_pool:
        raise FewshotDataError("demo pool must not be empty")
    if any(demo.sample_id == target.sample_id for demo in request.demo_pool):
        raise FewshotDataError("target sample cannot appear in its demonstration pool")
    start = _demo_start(target.sample_id, request.spec.seed, len(request.demo_pool))
    parts: list[str] = []
    demo_ids: list[str] = []
    for offset in range(len(request.demo_pool)):
        demo = request.demo_pool[(start + offset) % len(request.demo_pool)]
        parts.append(f"{demo.context.strip()}\nSummary: {demo.answer.strip()}")
        demo_ids.append(demo.sample_id)
        teacher_sample = TeacherSample(
            sample_id=f"samsum-fewshot-{target.sample_id}",
            dataset="samsum",
            question=f"{target.context.strip()}\nSummary: ",
            context="\n".join(parts),
            answer=target.answer,
        )
        tokenized = tokenize_samsum_prompt(
            request.tokenizer,
            teacher_sample,
            max_length=request.spec.max_length,
            reserve_length=request.spec.reserve_length,
        )
        if (
            len(tokenized.prompt_ids) == request.spec.prompt_cap
            and tokenized.truncation_offset > 0
        ):
            return PackedSamsumSample(
                teacher_sample=teacher_sample,
                source_target_id=target.sample_id,
                demo_ids=tuple(demo_ids),
                prompt_length=len(tokenized.prompt_ids),
                full_prompt_length=(
                    len(tokenized.prompt_ids) + tokenized.truncation_offset
                ),
                truncation_offset=tokenized.truncation_offset,
            )
    raise FewshotDataError(
        f"demo pool could not fill prompt cap for target {target.sample_id!r}"
    )


def packed_samsum_record(
    packed: PackedSamsumSample,
    split: FewshotSplit,
    spec: FewshotBuildSpec,
) -> SamsumFewshotRecord:
    """Convert a packed sample to the existing teacher-loader JSON boundary."""
    sample = packed.teacher_sample
    return SamsumFewshotRecord(
        _id=sample.sample_id,
        dataset=sample.dataset,
        context=sample.context,
        question=sample.question,
        answers=[sample.answer],
        all_classes=None,
        answer_prefix="",
        length=packed.prompt_length,
        language="en",
        task="Few-shot Learning",
        max_new_tokens=spec.reserve_length,
        source_dataset="knkarthick/samsum",
        source_target_id=packed.source_target_id,
        source_split=split.value,
        demo_ids=list(packed.demo_ids),
        prompt_token_length=packed.prompt_length,
        full_prompt_token_length=packed.full_prompt_length,
        truncation_offset=packed.truncation_offset,
        construction_seed=spec.seed,
    )


def _demo_start(sample_id: str, seed: int, pool_size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % pool_size
