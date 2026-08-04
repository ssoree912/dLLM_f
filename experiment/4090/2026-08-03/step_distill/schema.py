from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

import torch

from .artifact_paths import teacher_artifact_path as _teacher_artifact_path

SCHEMA_VERSION: Final = 2
TEACHER_KIND: Final = "causal_step_distill_v1"


@dataclass(frozen=True, slots=True)
class SchemaError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ShardMetadata:
    max_length: int
    gen_length: int
    steps: int
    block_length: int
    confidence_weight: bool
    gamma: float
    similarity_source: str
    context_timing: str
    temperature: float = 0.0
    mask_id: int = 126336
    model_id: str = ""
    tokenizer_id: str = ""
    torch_dtype: str = ""
    seed: int = 0
    source_path: str = ""


class MetadataRecord(TypedDict):
    max_length: int
    gen_length: int
    steps: int
    block_length: int
    confidence_weight: bool
    gamma: float
    similarity_source: str
    context_timing: str
    temperature: float
    mask_id: int
    model_id: str
    tokenizer_id: str
    torch_dtype: str
    seed: int
    source_path: str


class TeacherShardRecord(TypedDict):
    schema_version: int
    teacher_kind: str
    sample_id: str
    dataset: str
    prompt_input_ids: torch.Tensor
    question_token_indices: torch.Tensor
    generated_input_ids: torch.Tensor
    valid_step_mask: torch.Tensor
    commit_positions: torch.Tensor
    commit_counts: torch.Tensor
    commit_confidence: torch.Tensor
    context_pre: torch.Tensor
    top_order: torch.Tensor
    diverse_order: torch.Tensor
    candidate_scores: torch.Tensor
    metadata: MetadataRecord


@dataclass(frozen=True, slots=True)
class TeacherShard:
    sample_id: str
    dataset: str
    prompt_input_ids: torch.Tensor
    question_token_indices: torch.Tensor
    generated_input_ids: torch.Tensor
    valid_step_mask: torch.Tensor
    commit_positions: torch.Tensor
    commit_counts: torch.Tensor
    commit_confidence: torch.Tensor
    context_pre: torch.Tensor
    top_order: torch.Tensor
    diverse_order: torch.Tensor
    candidate_scores: torch.Tensor
    metadata: ShardMetadata

    def __post_init__(self) -> None:
        _validate_shard(self)

    @property
    def step_count(self) -> int:
        return int(self.valid_step_mask.numel())

    def to_record(self) -> TeacherShardRecord:
        return TeacherShardRecord(
            schema_version=SCHEMA_VERSION,
            teacher_kind=TEACHER_KIND,
            sample_id=self.sample_id,
            dataset=self.dataset,
            prompt_input_ids=self.prompt_input_ids.cpu(),
            question_token_indices=self.question_token_indices.cpu(),
            generated_input_ids=self.generated_input_ids.cpu(),
            valid_step_mask=self.valid_step_mask.cpu(),
            commit_positions=self.commit_positions.cpu(),
            commit_counts=self.commit_counts.cpu(),
            commit_confidence=self.commit_confidence.cpu(),
            context_pre=self.context_pre.cpu(),
            top_order=self.top_order.cpu(),
            diverse_order=self.diverse_order.cpu(),
            candidate_scores=self.candidate_scores.cpu(),
            metadata=MetadataRecord(**asdict(self.metadata)),
        )


def save_teacher_shard_atomic(shard: TeacherShard, path: Path) -> None:
    """Write, reload, and validate a shard before atomically publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(shard.to_record(), temporary_path)
        load_teacher_shard(temporary_path, expected_sample_id=shard.sample_id)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_teacher_shard(
    path: Path, *, expected_sample_id: str | None = None
) -> TeacherShard:
    """Load a tensor-only schema-v2 shard and reject stale or malformed content."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict):
        raise SchemaError(f"shard must be a mapping, got {type(record).__name__}")
    shard = _shard_from_record(record)
    if expected_sample_id is not None and shard.sample_id != expected_sample_id:
        raise SchemaError(
            f"sample id mismatch: expected {expected_sample_id!r}, got {shard.sample_id!r}"
        )
    return shard


def teacher_artifact_path(output_root: Path, dataset: str, sample_id: str) -> Path:
    return _teacher_artifact_path(output_root, dataset, sample_id)


def _shard_from_record(record) -> TeacherShard:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema version: {record.get('schema_version')!r}"
        )
    if record.get("teacher_kind") != TEACHER_KIND:
        raise SchemaError(f"unsupported teacher kind: {record.get('teacher_kind')!r}")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise SchemaError(f"metadata must be a mapping, got {type(metadata).__name__}")
    parsed_metadata = ShardMetadata(**metadata)
    try:
        return TeacherShard(
            sample_id=record["sample_id"],
            dataset=record["dataset"],
            prompt_input_ids=record["prompt_input_ids"],
            question_token_indices=record["question_token_indices"],
            generated_input_ids=record["generated_input_ids"],
            valid_step_mask=record["valid_step_mask"],
            commit_positions=record["commit_positions"],
            commit_counts=record["commit_counts"],
            commit_confidence=record["commit_confidence"],
            context_pre=record["context_pre"],
            top_order=record["top_order"],
            diverse_order=record["diverse_order"],
            candidate_scores=record["candidate_scores"],
            metadata=parsed_metadata,
        )
    except KeyError as error:
        raise SchemaError(f"missing shard field: {error.args[0]}") from error


def _validate_shard(shard: TeacherShard) -> None:
    if not shard.sample_id or not shard.dataset:
        raise SchemaError("sample_id and dataset must be non-empty")
    if shard.prompt_input_ids.ndim != 1 or shard.prompt_input_ids.dtype != torch.long:
        raise SchemaError("prompt_input_ids must be a one-dimensional long tensor")
    if (
        shard.question_token_indices.ndim != 1
        or shard.question_token_indices.numel() == 0
    ):
        raise SchemaError("question_token_indices must be a non-empty vector")
    if shard.valid_step_mask.ndim != 1 or shard.valid_step_mask.dtype != torch.bool:
        raise SchemaError("valid_step_mask must be a one-dimensional bool tensor")
    step_count = int(shard.valid_step_mask.numel())
    step_tensors = (
        shard.commit_positions,
        shard.commit_counts,
        shard.commit_confidence,
        shard.context_pre,
        shard.top_order,
        shard.diverse_order,
        shard.candidate_scores,
    )
    if any(tensor.shape[0] != step_count for tensor in step_tensors):
        raise SchemaError("all trajectory tensors must share the step axis")
    if shard.context_pre.ndim != 3:
        raise SchemaError("context_pre must have shape [step, layer, hidden]")
    if shard.top_order.ndim != 3 or shard.diverse_order.shape != shard.top_order.shape:
        raise SchemaError("top and diverse orders must share shape [step, layer, k]")
    if shard.candidate_scores.shape != shard.top_order.shape:
        raise SchemaError("candidate_scores must align with top_order")
    if (
        shard.commit_positions.ndim != 2
        or shard.commit_confidence.shape != shard.commit_positions.shape
    ):
        raise SchemaError("commit tensors must share shape [step, max_commit]")
    if shard.commit_counts.shape != (step_count,):
        raise SchemaError("commit_counts must have shape [step]")
    if shard.metadata.steps != step_count:
        raise SchemaError("metadata steps do not match the stored step axis")
    if shard.metadata.gen_length != shard.generated_input_ids.numel():
        raise SchemaError("metadata gen_length does not match generated_input_ids")
    if int(shard.commit_counts.sum()) != shard.metadata.gen_length:
        raise SchemaError("commit counts do not cover the generated suffix")
    if not torch.equal(shard.valid_step_mask, shard.commit_counts > 0):
        raise SchemaError(
            "valid_step_mask must identify steps with committed positions"
        )
    prompt_length = int(shard.prompt_input_ids.numel())
    if (
        shard.question_token_indices.min() < 0
        or shard.question_token_indices.max() >= prompt_length
    ):
        raise SchemaError("question indices fall outside the prompt")
    if shard.top_order.min() < 0 or shard.top_order.max() >= prompt_length:
        raise SchemaError("top order contains an invalid prompt index")
    if shard.diverse_order.min() < 0 or shard.diverse_order.max() >= prompt_length:
        raise SchemaError("diverse order contains an invalid prompt index")
    max_commits = shard.commit_positions.shape[1]
    for step_id, count_tensor in enumerate(shard.commit_counts):
        count = int(count_tensor)
        if count < 0 or count > max_commits:
            raise SchemaError("commit count falls outside the padded commit width")
        if count > 0:
            active = shard.commit_positions[step_id, :count]
            if active.min() < 0 or active.max() >= shard.metadata.gen_length:
                raise SchemaError("commit position falls outside the generated suffix")
        if not torch.all(shard.commit_positions[step_id, count:] == -1):
            raise SchemaError("commit position padding must be -1")
    for tensor in (shard.commit_confidence, shard.context_pre, shard.candidate_scores):
        if not torch.isfinite(tensor).all():
            raise SchemaError("floating-point shard tensors must be finite")
