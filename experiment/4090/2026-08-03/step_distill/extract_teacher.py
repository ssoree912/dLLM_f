from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from typing_extensions import assert_never

from .schema import (
    ShardMetadata,
    TeacherShard,
    load_teacher_shard,
    save_teacher_shard_atomic,
    teacher_artifact_path,
)
from .task_data import (
    OffsetTokenizer,
    TeacherSample,
    TokenizedPrompt,
    load_teacher_samples,
    tokenize_2wiki_prompt,
    tokenize_samsum_prompt,
)
from .teacher_generation import (
    PerStepTeacherConfig,
    PerStepTeacherResult,
    generate_per_step_teacher,
)


@dataclass(frozen=True, slots=True)
class ExtractionError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class TeacherTask(str, Enum):
    WIKI_2 = "2wikimqa"
    SAMSUM = "samsum"


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    task: TeacherTask
    model_path: Path
    data_path: Path
    output_root: Path
    device: str
    dtype: torch.dtype
    max_length: int
    sample_limit: int
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    confidence_weight: bool
    max_target_k: int
    diversity_gamma: float
    seed: int
    mask_id: int

    def metadata(self) -> ShardMetadata:
        return ShardMetadata(
            max_length=self.max_length,
            gen_length=self.gen_length,
            steps=self.steps,
            block_length=self.block_length,
            confidence_weight=self.confidence_weight,
            gamma=self.diversity_gamma,
            similarity_source="prompt_prefill_hidden",
            context_timing="pre_step",
            temperature=self.temperature,
            mask_id=self.mask_id,
            model_id=str(self.model_path.resolve()),
            tokenizer_id=str(self.model_path.resolve()),
            torch_dtype=str(self.dtype),
            seed=self.seed,
            source_path=str(self.data_path.resolve()),
        )


def build_teacher_shard(
    sample: TeacherSample,
    prompt: TokenizedPrompt,
    result: PerStepTeacherResult,
    metadata: ShardMetadata,
) -> TeacherShard:
    """Convert one generated trajectory into the validated storage boundary."""
    return TeacherShard(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        prompt_input_ids=torch.tensor(prompt.prompt_ids, dtype=torch.long),
        question_token_indices=prompt.question_indices.to(torch.long),
        generated_input_ids=result.generated_ids.to(torch.long),
        valid_step_mask=result.valid_step_mask.to(torch.bool),
        commit_positions=result.commit_positions.to(torch.long),
        commit_counts=result.commit_counts.to(torch.long),
        commit_confidence=result.commit_confidence.to(torch.float16),
        context_pre=result.context_pre.to(torch.float16),
        top_order=result.top_order.to(torch.long),
        diverse_order=result.diverse_order.to(torch.long),
        candidate_scores=result.candidate_scores.to(torch.float16),
        metadata=metadata,
    )


def run_extraction(config: ExtractConfig) -> int:
    torch.manual_seed(config.seed)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path), trust_remote_code=True
    )
    samples = load_teacher_samples(config.data_path, limit=config.sample_limit)
    teacher_config = PerStepTeacherConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        confidence_weight=config.confidence_weight,
        max_target_k=config.max_target_k,
        diversity_gamma=config.diversity_gamma,
        mask_id=config.mask_id,
    )
    metadata = config.metadata()
    saved = 0
    started = time.monotonic()
    for index, sample in enumerate(samples, start=1):
        output_path = teacher_artifact_path(
            config.output_root, sample.dataset, sample.sample_id
        )
        if output_path.exists():
            existing = load_teacher_shard(
                output_path, expected_sample_id=sample.sample_id
            )
            if existing.metadata != metadata:
                raise ExtractionError(
                    f"existing shard metadata mismatch: {output_path}"
                )
            saved += 1
            continue
        prompt = _tokenize_prompt(tokenizer, sample, config)
        prompt_ids = torch.tensor(
            [prompt.prompt_ids], dtype=torch.long, device=config.device
        )
        result = generate_per_step_teacher(
            model,
            prompt_ids,
            prompt.question_indices,
            teacher_config,
        )
        shard = build_teacher_shard(sample, prompt, result, metadata)
        save_teacher_shard_atomic(shard, output_path)
        saved += 1
        print(
            f"[teacher {index}/{len(samples)}] saved={output_path} "
            f"prompt={len(prompt.prompt_ids)} steps={shard.step_count}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.monotonic() - started:.1f}s", flush=True)
    return saved


def _tokenize_prompt(
    tokenizer: OffsetTokenizer,
    sample: TeacherSample,
    config: ExtractConfig,
) -> TokenizedPrompt:
    match config.task:
        case TeacherTask.WIKI_2:
            return tokenize_2wiki_prompt(
                tokenizer,
                sample,
                max_length=config.max_length,
                reserve_length=config.gen_length,
            )
        case TeacherTask.SAMSUM:
            return tokenize_samsum_prompt(
                tokenizer,
                sample,
                max_length=config.max_length,
                reserve_length=config.gen_length,
            )
        case unreachable:
            assert_never(unreachable)


def parse_args(argv: Sequence[str] | None = None) -> ExtractConfig:
    parser = argparse.ArgumentParser(
        description="Extract causal per-step attention teacher shards."
    )
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in TeacherTask),
        default=TeacherTask.WIKI_2.value,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--uniform-weight", action="store_false", dest="confidence_weight"
    )
    parser.set_defaults(confidence_weight=True)
    parser.add_argument("--max-target-k", type=int, default=512)
    parser.add_argument("--diversity-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--mask-id", type=int, default=126336)
    args = parser.parse_args(argv)
    return ExtractConfig(
        task=TeacherTask(args.task),
        model_path=args.model,
        data_path=args.data,
        output_root=args.output_root,
        device=args.device,
        dtype=_parse_dtype(args.dtype),
        max_length=args.max_length,
        sample_limit=args.sample_limit,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        confidence_weight=args.confidence_weight,
        max_target_k=args.max_target_k,
        diversity_gamma=args.diversity_gamma,
        seed=args.seed,
        mask_id=args.mask_id,
    )


def _parse_dtype(name: str) -> torch.dtype:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name)
    if dtype is None:
        raise ExtractionError(f"unsupported dtype: {name}")
    return dtype


def main(argv: Sequence[str] | None = None) -> int:
    run_extraction(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
