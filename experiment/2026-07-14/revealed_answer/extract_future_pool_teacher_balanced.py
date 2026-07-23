# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers"]
# ///

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import torch

SCRIPT_ROOT: Final = Path(__file__).resolve().parents[1]
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
EXPERIMENT_345_ROOT: Final = REPO_ROOT / "experiment" / "345" / "2026-07-15"
for import_root in (SCRIPT_ROOT, EXPERIMENT_345_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.common import DEFAULT_MODEL_PATH, record_output_path
from revealed_answer.extract_online_self_teacher_balanced import (
    BalancedSample,
    DecodeTokenizer,
    parse_balanced_sample,
    tokenize_prompt_only,
)
from revealed_answer.extract_teacher import load_model_and_tokenizer, parse_dtype
from revealed_answer.future_pool_teacher import FuturePoolTeacherConfig, generate_with_future_pool_teacher

DEFAULT_TRAIN_DATA: Final = Path(
    "/home/M2026107/dllm/data/train_balanced_2k/all_selected_train_longbench_format.jsonl"
)
DEFAULT_OUTPUT_ROOT: Final = Path("experiment/2026-07-14/results/future_pool_teacher_balanced_g128_top128")


@dataclass(frozen=True, slots=True)
class ExtractFuturePoolConfig:
    model_path: Path
    data_path: Path
    output_root: Path
    datasets: tuple[str, ...]
    max_length: int
    n_samples: int
    device: str
    dtype: torch.dtype
    question_window: int
    gen_length: int
    block_length: int
    steps: int
    active_top_k: int
    temperature: float
    confidence_weight: bool
    target_aggregation: str
    min_raw_length: int
    skip_matching_samples: int
    prompt_format: str
    fewshot_context_chars: int


def parse_args() -> ExtractFuturePoolConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["samsum"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--active-top-k", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--confidence-weight", action="store_true")
    parser.add_argument("--target-aggregation", choices=["max", "sum"], default="max")
    parser.add_argument("--min-raw-length", type=int, default=0)
    parser.add_argument("--skip-matching-samples", type=int, default=0)
    parser.add_argument("--prompt-format", choices=["train", "samsum-eval-fewshot"], default="train")
    parser.add_argument("--fewshot-context-chars", type=int, default=35000)
    args = parser.parse_args()
    return ExtractFuturePoolConfig(
        model_path=args.model,
        data_path=args.data,
        output_root=args.output_root,
        datasets=tuple(args.datasets),
        max_length=args.max_length,
        n_samples=args.n_samples,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        active_top_k=args.active_top_k,
        temperature=args.temperature,
        confidence_weight=args.confidence_weight,
        target_aggregation=args.target_aggregation,
        min_raw_length=args.min_raw_length,
        skip_matching_samples=args.skip_matching_samples,
        prompt_format=args.prompt_format,
        fewshot_context_chars=args.fewshot_context_chars,
    )


def main() -> int:
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_filtered_samples(config)
    saved = 0
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        out_path = record_output_path(config.output_root, sample)
        if out_path.exists():
            saved += 1
            continue
        example = tokenize_prompt_only(tokenizer, sample, config)
        rec = extract_one(model, tokenizer, sample, example, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, out_path)
        saved += 1
        print(
            f"[future-pool-teacher {index}/{len(samples)}] saved={out_path} "
            f"dataset={sample.dataset} prompt={example.prompt_length} "
            f"union_mean={rec['union_size_mean']:.1f}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - started:.1f}s", flush=True)
    return 0


def load_filtered_samples(config: ExtractFuturePoolConfig) -> list[BalancedSample]:
    candidates: list[BalancedSample] = []
    dataset_filter = set(config.datasets)
    with config.data_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("length") or 0) < config.min_raw_length:
                continue
            sample = parse_balanced_sample(row, index)
            if sample.dataset not in dataset_filter:
                continue
            candidates.append(sample)
    selected = candidates[config.skip_matching_samples :]
    if config.n_samples > 0:
        selected = selected[: config.n_samples]
    if not selected:
        raise RuntimeError(f"no samples selected for datasets={sorted(dataset_filter)}")
    return format_prompt_samples(selected, candidates, config)


def format_prompt_samples(
    selected: Sequence[BalancedSample],
    candidates: Sequence[BalancedSample],
    config: ExtractFuturePoolConfig,
) -> list[BalancedSample]:
    match config.prompt_format:
        case "train":
            return list(selected)
        case "samsum-eval-fewshot":
            if any(sample.dataset != "samsum" for sample in selected):
                raise RuntimeError("samsum-eval-fewshot prompt format only supports samsum")
            return build_samsum_eval_fewshot_samples(selected, candidates, config.fewshot_context_chars)
        case _:
            raise RuntimeError(f"unsupported prompt format: {config.prompt_format}")


def build_samsum_eval_fewshot_samples(
    selected: Sequence[BalancedSample],
    candidates: Sequence[BalancedSample],
    context_chars: int,
) -> list[BalancedSample]:
    if context_chars <= 0:
        raise RuntimeError("fewshot-context-chars must be positive")
    demos = [sample for sample in candidates if sample.dataset == "samsum"]
    if len(demos) < 2:
        raise RuntimeError("samsum-eval-fewshot requires at least two samsum samples")
    return [with_samsum_eval_fewshot_context(sample, demos, context_chars) for sample in selected]


def with_samsum_eval_fewshot_context(
    sample: BalancedSample,
    demos: Sequence[BalancedSample],
    context_chars: int,
) -> BalancedSample:
    parts: list[str] = []
    start = next((index for index, demo in enumerate(demos) if demo.sample_id == sample.sample_id), -1) + 1
    for offset in range(len(demos)):
        demo = demos[(start + offset) % len(demos)]
        if demo.sample_id == sample.sample_id:
            continue
        parts.append(f"{demo.context.strip()}\nSummary: {demo.answers[0].strip()}")
        if sum(len(part) for part in parts) >= context_chars:
            break
    return BalancedSample(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        task=sample.task,
        context="\n".join(parts),
        question=f"{sample.context.strip()}\nSummary: ",
        answers=sample.answers,
        answer_prefix=sample.answer_prefix,
    )


@torch.inference_mode()
def extract_one(
    model: torch.nn.Module,
    tokenizer: DecodeTokenizer,
    sample: BalancedSample,
    example,
    config: ExtractFuturePoolConfig,
) -> dict[str, str | int | float | torch.Tensor]:
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    teacher_config = FuturePoolTeacherConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        active_top_k=config.active_top_k,
        temperature=config.temperature,
        confidence_weight=config.confidence_weight,
        target_aggregation=config.target_aggregation,
    )
    result = generate_with_future_pool_teacher(model, prompt_ids, teacher_config)
    generated_answer = tokenizer.batch_decode(result.generated_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    prompt_tensor = torch.tensor(example.prompt_ids, dtype=torch.long)
    return {
        "teacher_kind": "future_temporal_union_pool",
        "teacher_formula": (
            "mask_union_t top_active_k(sum_committed_suffix_attention_t); "
            "future_frequency_i=count_t(i in top_active_k)/active_update_steps"
        ),
        "teacher_graph": "full_sequence_prompt_suffix",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "task": sample.task,
        "question": sample.question,
        "answers": json.dumps(sample.answers, ensure_ascii=False),
        "generated_answer": generated_answer,
        "prompt_input_ids": prompt_tensor,
        "answer_input_ids": result.generated_ids.to(torch.long),
        "generated_answer_input_ids": result.generated_ids.to(torch.long),
        "prompt_token_indices": torch.arange(example.prompt_length, dtype=torch.long),
        "question_token_indices": example.question_indices,
        "teacher_raw": result.teacher_raw.to(torch.float16),
        "teacher_norm": result.teacher_norm.to(torch.float16),
        "future_frequency": result.future_frequency.to(torch.float16),
        "future_frequency_count": result.future_frequency_count,
        "future_union_mask": result.union_mask,
        "future_union_size_by_layer": result.union_size_by_layer,
        "union_size_mean": float(result.union_size_by_layer.float().mean().item()),
        "frequency_denominator": result.frequency_denominator,
        "prompt_length": example.prompt_length,
        "generated_length": int(result.generated_ids.numel()),
        "sequence_length": example.prompt_length + int(result.generated_ids.numel()),
        "max_length": config.max_length,
        "gen_length": config.gen_length,
        "active_top_k": config.active_top_k,
        "target_aggregation": config.target_aggregation,
        "prompt_format": config.prompt_format,
        "fewshot_context_chars": config.fewshot_context_chars,
        "truncation_offset": example.truncation_offset,
        "commit_count": result.commit_count,
        "confidence_weight_sum": result.weight_sum,
        "confidence_weight": int(config.confidence_weight),
    }


if __name__ == "__main__":
    raise SystemExit(main())
