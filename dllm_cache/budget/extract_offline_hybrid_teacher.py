# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers"]
# ///

"""Extract offline reference-attention and prompt-K/V-delta teacher shards."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from dllm_cache.budget.balanced_teacher import BalancedSample, DecodeTokenizer, PromptOnlyExample
from dllm_cache.budget.common import DEFAULT_MODEL_PATH, record_output_path
from dllm_cache.budget.extract_future_pool_teacher_balanced import (
    load_filtered_samples,
    tokenize_prompt_only,
)
from dllm_cache.budget.extract_teacher import load_model_and_tokenizer, parse_dtype
from dllm_cache.budget.offline_hybrid_teacher import (
    OfflineHybridTeacherConfig,
    generate_with_offline_hybrid_teacher,
)


DEFAULT_TRAIN_DATA: Final = Path(
    "/home/M2026107/dllm/data/train/samsum/samsum_train_longbench_format.jsonl"
)
DEFAULT_OUTPUT_ROOT: Final = Path("results/budget/offline_hybrid_teacher_samsum_20260810")


@dataclass(frozen=True, slots=True)
class ExtractOfflineHybridConfig:
    model_path: Path
    data_path: Path
    output_root: Path
    datasets: tuple[str, ...]
    max_length: int
    n_samples: int
    samples_per_dataset: int
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
    apply_chat_template: bool
    fewshot_context_chars: int


def parse_args(argv: Sequence[str] | None = None) -> ExtractOfflineHybridConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["samsum"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=0)
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
    parser.add_argument(
        "--prompt-format",
        choices=["train", "samsum-eval-fewshot", "longbench-local"],
        default="train",
    )
    parser.add_argument("--fewshot-context-chars", type=int, default=35000)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="match the prompt wrapping used by the inference harness",
    )
    args = parser.parse_args(argv)
    return ExtractOfflineHybridConfig(
        model_path=args.model,
        data_path=args.data,
        output_root=args.output_root,
        datasets=tuple(args.datasets),
        max_length=args.max_length,
        n_samples=args.n_samples,
        samples_per_dataset=args.samples_per_dataset,
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
        apply_chat_template=args.apply_chat_template,
        fewshot_context_chars=args.fewshot_context_chars,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
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
            f"[offline-hybrid-teacher {index}/{len(samples)}] saved={out_path} "
            f"dataset={sample.dataset} prompt={example.prompt_length} "
            f"ref_union_mean={rec['ref_union_size_mean']:.1f} "
            f"delta_mean={rec['delta_raw'].float().mean().item():.6f}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - started:.1f}s", flush=True)
    return 0


@torch.inference_mode()
def extract_one(
    model: torch.nn.Module,
    tokenizer: DecodeTokenizer,
    sample: BalancedSample,
    example: PromptOnlyExample,
    config: ExtractOfflineHybridConfig,
) -> dict[str, str | int | float | torch.Tensor]:
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    teacher_config = OfflineHybridTeacherConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        active_top_k=config.active_top_k,
        temperature=config.temperature,
        confidence_weight=config.confidence_weight,
        target_aggregation=config.target_aggregation,
    )
    result = generate_with_offline_hybrid_teacher(model, prompt_ids, teacher_config)
    generated_answer = tokenizer.batch_decode(
        result.generated_ids.unsqueeze(0),
        skip_special_tokens=True,
    )[0].strip()
    prompt_tensor = torch.tensor(example.prompt_ids, dtype=torch.long)
    reference_formula = (
        f"reference={config.target_aggregation}_step_committed_suffix_to_prompt_attention_dense_all_prompt; "
        if config.active_top_k <= 0
        else f"reference={config.target_aggregation}_step_committed_suffix_to_prompt_attention_"
        "masked_by_temporal_union_topk; "
    )
    return {
        "teacher_kind": "offline_hybrid_ref_delta",
        "teacher_formula": reference_formula
        + "delta=sum_t mean(K_relative_stepwise,V_relative_stepwise)",
        "teacher_graph": "full_sequence_prompt_suffix_single_forward_ref_and_delta",
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
        "ref_union_mask": result.ref_union_mask,
        "ref_union_size_by_layer": result.ref_union_size_by_layer,
        "ref_union_size_mean": float(result.ref_union_size_by_layer.float().mean().item()),
        "delta_raw": result.delta_raw.to(torch.float16),
        "delta_norm": result.delta_norm.to(torch.float16),
        "delta_step_max": result.delta_step_max.to(torch.float16),
        "delta_observation_count": result.delta_observation_count,
        "prompt_length": example.prompt_length,
        "generated_length": int(result.generated_ids.numel()),
        "sequence_length": example.prompt_length + int(result.generated_ids.numel()),
        "max_length": config.max_length,
        "gen_length": config.gen_length,
        "block_length": config.block_length,
        "steps": config.steps,
        "active_top_k": config.active_top_k,
        "target_aggregation": config.target_aggregation,
        "prompt_format": config.prompt_format,
        "apply_chat_template": int(config.apply_chat_template),
        "fewshot_context_chars": config.fewshot_context_chars,
        "truncation_offset": example.truncation_offset,
        "reference_step_count": result.reference_step_count,
        "commit_count": result.commit_count,
        "confidence_weight_sum": result.weight_sum,
        "confidence_weight": int(config.confidence_weight),
    }


if __name__ == "__main__":
    raise SystemExit(main())
