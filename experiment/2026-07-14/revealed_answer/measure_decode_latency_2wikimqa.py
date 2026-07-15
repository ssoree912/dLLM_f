from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Final

import torch
from lm_eval.tasks.longbench.metrics import qa_f1_score

SCRIPT_ROOT: Final = Path(__file__).resolve().parents[1]
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
for import_root in (SCRIPT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dllm_cache.cache import dLLMCache
from revealed_answer.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    LongBenchSample,
    TokenizedExample,
    load_longbench_samples,
    tokenize_revealed_answer,
)
from revealed_answer.eval_student_prune_2wikimqa import (
    DEFAULT_STUDENT_PATH,
    EvalConfig as StudentEvalConfig,
    EvalRuntime as StudentEvalRuntime,
    load_student,
    predict_student_scores,
)
from revealed_answer.extract_teacher import ExtractConfig, load_model_and_tokenizer, parse_dtype
from revealed_answer.latency_memory_metrics import (
    aggregate,
    estimate_attention_work,
    finish_memory_sample,
    mean,
    start_memory_sample,
    sync_if_cuda,
)
from revealed_answer.oracle_prune import install_oracle_pruner
from revealed_answer.prompt_kv_cache import PromptKVCache, build_prompt_kv_cache
from revealed_answer.prompt_kv_generate import generate_with_prompt_kv
from utils import generate


def parse_args() -> StudentEvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--prompt-kv-cache", action="store_true")
    args = parser.parse_args()
    return StudentEvalConfig(
        model_path=args.model,
        student_path=args.student,
        data_path=args.data,
        output_dir=args.output_dir,
        limit=args.limit,
        max_length=args.max_length,
        budget=args.budget,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
        prompt_kv_cache=args.prompt_kv_cache,
    )


def main() -> int:
    config = parse_args()
    runtime = build_runtime(config)
    samples = load_longbench_samples(config.data_path, config.limit)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dLLMCache.new_instance()

    rows: list[dict[str, str | int | float]] = []
    sample_path = config.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            row = measure_sample(config, runtime, sample)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[latency {index}/{len(samples)}] full={row['full_cache_decode_seconds']:.4f}s "
                  f"student_total={row['student_method_seconds']:.4f}s "
                  f"cache_build={row['student_cache_build_seconds']:.4f}s "
                  f"student_decode={row['student_decode_seconds']:.4f}s", flush=True)

    summary = build_summary(config, rows, sample_path)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {summary}", flush=True)
    return 0


def build_runtime(config: StudentEvalConfig) -> StudentEvalRuntime:
    model_config = ExtractConfig(
        config.model_path, config.data_path, config.output_dir / "teacher_cache_unused",
        config.max_length, config.limit, config.device, config.dtype, config.question_window,
    )
    model, tokenizer = load_model_and_tokenizer(model_config)
    student = load_student(config.student_path, config.device)
    return StudentEvalRuntime(model=model, tokenizer=tokenizer, student=student)


def measure_sample(
    config: StudentEvalConfig,
    runtime: StudentEvalRuntime,
    sample: LongBenchSample,
) -> dict[str, str | int | float]:
    example = tokenize_revealed_answer(runtime.tokenizer, sample, config.max_length,
                                       config.question_window, reserve_length=config.gen_length)
    full_prediction, full_decode, full_peak, full_delta = timed_full_cache(config, runtime, example)
    student_prediction, score_seconds, cache_seconds, student_decode, student_total, \
        student_score_delta, student_cache_delta, student_decode_delta, student_delta = timed_student(
            config,
            runtime,
            example,
        )
    work = estimate_attention_work(
        example.prompt_length,
        config.budget,
        config.gen_length,
        config.steps,
        len(runtime.student.layer_indices),
    )
    return {
        "sample_id": sample.sample_id,
        "prompt_length": example.prompt_length,
        "budget": config.budget,
        "answer": sample.answer,
        "full_cache_prediction": full_prediction,
        "full_cache_f1": float(qa_f1_score(full_prediction, sample.answer)),
        "full_cache_decode_seconds": full_decode,
        "full_cache_peak_mib": full_peak,
        "full_cache_peak_delta_mib": full_delta,
        "student_prediction": student_prediction,
        "student_f1": float(qa_f1_score(student_prediction, sample.answer)),
        "student_score_seconds": score_seconds,
        "student_cache_build_seconds": cache_seconds,
        "student_decode_seconds": student_decode,
        "student_method_seconds": student_total,
        "student_score_peak_delta_mib": student_score_delta,
        "student_cache_build_peak_delta_mib": student_cache_delta,
        "student_decode_peak_delta_mib": student_decode_delta,
        "student_method_peak_delta_mib": student_delta,
        "prompt_kv_cache": int(config.prompt_kv_cache),
        **work,
    }


def timed_full_cache(
    config: StudentEvalConfig,
    runtime: StudentEvalRuntime,
    example: TokenizedExample,
) -> tuple[str, float, float, float]:
    sync_if_cuda(config.device)
    mem_start = start_memory_sample(config.device)
    started = time.perf_counter()
    prediction = generate_text(config, runtime, example)
    sync_if_cuda(config.device)
    memory = finish_memory_sample(mem_start)
    return prediction, time.perf_counter() - started, memory.peak_mib, memory.delta_mib


def timed_student(
    config: StudentEvalConfig,
    runtime: StudentEvalRuntime,
    example: TokenizedExample,
) -> tuple[str, float, float, float, float, float, float, float, float]:
    sync_if_cuda(config.device)
    method_started = time.perf_counter()
    score_memory_start = start_memory_sample(config.device)
    score_started = time.perf_counter()
    scores = predict_student_scores(config, runtime, example)
    sync_if_cuda(config.device)
    score_seconds = time.perf_counter() - score_started
    score_memory = finish_memory_sample(score_memory_start)
    prompt_cache: PromptKVCache | None = None
    cache_seconds = 0.0
    cache_delta = 0.0
    if config.prompt_kv_cache:
        sync_if_cuda(config.device)
        cache_memory_start = start_memory_sample(config.device)
        cache_started = time.perf_counter()
        prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
        prompt_cache = build_prompt_kv_cache(runtime.model, prompt_ids, config.budget, scores)
        sync_if_cuda(config.device)
        cache_seconds = time.perf_counter() - cache_started
        cache_delta = finish_memory_sample(cache_memory_start).delta_mib
        prediction, decode_seconds, decode_delta = timed_decode(config, runtime, example, prompt_cache)
    else:
        controller = install_oracle_pruner(
            runtime.model,
            prompt_length=example.prompt_length,
            budget=config.budget,
            teacher_scores=scores,
        )
        try:
            prediction, decode_seconds, decode_delta = timed_decode(config, runtime, example, None)
        finally:
            controller.restore()
    sync_if_cuda(config.device)
    return (
        prediction,
        score_seconds,
        cache_seconds,
        decode_seconds,
        time.perf_counter() - method_started,
        score_memory.delta_mib,
        cache_delta,
        decode_delta,
        max(score_memory.delta_mib, cache_delta, decode_delta),
    )


def timed_decode(
    config: StudentEvalConfig,
    runtime: StudentEvalRuntime,
    example: TokenizedExample,
    prompt_cache: PromptKVCache | None,
) -> tuple[str, float, float]:
    sync_if_cuda(config.device)
    decode_memory_start = start_memory_sample(config.device)
    decode_started = time.perf_counter()
    prediction = generate_text(config, runtime, example, prompt_cache)
    sync_if_cuda(config.device)
    decode_seconds = time.perf_counter() - decode_started
    decode_memory = finish_memory_sample(decode_memory_start)
    return prediction, decode_seconds, decode_memory.delta_mib


@torch.inference_mode()
def generate_text(
    config: StudentEvalConfig,
    runtime: StudentEvalRuntime,
    example: TokenizedExample,
    prompt_cache: PromptKVCache | None = None,
) -> str:
    input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    if prompt_cache is None:
        generated = generate(
            input_ids,
            torch.ones_like(input_ids),
            runtime.model,
            steps=config.steps,
            gen_length=config.gen_length,
            block_length=config.block_length,
            temperature=0.0,
            cfg_scale=0.0,
        )
    else:
        generated = generate_with_prompt_kv(
            input_ids,
            runtime.model,
            prompt_cache,
            steps=config.steps,
            gen_length=config.gen_length,
            block_length=config.block_length,
            temperature=0.0,
            cfg_scale=0.0,
        )
    return runtime.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def build_summary(
    config: StudentEvalConfig,
    rows: list[dict[str, str | int | float]],
    sample_path: Path,
) -> dict[str, str | int | float | dict[str, int | float]]:
    return {
        "task": "2wikimqa",
        "samples": len(rows),
        "budget": config.budget,
        "max_length": config.max_length,
        "gen_length": config.gen_length,
        "steps": config.steps,
        "full_cache_decode_seconds": aggregate(rows, "full_cache_decode_seconds"),
        "student_score_seconds": aggregate(rows, "student_score_seconds"),
        "student_cache_build_seconds": aggregate(rows, "student_cache_build_seconds"),
        "student_decode_seconds": aggregate(rows, "student_decode_seconds"),
        "student_method_seconds": aggregate(rows, "student_method_seconds"),
        "full_cache_peak_delta_mib": aggregate(rows, "full_cache_peak_delta_mib"),
        "student_score_peak_delta_mib": aggregate(rows, "student_score_peak_delta_mib"),
        "student_cache_build_peak_delta_mib": aggregate(rows, "student_cache_build_peak_delta_mib"),
        "student_decode_peak_delta_mib": aggregate(rows, "student_decode_peak_delta_mib"),
        "student_method_peak_delta_mib": aggregate(rows, "student_method_peak_delta_mib"),
        "prompt_kv_keep_ratio": aggregate(rows, "prompt_kv_keep_ratio"),
        "attention_qk_keep_ratio": aggregate(rows, "attention_qk_keep_ratio"),
        "attention_qk_reduction_ratio": aggregate(rows, "attention_qk_reduction_ratio"),
        "full_attention_qk_elements": aggregate(rows, "full_attention_qk_elements"),
        "student_attention_qk_elements": aggregate(rows, "student_attention_qk_elements"),
        "full_cache_f1": mean(rows, "full_cache_f1"),
        "student_f1": mean(rows, "student_f1"),
        "student": str(config.student_path),
        "sample_file": str(sample_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
