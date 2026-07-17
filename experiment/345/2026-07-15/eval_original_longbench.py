from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
REVEALED_ROOT = REPO_ROOT / "experiment/2026-07-14"
PROJECT_ROOT = REPO_ROOT.parent
for import_root in (SCRIPT_ROOT, REVEALED_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dllm_cache.cache import dLLMCache
from eval_student_prune_longbench import (
    TASKS,
    EvalTokenizer,
    LongBenchSample,
    TaskSpec,
    TokenizedExample,
    load_samples,
    parse_dtype,
    tokenize_sample,
)
from revealed_answer.extract_teacher import ExtractConfig, load_model_and_tokenizer
from utils.generate_function import generate


@dataclass(frozen=True, slots=True)
class EvalConfig:
    task: TaskSpec
    model_path: Path
    data_dir: Path
    output_dir: Path
    limit: int
    max_length: int
    block_length: int
    steps: int
    device: str
    dtype: torch.dtype
    question_window: int


@dataclass(frozen=True, slots=True)
class EvalRuntime:
    model: torch.nn.Module
    tokenizer: EvalTokenizer


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model/LLaDA-8B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/longbench")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    args = parser.parse_args()
    task = TASKS[args.task]
    return EvalConfig(
        task=task,
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        max_length=args.max_length,
        block_length=args.block_length,
        steps=args.steps or task.gen_length,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
    )


def main() -> int:
    config = parse_args()
    runtime = build_runtime(config)
    samples = load_samples(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dLLMCache.new_instance()
    t0 = time.time()
    score_sum = 0.0
    sample_path = config.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            result = run_one_sample(config, runtime, sample)
            score_sum += float(result["score"])
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[original {config.task.name} {index}/{len(samples)}] "
                f"{config.task.metric_name}={result['score']:.5f} "
                f"prompt={result['prompt_length']} pred={result['prediction']!r}",
                flush=True,
            )
    summary = {
        "task": config.task.name,
        "model": str(config.model_path),
        "score_source": "original_full_prompt",
        "max_length": config.max_length,
        "gen_length": config.task.gen_length,
        "steps": config.steps,
        "block_length": config.block_length,
        "samples": len(samples),
        "metric": config.task.metric_name,
        "score": score_sum / max(1, len(samples)),
        "elapsed": time.time() - t0,
        "sample_file": str(sample_path),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {summary}", flush=True)
    return 0


def build_runtime(config: EvalConfig) -> EvalRuntime:
    model_config = ExtractConfig(
        model_path=config.model_path,
        data_path=config.data_dir / config.task.data_file,
        output_root=config.output_dir / "teacher_cache_unused",
        max_length=config.max_length,
        n_samples=config.limit,
        device=config.device,
        dtype=config.dtype,
        question_window=config.question_window,
    )
    model, tokenizer = load_model_and_tokenizer(model_config)
    return EvalRuntime(model=model, tokenizer=tokenizer)


def run_one_sample(
    config: EvalConfig,
    runtime: EvalRuntime,
    sample: LongBenchSample,
) -> dict[str, object]:
    example = tokenize_sample(
        runtime.tokenizer,
        sample,
        config.max_length,
        config.question_window,
        reserve_length=config.task.gen_length,
    )
    prediction = generate_prediction(config, runtime, example)
    score = max(config.task.metric_fn(prediction, answer) for answer in sample.answers)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "answers": list(sample.answers),
        "prediction": prediction,
        "metric": config.task.metric_name,
        "score": float(score),
        "prompt_length": example.prompt_length,
        "truncation_offset": example.truncation_offset,
    }


@torch.inference_mode()
def generate_prediction(
    config: EvalConfig,
    runtime: EvalRuntime,
    example: TokenizedExample,
) -> str:
    input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    generated = generate(
        input_ids,
        torch.ones_like(input_ids),
        runtime.model,
        steps=config.steps,
        gen_length=config.task.gen_length,
        block_length=config.block_length,
        temperature=0.0,
        cfg_scale=0.0,
    )
    return runtime.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    raise SystemExit(main())
