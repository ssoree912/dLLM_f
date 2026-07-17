from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from lm_eval.tasks.longbench.metrics import qa_f1_score

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
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
from revealed_answer.extract_teacher import ExtractConfig, load_model_and_tokenizer
from utils import generate


class DecodeTokenizer(Protocol):
    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class EvalConfig:
    model_path: Path
    data_path: Path
    output_dir: Path
    limit: int
    max_length: int
    gen_length: int
    block_length: int
    steps: int
    device: str
    dtype: torch.dtype
    question_window: int


@dataclass(frozen=True, slots=True)
class EvalRuntime:
    model: torch.nn.Module
    tokenizer: DecodeTokenizer


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    args = parser.parse_args()
    return EvalConfig(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        limit=args.limit,
        max_length=args.max_length,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
    )


def parse_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case _:
            raise argparse.ArgumentTypeError(f"unsupported dtype: {dtype}")


def main() -> int:
    config = parse_args()
    runtime = build_runtime(config)
    samples = load_longbench_samples(config.data_path, config.limit)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dLLMCache.new_instance()
    t0 = time.time()
    total = 0.0
    sample_path = config.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            result = run_one_sample(config, runtime, sample)
            total += result["f1"]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[original {index}/{len(samples)}] f1={result['f1']:.5f} "
                f"prompt={result['prompt_length']} pred={result['prediction']!r}",
                flush=True,
            )
    summary = {
        "task": "2wikimqa",
        "model": str(config.model_path),
        "score_source": "original_full_prompt",
        "max_length": config.max_length,
        "gen_length": config.gen_length,
        "samples": len(samples),
        "f1": total / max(1, len(samples)),
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
        data_path=config.data_path,
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
) -> dict[str, str | float | int]:
    example = tokenize_revealed_answer(
        runtime.tokenizer,
        sample,
        config.max_length,
        config.question_window,
        reserve_length=config.gen_length,
    )
    prediction = generate_prediction(config, runtime, example)
    score = float(qa_f1_score(prediction, sample.answer))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "sample_id": sample.sample_id,
        "answer": sample.answer,
        "prediction": prediction,
        "f1": score,
        "prompt_length": example.prompt_length,
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
        gen_length=config.gen_length,
        block_length=config.block_length,
        temperature=0.0,
        cfg_scale=0.0,
    )
    return runtime.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    raise SystemExit(main())
