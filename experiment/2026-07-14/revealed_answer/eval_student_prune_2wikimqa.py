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
    TextTokenizer,
    TokenizedExample,
    load_longbench_samples,
    tokenize_revealed_answer,
)
from revealed_answer.extract_teacher import ExtractConfig, load_model_and_tokenizer
from revealed_answer.oracle_prune import install_oracle_pruner
from revealed_answer.prompt_kv_cache import PromptKVCache, build_prompt_kv_cache
from revealed_answer.prompt_kv_generate import generate_with_prompt_kv
from revealed_answer.student_model import PromptUtilityStudent, StudentConfig
from utils import generate

DEFAULT_STUDENT_PATH = Path(
    "experiment/2026-07-14/results/"
    "revealed_answer_student_train_n1000_e10_lr5e-5/checkpoint-last"
)


class EvalTokenizer(TextTokenizer, Protocol):
    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class EvalConfig:
    model_path: Path
    student_path: Path
    data_path: Path
    output_dir: Path
    limit: int
    max_length: int
    budget: int
    gen_length: int
    block_length: int
    steps: int
    device: str
    dtype: torch.dtype
    question_window: int
    prompt_kv_cache: bool


@dataclass(frozen=True, slots=True)
class EvalRuntime:
    model: torch.nn.Module
    tokenizer: EvalTokenizer
    student: PromptUtilityStudent


def parse_args() -> EvalConfig:
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
    return EvalConfig(
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
                f"[student {index}/{len(samples)}] f1={result['f1']:.5f} "
                f"prompt={result['prompt_length']} pred={result['prediction']!r}",
                flush=True,
            )
    summary = {
        "task": "2wikimqa",
        "student": str(config.student_path),
        "score_source": "student_prompt_hidden_state",
        "budget": config.budget,
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
    student = load_student(config.student_path, config.device)
    return EvalRuntime(model=model, tokenizer=tokenizer, student=student)


def load_student(checkpoint_dir: Path, device: str) -> PromptUtilityStudent:
    config_path = checkpoint_dir / "config.json"
    state_path = checkpoint_dir / "pytorch_model.bin"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    student = PromptUtilityStudent(StudentConfig(**raw_config))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    student.load_state_dict(state)
    student.to(device)
    student.eval()
    print(f"[student] checkpoint={checkpoint_dir}", flush=True)
    return student


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
    student_scores = predict_student_scores(config, runtime, example)
    if config.prompt_kv_cache:
        input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
        prompt_cache = build_prompt_kv_cache(
            runtime.model,
            input_ids,
            budget=config.budget,
            teacher_scores=student_scores,
        )
        prediction = generate_prediction(config, runtime, example, prompt_cache)
    else:
        controller = install_oracle_pruner(
            runtime.model,
            prompt_length=example.prompt_length,
            budget=config.budget,
            teacher_scores=student_scores,
        )
        try:
            prediction = generate_prediction(config, runtime, example, None)
        finally:
            controller.restore()
    score = float(qa_f1_score(prediction, sample.answer))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "sample_id": sample.sample_id,
        "answer": sample.answer,
        "prediction": prediction,
        "f1": score,
        "prompt_length": example.prompt_length,
        "budget": config.budget,
        "prompt_kv_cache": int(config.prompt_kv_cache),
    }


@torch.inference_mode()
def predict_student_scores(
    config: EvalConfig,
    runtime: EvalRuntime,
    example: TokenizedExample,
) -> torch.Tensor:
    input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    out = runtime.model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    prompt_indices = torch.arange(example.prompt_length, dtype=torch.long, device=config.device)
    question_indices = example.question_indices.to(config.device)
    scores: list[torch.Tensor] = []
    for layer_id in runtime.student.layer_indices:
        layer_scores = runtime.student.forward_layer(
            layer_id,
            out.hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        )
        scores.append(torch.softmax(layer_scores.float(), dim=-1).squeeze(0).cpu())
    return torch.stack(scores)


@torch.inference_mode()
def generate_prediction(
    config: EvalConfig,
    runtime: EvalRuntime,
    example: TokenizedExample,
    prompt_cache: PromptKVCache | None,
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


if __name__ == "__main__":
    raise SystemExit(main())
