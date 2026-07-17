from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Protocol

import torch

try:
    from lm_eval.tasks.longbench.metrics import code_sim_score, qa_f1_score
except ModuleNotFoundError:

    def qa_f1_score(prediction: str, ground_truth: str) -> float:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        if not prediction_tokens or not ground_truth_tokens:
            return float(prediction_tokens == ground_truth_tokens)
        common = set(prediction_tokens) & set(ground_truth_tokens)
        overlap = sum(min(prediction_tokens.count(token), ground_truth_tokens.count(token)) for token in common)
        if overlap == 0:
            return 0.0
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def code_sim_score(prediction: str, ground_truth: str) -> float:
        return SequenceMatcher(None, prediction.strip(), ground_truth.strip()).ratio()

    def normalize_answer(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^0-9a-z]+", " ", text)
        return " ".join(text.split())

REPO_ROOT = Path(__file__).resolve().parents[3]
REVEALED_ROOT = REPO_ROOT / "experiment/2026-07-14"
PROJECT_ROOT = REPO_ROOT.parent
for import_root in (REVEALED_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dllm_cache.cache import dLLMCache
from revealed_answer.extract_teacher import ExtractConfig, load_model_and_tokenizer
from revealed_answer.oracle_prune import install_oracle_pruner
from revealed_answer.prompt_kv_cache import PromptKVCache, build_prompt_kv_cache
from revealed_answer.prompt_kv_generate import generate_with_prompt_kv
from revealed_answer.student_model import PromptUtilityStudent, StudentConfig
from revealed_answer_345.dynamic_prompt_kv import (
    DynamicPromptKVCache,
    build_dynamic_prompt_kv_cache,
    generate_with_dynamic_prompt_kv,
)
from utils.generate_function import generate


class EvalTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        ...

    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


MetricFn = Callable[[str, str], float]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    data_file: str
    gen_length: int
    metric_name: str
    metric_fn: MetricFn


TASKS: dict[str, TaskSpec] = {
    "2wikimqa": TaskSpec(
        name="2wikimqa",
        data_file="2wikimqa.jsonl",
        gen_length=32,
        metric_name="qa_f1_score",
        metric_fn=qa_f1_score,
    ),
    "multifieldqa_en": TaskSpec(
        name="multifieldqa_en",
        data_file="multifieldqa_en.jsonl",
        gen_length=64,
        metric_name="qa_f1_score",
        metric_fn=qa_f1_score,
    ),
    "lcc": TaskSpec(
        name="lcc",
        data_file="lcc.jsonl",
        gen_length=64,
        metric_name="code_sim_score",
        metric_fn=code_sim_score,
    ),
}


@dataclass(frozen=True, slots=True)
class LongBenchSample:
    sample_id: str
    task: str
    question: str
    context: str
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    prompt_ids: list[int]
    answer_ids: list[int]
    question_indices: torch.Tensor
    truncation_offset: int

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    task: TaskSpec
    model_path: Path
    student_path: Path
    data_dir: Path
    output_dir: Path
    limit: int
    max_length: int
    budget: int
    block_length: int
    steps: int
    device: str
    dtype: torch.dtype
    question_window: int
    prompt_kv_cache: bool
    prompt_kv_mode: str
    prompt_refresh_interval: int
    prompt_selection_mode: str


@dataclass(frozen=True, slots=True)
class EvalRuntime:
    model: torch.nn.Module
    tokenizer: EvalTokenizer
    student: PromptUtilityStudent


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model/LLaDA-8B-Instruct")
    parser.add_argument(
        "--student",
        type=Path,
        default=(
            REPO_ROOT
            / "experiment/2026-07-14/results/"
            / "revealed_answer_student_train_n5000_topk128_e5_lr5e-5_tw0.02/checkpoint-best"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/longbench")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--prompt-kv-cache", action="store_true")
    parser.add_argument(
        "--prompt-kv-mode",
        choices=["static", "refresh", "dynamic"],
        default="static",
        help="Prompt-KV decode mode. Existing --prompt-kv-cache defaults to static.",
    )
    parser.add_argument(
        "--prompt-refresh-interval",
        type=int,
        default=0,
        help="Refresh interval for dynamic prompt K/V. Positive values imply refresh mode.",
    )
    parser.add_argument(
        "--prompt-selection-mode",
        choices=["layer_union", "global"],
        default="layer_union",
        help="Prompt token set for dynamic/refresh modes. global gives an exact B+R reduced sequence.",
    )
    args = parser.parse_args()
    task = TASKS[args.task]
    prompt_kv_mode = args.prompt_kv_mode
    if args.prompt_refresh_interval > 0 and prompt_kv_mode == "static":
        prompt_kv_mode = "refresh"
    prompt_kv_cache = (
        bool(args.prompt_kv_cache)
        or prompt_kv_mode != "static"
        or args.prompt_refresh_interval > 0
    )
    if prompt_kv_mode == "dynamic":
        prompt_refresh_interval = 1
    elif prompt_kv_mode == "refresh":
        prompt_refresh_interval = args.prompt_refresh_interval or 8
    else:
        prompt_refresh_interval = 0
    return EvalConfig(
        task=task,
        model_path=args.model,
        student_path=args.student,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        max_length=args.max_length,
        budget=args.budget,
        block_length=args.block_length,
        steps=args.steps or task.gen_length,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
        prompt_kv_cache=prompt_kv_cache,
        prompt_kv_mode=prompt_kv_mode,
        prompt_refresh_interval=prompt_refresh_interval,
        prompt_selection_mode=args.prompt_selection_mode,
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
                f"[student {config.task.name} {index}/{len(samples)}] "
                f"{config.task.metric_name}={result['score']:.5f} "
                f"prompt={result['prompt_length']} pred={result['prediction']!r}",
                flush=True,
            )
    summary = {
        "task": config.task.name,
        "student": str(config.student_path),
        "score_source": "student_prompt_hidden_state",
        "budget": config.budget,
        "max_length": config.max_length,
        "gen_length": config.task.gen_length,
        "steps": config.steps,
        "block_length": config.block_length,
        "samples": len(samples),
        "metric": config.task.metric_name,
        "score": score_sum / max(1, len(samples)),
        "prompt_kv_cache": config.prompt_kv_cache,
        "prompt_kv_mode": decode_mode_name(config),
        "prompt_refresh_interval": config.prompt_refresh_interval,
        "prompt_selection_mode": config.prompt_selection_mode,
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


def load_samples(config: EvalConfig) -> list[LongBenchSample]:
    path = config.data_dir / config.task.data_file
    samples: list[LongBenchSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if config.limit > 0 and len(samples) >= config.limit:
                break
            row = json.loads(line)
            samples.append(parse_sample(config.task.name, row, index))
    return samples


def parse_sample(task: str, row: dict[str, object], index: int) -> LongBenchSample:
    sample_id = row.get("_id", f"sample_{index}")
    question = row.get("question", "")
    context = row.get("context")
    answers = parse_answers(row.get("answers"))
    if not isinstance(sample_id, str):
        sample_id = f"sample_{index}"
    if not isinstance(question, str):
        raise RuntimeError(f"row {index} has non-string question")
    if not isinstance(context, str):
        raise RuntimeError(f"row {index} has non-string context")
    return LongBenchSample(
        sample_id=sample_id,
        task=task,
        question=question,
        context=context,
        answers=tuple(answers),
    )


def parse_answers(value: object) -> list[str]:
    match value:
        case [*items] if items and all(isinstance(item, str) for item in items):
            return list(items)
        case str() as answer:
            return [answer]
        case _:
            raise RuntimeError("answers must be a string or a non-empty string list")


def build_prompt(sample: LongBenchSample) -> str:
    if sample.task == "2wikimqa":
        instruction = (
            "Answer the question based on the given passages. Only give me the answer "
            "and do not output any other words."
        )
        return (
            f"{instruction}\n\n"
            f"The following are given passages.\n{sample.context}\n\n"
            f"{instruction}\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    if sample.task == "multifieldqa_en":
        return (
            "Read the following text and answer briefly.\n\n"
            f"{sample.context}\n\n"
            "Now, answer the following question based on the above text, only give me "
            "the answer and do not output any other words.\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    if sample.task == "lcc":
        return "Please complete the code given below. \n" + sample.context + "Next line of code:"
    raise RuntimeError(f"unsupported task: {sample.task}")


def encode_text(tokenizer: EvalTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def tokenize_sample(
    tokenizer: EvalTokenizer,
    sample: LongBenchSample,
    max_length: int,
    question_window: int,
    reserve_length: int,
) -> TokenizedExample:
    prompt_ids_full = encode_text(tokenizer, build_prompt(sample))
    answer_ids = encode_text(tokenizer, " " + sample.answers[0].strip())
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[: max(1, max_length // 4)]
    prompt_cap = max(1, max_length - max(len(answer_ids), reserve_length))
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    prompt_ids = prompt_ids_full[truncation_offset:]
    question_count = min(max(1, question_window), len(prompt_ids))
    question_start = len(prompt_ids) - question_count
    question_indices = torch.arange(question_start, len(prompt_ids), dtype=torch.long)
    return TokenizedExample(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        question_indices=question_indices,
        truncation_offset=truncation_offset,
    )


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
    student_scores = predict_student_scores(config, runtime, example)
    reduced_prompt_tokens: int | None = None
    if config.prompt_kv_cache:
        input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
        if config.prompt_kv_mode == "static":
            prompt_cache = build_prompt_kv_cache(
                runtime.model,
                input_ids,
                budget=config.budget,
                teacher_scores=student_scores,
            )
        else:
            prompt_cache = build_dynamic_prompt_kv_cache(
                runtime.model,
                input_ids,
                budget=config.budget,
                teacher_scores=student_scores,
                selection_mode=config.prompt_selection_mode,
            )
            reduced_prompt_tokens = prompt_cache.reduced_prompt_length
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
        "budget": config.budget,
        "prompt_kv_cache": config.prompt_kv_cache,
        "prompt_kv_mode": decode_mode_name(config),
        "prompt_refresh_interval": config.prompt_refresh_interval,
        "prompt_selection_mode": config.prompt_selection_mode,
        "reduced_prompt_tokens": reduced_prompt_tokens,
    }


def decode_mode_name(config: EvalConfig) -> str:
    if not config.prompt_kv_cache:
        return "oracle_prune"
    return config.prompt_kv_mode


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
    prompt_cache: PromptKVCache | DynamicPromptKVCache | None,
) -> str:
    input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    if prompt_cache is None:
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
    elif isinstance(prompt_cache, DynamicPromptKVCache):
        generated = generate_with_dynamic_prompt_kv(
            input_ids,
            runtime.model,
            prompt_cache,
            steps=config.steps,
            gen_length=config.task.gen_length,
            block_length=config.block_length,
            refresh_interval=config.prompt_refresh_interval,
            temperature=0.0,
            cfg_scale=0.0,
        )
    else:
        generated = generate_with_prompt_kv(
            input_ids,
            runtime.model,
            prompt_cache,
            steps=config.steps,
            gen_length=config.task.gen_length,
            block_length=config.block_length,
            temperature=0.0,
            cfg_scale=0.0,
        )
    return runtime.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    raise SystemExit(main())
