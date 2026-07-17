from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import torch

SCRIPT_ROOT: Final = Path(__file__).resolve().parents[1]
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
for import_root in (SCRIPT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.common import (
    DEFAULT_MODEL_PATH,
    LongBenchSample,
    build_2wikimqa_prompt,
    encode_text,
    load_longbench_samples,
    record_output_path,
)
from revealed_answer.extract_teacher import load_model_and_tokenizer, parse_dtype
from revealed_answer.online_teacher import OnlineTeacherConfig, generate_with_online_teacher

DEFAULT_TRAIN_DATA: Final = Path(
    "/home/M2026107/dllm/data/train/2wikimultihopqa/"
    "2wikimultihopqa_train_longbench_format.jsonl"
)
DEFAULT_OUTPUT_ROOT: Final = Path(
    "experiment/2026-07-14/results/online_self_generated_teacher"
)


class DecodeTokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ...

    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class PromptOnlyExample:
    prompt_text: str
    prompt_ids: list[int]
    question_indices: torch.Tensor
    truncation_offset: int

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)


@dataclass(frozen=True, slots=True)
class ExtractOnlineConfig:
    model_path: Path
    data_path: Path
    output_root: Path
    max_length: int
    n_samples: int
    device: str
    dtype: torch.dtype
    question_window: int
    gen_length: int
    block_length: int
    steps: int
    temperature: float
    confidence_weight: bool


def parse_args() -> ExtractOnlineConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--confidence-weight", action="store_true")
    args = parser.parse_args()
    return ExtractOnlineConfig(
        model_path=args.model,
        data_path=args.data,
        output_root=args.output_root,
        max_length=args.max_length,
        n_samples=args.n_samples,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        question_window=args.question_window,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        confidence_weight=args.confidence_weight,
    )


def main() -> int:
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_longbench_samples(config.data_path, config.n_samples)
    saved = 0
    t0 = time.time()
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
            f"[online-teacher {index}/{len(samples)}] saved={out_path} "
            f"prompt={example.prompt_length} commits={rec['commit_count']} "
            f"generated={rec['generated_answer']!r}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


def tokenize_prompt_only(
    tokenizer: DecodeTokenizer,
    sample: LongBenchSample,
    config: ExtractOnlineConfig,
) -> PromptOnlyExample:
    prompt_text = build_2wikimqa_prompt(sample)
    prompt_ids_full = encode_text(tokenizer, prompt_text)
    prompt_cap = max(1, config.max_length - config.gen_length)
    truncation_offset = max(0, len(prompt_ids_full) - prompt_cap)
    prompt_ids = prompt_ids_full[truncation_offset:]
    question_count = min(max(1, config.question_window), len(prompt_ids))
    question_start = len(prompt_ids) - question_count
    return PromptOnlyExample(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        question_indices=torch.arange(question_start, len(prompt_ids), dtype=torch.long),
        truncation_offset=truncation_offset,
    )


@torch.inference_mode()
def extract_one(
    model: torch.nn.Module,
    tokenizer: DecodeTokenizer,
    sample: LongBenchSample,
    example: PromptOnlyExample,
    config: ExtractOnlineConfig,
) -> dict[str, str | int | float | torch.Tensor]:
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    teacher_config = OnlineTeacherConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        confidence_weight=config.confidence_weight,
    )
    result = generate_with_online_teacher(model, prompt_ids, teacher_config)
    generated_answer = tokenizer.batch_decode(
        result.generated_ids.unsqueeze(0),
        skip_special_tokens=True,
    )[0].strip()
    prompt_tensor = torch.tensor(example.prompt_ids, dtype=torch.long)
    return {
        "teacher_kind": "online_self_generated_prompt_kv",
        "teacher_formula": (
            "sum_commit mean_heads softmax(Q_suffix K_[prompt,suffix]^T / sqrt(d))"
            " sliced_to_prompt"
        ),
        "teacher_graph": "static_full_prompt_kv_B_equals_prompt_length",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "question": sample.question,
        "generated_answer": generated_answer,
        "prompt_input_ids": prompt_tensor,
        "answer_input_ids": result.generated_ids.to(torch.long),
        "generated_answer_input_ids": result.generated_ids.to(torch.long),
        "prompt_token_indices": torch.arange(example.prompt_length, dtype=torch.long),
        "question_token_indices": example.question_indices,
        "teacher_raw": result.teacher_raw.to(torch.float16),
        "teacher_norm": result.teacher_norm.to(torch.float16),
        "prompt_length": example.prompt_length,
        "generated_length": int(result.generated_ids.numel()),
        "sequence_length": example.prompt_length + int(result.generated_ids.numel()),
        "max_length": config.max_length,
        "gen_length": config.gen_length,
        "truncation_offset": example.truncation_offset,
        "commit_count": result.commit_count,
        "confidence_weight_sum": result.confidence_weight_sum,
        "confidence_weight": int(config.confidence_weight),
    }


if __name__ == "__main__":
    raise SystemExit(main())
