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

from revealed_answer.attention_teacher import install_answer_attention_collector
from revealed_answer.common import (
    DEFAULT_MODEL_PATH,
    LongBenchSample,
    TextTokenizer,
    TokenizedExample,
    encode_text,
    load_longbench_samples,
    record_output_path,
    tokenize_revealed_answer,
)
from revealed_answer.extract_teacher import load_model_and_tokenizer, parse_dtype
from utils import generate

DEFAULT_TRAIN_DATA: Final = Path(
    "/home/M2026107/dllm/data/train/2wikimultihopqa/"
    "2wikimultihopqa_train_longbench_format.jsonl"
)
DEFAULT_SELF_TEACHER_ROOT: Final = Path(
    "experiment/2026-07-14/results/self_generated_teacher"
)
MASK_ID: Final = 126336


class DecodeTokenizer(TextTokenizer, Protocol):
    def batch_decode(self, sequences: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class SelfTeacherConfig:
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


def parse_args() -> SelfTeacherConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SELF_TEACHER_ROOT)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    return SelfTeacherConfig(
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
    )


def main() -> int:
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_longbench_samples(config.data_path, config.n_samples)
    saved = skipped = 0
    t0 = time.time()
    for index, sample in enumerate(samples, start=1):
        example = tokenize_revealed_answer(
            tokenizer,
            sample,
            config.max_length,
            config.question_window,
            reserve_length=config.gen_length,
        )
        out_path = record_output_path(config.output_root, sample)
        if out_path.exists():
            saved += 1
            continue
        rec = extract_one(model, tokenizer, sample, example, config)
        if rec is None:
            skipped += 1
            print(f"[self-teacher {index}/{len(samples)}] skipped empty generation", flush=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec, out_path)
        saved += 1
        print(
            f"[self-teacher {index}/{len(samples)}] saved={out_path} "
            f"prompt={example.prompt_length} generated={rec['generated_answer']!r}",
            flush=True,
        )
    print(
        f"[done] saved={saved} skipped={skipped} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


@torch.inference_mode()
def extract_one(
    model: torch.nn.Module,
    tokenizer: DecodeTokenizer,
    sample: LongBenchSample,
    example: TokenizedExample,
    config: SelfTeacherConfig,
) -> dict[str, str | int | torch.Tensor] | None:
    generated_answer = generate_answer(model, tokenizer, example, config)
    if not generated_answer:
        return None
    generated_answer_ids = encode_text(tokenizer, " " + generated_answer)
    if not generated_answer_ids:
        return None
    teacher_raw = collect_answer_attention(
        model,
        example.prompt_ids,
        generated_answer_ids,
        config.device,
    )
    teacher_norm = teacher_raw / teacher_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prompt_ids = torch.tensor(example.prompt_ids, dtype=torch.long)
    answer_ids = torch.tensor(generated_answer_ids, dtype=torch.long)
    return {
        "teacher_kind": "self_generated_answer",
        "teacher_formula": "mean_heads(sum_self_answer softmax(Q_self_answer K_prompt^T / sqrt(d)))",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "question": sample.question,
        "answer": sample.answer,
        "generated_answer": generated_answer,
        "prompt_input_ids": prompt_ids,
        "answer_input_ids": answer_ids,
        "prompt_token_indices": torch.arange(example.prompt_length, dtype=torch.long),
        "question_token_indices": example.question_indices,
        "teacher_raw": teacher_raw.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "prompt_length": example.prompt_length,
        "answer_length": len(generated_answer_ids),
        "sequence_length": example.prompt_length + len(generated_answer_ids),
        "max_length": config.max_length,
        "truncation_offset": example.truncation_offset,
    }


def generate_answer(
    model: torch.nn.Module,
    tokenizer: DecodeTokenizer,
    example: TokenizedExample,
    config: SelfTeacherConfig,
) -> str:
    input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    generated = generate(
        input_ids,
        torch.ones_like(input_ids),
        model,
        steps=config.steps,
        gen_length=config.gen_length,
        block_length=config.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        mask_id=MASK_ID,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def collect_answer_attention(
    model: torch.nn.Module,
    prompt_ids: list[int],
    answer_ids: list[int],
    device: str,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    collector = install_answer_attention_collector(
        model,
        prompt_length=len(prompt_ids),
        answer_length=len(answer_ids),
    )
    try:
        model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        return collector.scores_tensor().float()
    finally:
        collector.restore()


if __name__ == "__main__":
    raise SystemExit(main())
