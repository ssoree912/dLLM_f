from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.attention_teacher import install_answer_attention_collector
from dllm_cache.budget.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEACHER_ROOT,
    LongBenchSample,
    TokenizedExample,
    load_longbench_samples,
    record_output_path,
    tokenize_revealed_answer,
)


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    model_path: Path
    data_path: Path
    output_root: Path
    max_length: int
    n_samples: int
    device: str
    dtype: torch.dtype
    question_window: int


def parse_args() -> ExtractConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    args = parser.parse_args()
    return ExtractConfig(
        model_path=args.model,
        data_path=args.data,
        output_root=args.output_root,
        max_length=args.max_length,
        n_samples=args.n_samples,
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
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_longbench_samples(config.data_path, config.n_samples)
    t0 = time.time()
    saved = 0
    for index, sample in enumerate(samples, start=1):
        example = tokenize_revealed_answer(
            tokenizer,
            sample,
            max_length=config.max_length,
            question_window=config.question_window,
        )
        out_path = record_output_path(config.output_root, sample)
        if out_path.exists():
            saved += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rec = extract_one(model, sample, example, config)
        torch.save(rec, out_path)
        saved += 1
        print(
            f"[teacher {index}/{len(samples)}] saved={out_path} "
            f"prompt={example.prompt_length} answer={len(example.answer_ids)}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


def load_model_and_tokenizer(config: ExtractConfig):
    print(f"[load] model={config.model_path} dtype={config.dtype} device={config.device}", flush=True)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    return model, tokenizer


@torch.inference_mode()
def extract_one(
    model: torch.nn.Module,
    sample: LongBenchSample,
    example: TokenizedExample,
    config: ExtractConfig,
) -> dict[str, str | int | torch.Tensor]:
    input_ids = torch.tensor(
        [example.prompt_ids + example.answer_ids],
        dtype=torch.long,
        device=config.device,
    )
    attention_mask = torch.ones_like(input_ids)
    collector = install_answer_attention_collector(
        model,
        prompt_length=example.prompt_length,
        answer_length=len(example.answer_ids),
    )
    try:
        model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        teacher_raw = collector.scores_tensor().float()
    finally:
        collector.restore()
    teacher_norm = teacher_raw / teacher_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prompt_ids = torch.tensor(example.prompt_ids, dtype=torch.long)
    answer_ids = torch.tensor(example.answer_ids, dtype=torch.long)
    return {
        "teacher_kind": "gold_revealed_answer",
        "teacher_formula": "mean_heads(sum_answer softmax(Q_answer K_prompt^T / sqrt(d)))",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "question": sample.question,
        "answer": sample.answer,
        "prompt_input_ids": prompt_ids,
        "answer_input_ids": answer_ids,
        "prompt_token_indices": torch.arange(example.prompt_length, dtype=torch.long),
        "question_token_indices": example.question_indices,
        "teacher_raw": teacher_raw.to(torch.float16),
        "teacher_norm": teacher_norm.to(torch.float16),
        "prompt_length": example.prompt_length,
        "answer_length": len(example.answer_ids),
        "sequence_length": example.sequence_length,
        "max_length": config.max_length,
        "truncation_offset": example.truncation_offset,
    }


if __name__ == "__main__":
    raise SystemExit(main())
