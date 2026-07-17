from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

SCRIPT_ROOT: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
REVEALED_ROOT: Final = REPO_ROOT / "experiment/2026-07-14"
for import_root in (SCRIPT_ROOT, REVEALED_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.common import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LongBenchSample,
    build_2wikimqa_prompt,
    encode_text,
    load_longbench_samples,
    record_output_path,
)
from revealed_answer.extract_teacher import load_model_and_tokenizer, parse_dtype  # noqa: E402
from revealed_answer_345.full_dynamic_trajectory_teacher import (  # noqa: E402
    FullDynamicTrajectoryConfig,
    decode_generated_answer,
    generate_with_full_dynamic_trajectory_teacher,
)


DEFAULT_TRAIN_DATA: Final = Path(
    "/mnt/srv/home/dlpcg.325/dllm/data/train/2wikimultihopqa/"
    "2wikimultihopqa_train_longbench_format.jsonl"
)


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
class ExtractConfig:
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
    max_weight: float
    confidence_weight: bool
    skip_sample_ids: frozenset[str]
    skip_indices: frozenset[int]


def parse_args() -> ExtractConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.5)
    parser.add_argument("--confidence-weight", action="store_true")
    parser.add_argument(
        "--skip-sample-ids",
        default="",
        help="Comma-separated sample ids to skip before extraction.",
    )
    parser.add_argument(
        "--skip-indices",
        default="",
        help="Comma-separated 1-based sample indices to skip before extraction.",
    )
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
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        max_weight=args.max_weight,
        confidence_weight=args.confidence_weight,
        skip_sample_ids=frozenset(_split_csv(args.skip_sample_ids)),
        skip_indices=frozenset(int(item) for item in _split_csv(args.skip_indices)),
    )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_longbench_samples(config.data_path, config.n_samples)
    saved = 0
    t0 = time.time()
    for index, sample in enumerate(samples, start=1):
        if index in config.skip_indices or sample.sample_id in config.skip_sample_ids:
            print(
                f"[skip {index}/{len(samples)}] sample_id={sample.sample_id}",
                flush=True,
            )
            continue
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
            f"[full-dynamic-teacher {index}/{len(samples)}] saved={out_path} "
            f"prompt={example.prompt_length} commits={rec['commit_count']} "
            f"generated={rec['generated_answer']!r}",
            flush=True,
        )
    print(f"[done] saved={saved} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


def tokenize_prompt_only(
    tokenizer,
    sample: LongBenchSample,
    config: ExtractConfig,
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
    tokenizer,
    sample: LongBenchSample,
    example: PromptOnlyExample,
    config: ExtractConfig,
) -> dict[str, str | int | float | torch.Tensor]:
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    teacher_config = FullDynamicTrajectoryConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        max_weight=config.max_weight,
        confidence_weight=config.confidence_weight,
    )
    result = generate_with_full_dynamic_trajectory_teacher(model, prompt_ids, teacher_config)
    generated_answer = decode_generated_answer(tokenizer, result.generated_ids)
    prompt_tensor = torch.tensor(example.prompt_ids, dtype=torch.long)
    return {
        "teacher_kind": "full_dynamic_trajectory",
        "teacher_formula": (
            "max_weight * max_commit(mean_heads softmax(Q_suffix K_full^T)_prompt) "
            "+ (1 - max_weight) * mean_commit(mean_heads softmax(Q_suffix K_full^T)_prompt)"
        ),
        "teacher_graph": "full_prompt_plus_suffix_forward_each_denoising_step",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "question": sample.question,
        "answer": sample.answer,
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
        "weight_sum": result.weight_sum,
        "max_weight": config.max_weight,
        "confidence_weight": int(config.confidence_weight),
    }


if __name__ == "__main__":
    raise SystemExit(main())
