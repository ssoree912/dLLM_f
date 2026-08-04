from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .eval_oracle import EvalMethod, generate_method
from .generation_output import llada_stop_token_ids
from .schema import load_teacher_shard
from .summary_oracle_results import (
    aggregate_summary_results,
    build_summary_result,
    load_summary_results,
    save_summary_aggregates_atomic,
    save_summary_result_atomic,
    summary_result_path,
)
from .task_data import load_teacher_samples


@dataclass(frozen=True, slots=True)
class SamsumEvalError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class SamsumEvalConfig:
    model_path: Path
    data_path: Path
    teacher_root: Path
    output_root: Path
    device: str
    dtype: torch.dtype
    limit: int
    budgets: tuple[int, ...]
    methods: tuple[EvalMethod, ...]


def run_samsum_eval(config: SamsumEvalConfig) -> int:
    shard_paths = sorted(config.teacher_root.rglob("*.pt"))
    if config.limit > 0:
        shard_paths = shard_paths[: config.limit]
    if not shard_paths:
        raise SamsumEvalError(
            f"no teacher shards found below {config.teacher_root}"
        )
    samples = {
        sample.sample_id: sample
        for sample in load_teacher_samples(config.data_path)
    }
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
    )
    stop_token_ids = llada_stop_token_ids(tokenizer)
    completed = 0
    for shard_index, shard_path in enumerate(shard_paths, start=1):
        shard = load_teacher_shard(shard_path)
        sample = samples[shard.sample_id]
        prompt_ids = shard.prompt_input_ids.unsqueeze(0).to(config.device)
        for method in config.methods:
            budgets: tuple[int | None, ...] = (
                (None,) if method is EvalMethod.FULL else config.budgets
            )
            for budget in budgets:
                output_path = summary_result_path(
                    config.output_root,
                    method.value,
                    budget,
                    shard_path,
                )
                if output_path.exists():
                    completed += 1
                    continue
                started = time.monotonic()
                generated = generate_method(
                    model,
                    prompt_ids,
                    shard,
                    method,
                    budget,
                )
                matches_teacher = torch.equal(
                    generated.cpu().squeeze(0),
                    shard.generated_input_ids,
                )
                if method is EvalMethod.FULL and not matches_teacher:
                    raise SamsumEvalError(
                        f"full replay diverged from teacher for {shard.sample_id}"
                    )
                result = build_summary_result(
                    tokenizer,
                    generated,
                    stop_token_ids=stop_token_ids,
                    sample_id=shard.sample_id,
                    dataset=shard.dataset,
                    method=method.value,
                    budget=budget,
                    prompt_length=int(shard.prompt_input_ids.numel()),
                    gold=sample.answer,
                    matches_teacher_trajectory=matches_teacher,
                    elapsed_seconds=time.monotonic() - started,
                )
                save_summary_result_atomic(result, output_path)
                completed += 1
                print(
                    f"[samsum {shard_index}/{len(shard_paths)}] "
                    f"method={method.value} budget={budget} "
                    f"rouge_l={result.rouge_l:.3f}",
                    flush=True,
                )
    aggregates = aggregate_summary_results(load_summary_results(config.output_root))
    save_summary_aggregates_atomic(
        aggregates,
        config.output_root / "summary.json",
    )
    return completed


def parse_args(argv: Sequence[str] | None = None) -> SamsumEvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate SAMSum stored masks with word-level ROUGE-L."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=(128, 64, 32))
    parser.add_argument(
        "--methods",
        choices=tuple(method.value for method in EvalMethod),
        nargs="+",
        default=("full", "static_plain", "dynamic_plain"),
    )
    args = parser.parse_args(argv)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    return SamsumEvalConfig(
        model_path=args.model,
        data_path=args.data,
        teacher_root=args.teacher_root,
        output_root=args.output_root,
        device=args.device,
        dtype=dtype,
        limit=args.limit,
        budgets=tuple(args.budgets),
        methods=tuple(EvalMethod(method) for method in args.methods),
    )


def main(argv: Sequence[str] | None = None) -> int:
    run_samsum_eval(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
