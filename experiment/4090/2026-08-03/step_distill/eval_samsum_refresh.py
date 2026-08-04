from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from typing_extensions import assert_never

from .eval_oracle import _generation_config
from .generation_output import llada_stop_token_ids
from .refresh_cache import (
    AttentionPolicy,
    RefreshCacheConfig,
    RefreshPromptKVCache,
    build_refresh_targets,
)
from .refresh_cache_forward import LladaRefreshRunner
from .refresh_cache_generation import generate_with_refresh_cache
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


class RefreshEvalMethod(str, Enum):
    CACHE_ALL = "refresh4_cache_all"
    CACHE_TOPK = "refresh4_cache_dynamic_top128"


@dataclass(frozen=True, slots=True)
class RefreshEvalError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class RefreshEvalConfig:
    model_path: Path
    data_path: Path
    teacher_root: Path
    output_root: Path
    device: str
    dtype: torch.dtype
    limit: int
    cache_budget: int
    attention_budget: int
    refresh_interval: int
    methods: tuple[RefreshEvalMethod, ...]


def run_refresh_eval(config: RefreshEvalConfig) -> int:
    shard_paths = sorted(config.teacher_root.rglob("*.pt"))
    if config.limit > 0:
        shard_paths = shard_paths[: config.limit]
    if not shard_paths:
        raise RefreshEvalError(f"no teacher shards found below {config.teacher_root}")
    samples = {
        sample.sample_id: sample for sample in load_teacher_samples(config.data_path)
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
        prompt_length = int(shard.prompt_input_ids.numel())
        if shard.top_order.shape[2] < prompt_length:
            raise RefreshEvalError(
                f"teacher shard lacks full prompt order: {shard_path}"
            )
        for method in config.methods:
            policy, attention_budget = _method_settings(method, config)
            output_path = summary_result_path(
                config.output_root,
                method.value,
                attention_budget,
                shard_path,
            )
            if output_path.exists():
                completed += 1
                continue
            cache = RefreshPromptKVCache(
                prompt_length=prompt_length,
                targets=build_refresh_targets(
                    shard.top_order,
                    shard.candidate_scores,
                ),
                config=RefreshCacheConfig(
                    cache_budget=config.cache_budget,
                    attention_budget=attention_budget,
                    refresh_interval=config.refresh_interval,
                    attention_policy=policy,
                ),
            )
            runner = LladaRefreshRunner(model, cache)
            _synchronize(config.device)
            started = time.monotonic()
            generated = generate_with_refresh_cache(
                runner,
                prompt_ids,
                _generation_config(shard),
            )
            _synchronize(config.device)
            result = build_summary_result(
                tokenizer,
                generated,
                stop_token_ids=stop_token_ids,
                sample_id=shard.sample_id,
                dataset=shard.dataset,
                method=method.value,
                budget=attention_budget,
                prompt_length=prompt_length,
                gold=sample.answer,
                matches_teacher_trajectory=torch.equal(
                    generated.cpu().squeeze(0),
                    shard.generated_input_ids,
                ),
                elapsed_seconds=time.monotonic() - started,
            )
            save_summary_result_atomic(result, output_path)
            completed += 1
            print(
                f"[refresh {shard_index}/{len(shard_paths)}] "
                f"method={method.value} rouge_l={result.rouge_l:.3f} "
                f"seconds={result.elapsed_seconds:.2f}",
                flush=True,
            )
    aggregates = aggregate_summary_results(load_summary_results(config.output_root))
    save_summary_aggregates_atomic(aggregates, config.output_root / "summary.json")
    return completed


def _method_settings(
    method: RefreshEvalMethod,
    config: RefreshEvalConfig,
) -> tuple[AttentionPolicy, int]:
    match method:
        case RefreshEvalMethod.CACHE_ALL:
            return AttentionPolicy.ALL_CACHE, config.cache_budget
        case RefreshEvalMethod.CACHE_TOPK:
            return AttentionPolicy.DYNAMIC_TOPK, config.attention_budget
        case unreachable:
            assert_never(unreachable)


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def parse_args(argv: Sequence[str] | None = None) -> RefreshEvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate step-conditioned attention within a refreshed prompt cache."
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
    parser.add_argument("--cache-budget", type=int, default=1024)
    parser.add_argument("--attention-budget", type=int, default=128)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument(
        "--methods",
        choices=tuple(method.value for method in RefreshEvalMethod),
        nargs="+",
        default=tuple(method.value for method in RefreshEvalMethod),
    )
    args = parser.parse_args(argv)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    return RefreshEvalConfig(
        model_path=args.model,
        data_path=args.data,
        teacher_root=args.teacher_root,
        output_root=args.output_root,
        device=args.device,
        dtype=dtype,
        limit=args.limit,
        cache_budget=args.cache_budget,
        attention_budget=args.attention_budget,
        refresh_interval=args.refresh_interval,
        methods=tuple(RefreshEvalMethod(method) for method in args.methods),
    )


def main(argv: Sequence[str] | None = None) -> int:
    run_refresh_eval(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
