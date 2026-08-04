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

from .generation_output import llada_stop_token_ids
from .oracle_generation import ReplayGenerationConfig, generate_offline_replay
from .oracle_pruning import (
    OfflineReplayController,
    ReplayMode,
    install_offline_replay_pruner,
)
from .oracle_results import (
    build_oracle_result,
    load_results,
    result_path,
    save_result_atomic,
    save_summary_atomic,
    summarize_results,
)
from .schema import TeacherShard, load_teacher_shard
from .task_data import TeacherSample, load_teacher_samples


class EvalMethod(str, Enum):
    FULL = "full"
    STATIC_PLAIN = "static_plain"
    DYNAMIC_PLAIN = "dynamic_plain"


@dataclass(frozen=True, slots=True)
class OracleEvalError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class OracleEvalConfig:
    model_path: Path
    data_path: Path
    teacher_root: Path
    output_root: Path
    device: str
    dtype: torch.dtype
    limit: int
    budgets: tuple[int, ...]
    methods: tuple[EvalMethod, ...]


def _controller(
    shard: TeacherShard,
    method: EvalMethod,
    budget: int,
) -> OfflineReplayController:
    match method:
        case EvalMethod.STATIC_PLAIN:
            mode, order = ReplayMode.STATIC, shard.top_order
        case EvalMethod.DYNAMIC_PLAIN:
            mode, order = ReplayMode.DYNAMIC, shard.top_order
        case EvalMethod.FULL:
            raise OracleEvalError("full inference does not use a replay controller")
        case unreachable:
            assert_never(unreachable)
    return OfflineReplayController(
        prompt_length=int(shard.prompt_input_ids.numel()),
        budget=budget,
        order=order,
        mode=mode,
    )


def _generation_config(shard: TeacherShard) -> ReplayGenerationConfig:
    return ReplayGenerationConfig(
        gen_length=shard.metadata.gen_length,
        block_length=shard.metadata.block_length,
        steps=shard.metadata.steps,
        temperature=shard.metadata.temperature,
        mask_id=shard.metadata.mask_id,
    )


def generate_method(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    shard: TeacherShard,
    method: EvalMethod,
    budget: int | None,
) -> torch.Tensor:
    if method is EvalMethod.FULL:
        return generate_offline_replay(model, prompt_ids, _generation_config(shard))
    if budget is None:
        raise OracleEvalError("pruned replay requires a concrete budget")
    controller = _controller(shard, method, budget)
    install_offline_replay_pruner(model, controller)
    try:
        return generate_offline_replay(
            model,
            prompt_ids,
            _generation_config(shard),
            step_controller=controller,
        )
    finally:
        controller.restore()


def run_oracle_eval(config: OracleEvalConfig) -> int:
    shard_paths = sorted(config.teacher_root.rglob("*.pt"))
    if config.limit > 0:
        shard_paths = shard_paths[: config.limit]
    if not shard_paths:
        raise OracleEvalError(
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
        sample: TeacherSample = samples[shard.sample_id]
        prompt_ids = shard.prompt_input_ids.unsqueeze(0).to(config.device)
        for method in config.methods:
            budgets: tuple[int | None, ...] = (
                (None,) if method is EvalMethod.FULL else config.budgets
            )
            for budget in budgets:
                output_path = result_path(
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
                    raise OracleEvalError(
                        f"full replay diverged from teacher for {shard.sample_id}"
                    )
                result = build_oracle_result(
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
                save_result_atomic(result, output_path)
                completed += 1
                print(
                    f"[oracle {shard_index}/{len(shard_paths)}] "
                    f"method={method.value} budget={budget} "
                    f"recall={result.recall:.3f} f1={result.f1:.3f}",
                    flush=True,
                )
    summaries = summarize_results(load_results(config.output_root))
    save_summary_atomic(summaries, config.output_root / "summary.json")
    return completed


def parse_args(argv: Sequence[str] | None = None) -> OracleEvalConfig:
    parser = argparse.ArgumentParser(
        description="Run stored per-step teacher orders as offline replay oracles."
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
    parser.add_argument("--budgets", type=int, nargs="+", default=(512, 256, 128))
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
    return OracleEvalConfig(
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
    run_oracle_eval(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
