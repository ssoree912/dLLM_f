"""Train one two-head Attention+Delta prune-cache scorer.

The training recipe is fixed so the public command only selects data and output
paths.  Both heads share the token/question projections and are optimized in
the same step from ``teacher_norm`` and ``delta_norm``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from dllm_cache.budget.train_config import TrainConfig, parse_dtype, serializable_config
from dllm_cache.budget.training_loop import build_runtime, run_training, split_teacher_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args(argv)


def fixed_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        teacher_root=args.teacher_root,
        output_dir=args.output_dir,
        resume_from=None,
        model_path=args.model,
        datasets=list(args.datasets),
        n_samples=args.n_samples,
        val_ratio=0.1,
        epochs=20,
        lr=2e-5,
        weight_decay=0.01,
        target_mode="attention_delta",
        loss_mode="mse",
        bce_positive_weight=1.0,
        rank_weight=0.1,
        rank_margin=0.05,
        rank_top_ratio=0.2,
        rank_bottom_ratio=0.4,
        rank_input="prob",
        topk_weight=0.0,
        topk_k=128,
        topk_positive_weight=8.0,
        max_grad_norm=1.0,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        seed=0,
        log_every=10,
        proj_dim=256,
        mlp_dim=512,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = fixed_train_config(parse_args(argv))
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_config.json").write_text(
        json.dumps(serializable_config(config), indent=2),
        encoding="utf-8",
    )
    runtime = build_runtime(config)
    split = split_teacher_files(config)
    print(f"[data] train={len(split.train_files)} val={len(split.val_files)}", flush=True)
    with (config.output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        run_training(runtime, split, log_file)
    runtime.student.save_pretrained(config.output_dir / "checkpoint-last")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
