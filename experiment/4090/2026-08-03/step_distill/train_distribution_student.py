from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from .distribution_training import OnlineTrainConfig, run_training


def parse_args(argv: Sequence[str] | None = None) -> OnlineTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the B=960 online distribution-distilled selector."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-rollout-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4090)
    args = parser.parse_args(argv)
    return OnlineTrainConfig(
        model_path=args.model,
        train_data=args.train_data,
        validation_data=args.validation_data,
        output_dir=args.output_dir,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        max_length=args.max_length,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        epochs=args.epochs,
        max_rollout_steps=args.max_rollout_steps,
        learning_rate=args.learning_rate,
        projection_dim=args.projection_dim,
        mlp_dim=args.mlp_dim,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    checkpoint = run_training(parse_args(argv))
    print(f"[done] checkpoint={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
