from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from revealed_answer.common import DEFAULT_MODEL_PATH, DEFAULT_TEACHER_ROOT


@dataclass(frozen=True, slots=True)
class TrainConfig:
    teacher_root: Path
    output_dir: Path
    model_path: Path
    datasets: list[str]
    n_samples: int
    val_ratio: float
    epochs: int
    lr: float
    weight_decay: float
    rank_weight: float
    rank_margin: float
    rank_top_ratio: float
    rank_bottom_ratio: float
    topk_weight: float
    topk_k: int
    topk_positive_weight: float
    max_grad_norm: float
    device: str
    dtype: torch.dtype
    seed: int
    log_every: int
    proj_dim: int
    mlp_dim: int


def parse_train_config() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--datasets", nargs="+", default=["2wikimqa"])
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-top-ratio", type=float, default=0.2)
    parser.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    parser.add_argument("--topk-weight", type=float, default=0.0)
    parser.add_argument("--topk-k", type=int, default=128)
    parser.add_argument("--topk-positive-weight", type=float, default=8.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--mlp-dim", type=int, default=512)
    args = parser.parse_args()
    return TrainConfig(
        teacher_root=args.teacher_root,
        output_dir=args.output_dir,
        model_path=args.model,
        datasets=args.datasets,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
        rank_top_ratio=args.rank_top_ratio,
        rank_bottom_ratio=args.rank_bottom_ratio,
        topk_weight=args.topk_weight,
        topk_k=args.topk_k,
        topk_positive_weight=args.topk_positive_weight,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        seed=args.seed,
        log_every=args.log_every,
        proj_dim=args.proj_dim,
        mlp_dim=args.mlp_dim,
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


def serializable_config(config: TrainConfig) -> dict[str, str | int | float | list[str]]:
    return {
        "teacher_root": str(config.teacher_root),
        "output_dir": str(config.output_dir),
        "model_path": str(config.model_path),
        "datasets": config.datasets,
        "n_samples": config.n_samples,
        "val_ratio": config.val_ratio,
        "epochs": config.epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "rank_weight": config.rank_weight,
        "rank_margin": config.rank_margin,
        "rank_top_ratio": config.rank_top_ratio,
        "rank_bottom_ratio": config.rank_bottom_ratio,
        "topk_weight": config.topk_weight,
        "topk_k": config.topk_k,
        "topk_positive_weight": config.topk_positive_weight,
        "max_grad_norm": config.max_grad_norm,
        "device": config.device,
        "dtype": str(config.dtype),
        "seed": config.seed,
        "log_every": config.log_every,
        "proj_dim": config.proj_dim,
        "mlp_dim": config.mlp_dim,
    }
