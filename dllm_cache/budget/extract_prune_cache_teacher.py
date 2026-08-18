"""Extract the fixed no-chat Attention+Delta prune-cache teacher.

One generation trajectory produces both targets:

* dense, confidence-weighted commit-time Attention with max aggregation;
* cumulative stepwise prompt K/V movement.

There are deliberately no chat, top-k, confidence, or aggregation flags.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from dllm_cache.budget.extract_offline_hybrid_from_shards import main as extract_main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--n-samples", type=int, default=0, help="per dataset; 0 = all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args(argv)


def validate_no_chat_sources(source_root: Path, datasets: list[str], n_samples: int) -> None:
    checked = 0
    for dataset in datasets:
        paths = sorted((source_root / dataset).glob("*.pt"))
        if n_samples > 0:
            paths = paths[:n_samples]
        if not paths:
            raise RuntimeError(f"no source shards found for dataset: {dataset}")
        for path in paths:
            record = torch.load(path, map_location="cpu", weights_only=False)
            if bool(record.get("apply_chat_template", 0)):
                raise RuntimeError(
                    "prune-cache teacher sources must be no-chat; found a chat-wrapped "
                    f"source shard: {path}"
                )
            checked += 1
    if checked == 0:
        raise RuntimeError(f"no source shards found under {source_root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_no_chat_sources(args.source_root, args.datasets, args.n_samples)
    fixed_argv = [
        "--model",
        str(args.model),
        "--source-root",
        str(args.source_root),
        "--output-root",
        str(args.output_root),
        "--datasets",
        *args.datasets,
        "--n-samples",
        str(args.n_samples),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--gen-length",
        "128",
        "--block-length",
        "8",
        "--steps",
        "128",
        "--active-top-k",
        "0",
        "--confidence-weight",
        "--target-aggregation",
        "max",
    ]
    return extract_main(fixed_argv)


if __name__ == "__main__":
    raise SystemExit(main())
