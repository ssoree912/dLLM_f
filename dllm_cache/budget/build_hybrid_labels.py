# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///

"""Build fixed-budget hybrid label shards from offline hybrid teacher shards.

Reads every ``<teacher-root>/<dataset>/<sample>.pt`` produced by
``extract_offline_hybrid_teacher`` and writes one label shard per sample with
the four comparison masks at the requested budget.  CPU-only and cheap, so
budget/ref-ratio sweeps re-run this step instead of re-extracting.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from dllm_cache.budget.hybrid_labels import (
    MASK_VARIANTS,
    build_hybrid_label_masks,
    sample_generator,
)


@dataclass(frozen=True, slots=True)
class BuildHybridLabelsConfig:
    teacher_root: Path
    output_root: Path
    datasets: tuple[str, ...]
    budget: int
    ref_ratio: float
    seed: int
    overwrite: bool


def parse_args(argv: Sequence[str] | None = None) -> BuildHybridLabelsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--ref-ratio",
        type=float,
        default=0.75,
        help="fraction of the budget reserved for reference top-k; 1.0 degenerates "
        "reference_random/reference_delta into reference_only",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= args.ref_ratio <= 1.0:
        raise SystemExit("--ref-ratio must be in [0, 1]")
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    output_root = args.output_root
    if output_root is None:
        ratio_tag = f"{args.ref_ratio:.2f}".replace(".", "p")
        output_root = args.teacher_root.parent / (
            f"{args.teacher_root.name}_labels_b{args.budget}_r{ratio_tag}"
        )
    return BuildHybridLabelsConfig(
        teacher_root=args.teacher_root,
        output_root=output_root,
        datasets=tuple(args.datasets) if args.datasets else (),
        budget=args.budget,
        ref_ratio=args.ref_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


def list_teacher_files(config: BuildHybridLabelsConfig) -> list[Path]:
    if not config.teacher_root.is_dir():
        raise RuntimeError(f"teacher root is not a directory: {config.teacher_root}")
    files: list[Path] = []
    for dataset_dir in sorted(path for path in config.teacher_root.iterdir() if path.is_dir()):
        if config.datasets and dataset_dir.name not in config.datasets:
            continue
        files.extend(sorted(dataset_dir.glob("*.pt")))
    if not files:
        raise RuntimeError(f"no teacher shards found under {config.teacher_root}")
    return files


def build_one(rec: dict, config: BuildHybridLabelsConfig) -> dict:
    ref_scores = rec["teacher_raw"].float()
    delta_scores = rec["delta_raw"].float()
    ref_k = int(round(config.budget * config.ref_ratio))
    result = build_hybrid_label_masks(
        ref_scores,
        delta_scores,
        budget=config.budget,
        ref_k=ref_k,
        generator=sample_generator(config.seed, str(rec["sample_id"])),
    )
    label = {
        "label_kind": "hybrid_topk_priority",
        "teacher_kind": rec.get("teacher_kind", ""),
        "sample_id": rec["sample_id"],
        "dataset": rec["dataset"],
        "task": rec.get("task", ""),
        "prompt_input_ids": rec["prompt_input_ids"],
        "prompt_token_indices": rec["prompt_token_indices"],
        "question_token_indices": rec["question_token_indices"],
        "prompt_length": int(rec["prompt_length"]),
        "budget": result.budget,
        "requested_budget": config.budget,
        "ref_k": result.ref_k,
        "ref_ratio": config.ref_ratio,
        "seed": config.seed,
        "hybrid_mask": result.masks["reference_delta"],
        "ref_delta_jaccard": result.ref_delta_jaccard,
        "ref_delta_jaccard_mean": float(result.ref_delta_jaccard.mean().item()),
    }
    for name in MASK_VARIANTS:
        label[f"mask_{name}"] = result.masks[name]
    return label


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    files = list_teacher_files(config)
    started = time.time()
    saved = 0
    jaccard_total = 0.0
    for index, path in enumerate(files, start=1):
        out_path = config.output_root / path.relative_to(config.teacher_root)
        if out_path.exists() and not config.overwrite:
            saved += 1
            continue
        rec = torch.load(path, map_location="cpu", weights_only=False)
        label = build_one(rec, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(label, out_path)
        saved += 1
        jaccard_total += label["ref_delta_jaccard_mean"]
        print(
            f"[hybrid-labels {index}/{len(files)}] saved={out_path} "
            f"budget={label['budget']} ref_k={label['ref_k']} "
            f"jaccard={label['ref_delta_jaccard_mean']:.3f}",
            flush=True,
        )
    mean_jaccard = jaccard_total / max(1, saved)
    print(
        f"[done] saved={saved} output={config.output_root} "
        f"mean_ref_delta_jaccard={mean_jaccard:.3f} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
