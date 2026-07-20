#!/usr/bin/env python3
"""Create a symlinked teacher root for balanced multi-dataset student training."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


DEFAULT_WIKI_TEACHER_ROOT = Path(
    "experiment/345/2026-07-15/results/wiki_train_full_dynamic_teacher_n5000"
)
DEFAULT_BALANCED_TEACHER_ROOT = Path(
    "experiment/345/2026-07-15/results/balanced_train_full_dynamic_teacher_n19257"
)
DEFAULT_WIKI_DATA = Path(
    "/mnt/srv/home/dlpcg.325/dllm/data/train/2wikimultihopqa/"
    "2wikimultihopqa_train_longbench_format.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiment/345/2026-07-15/results/full_dynamic_teacher_balanced_plus_wiki_2k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-teacher-root", type=Path, default=DEFAULT_WIKI_TEACHER_ROOT)
    parser.add_argument("--balanced-teacher-root", type=Path, default=DEFAULT_BALANCED_TEACHER_ROOT)
    parser.add_argument("--wiki-data", type=Path, default=DEFAULT_WIKI_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wiki-dataset", default="2wikimultihopqa_train")
    parser.add_argument("--per-dataset-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_wiki_categories(path: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("_id")
            source_type = row.get("source_type")
            if isinstance(sample_id, str):
                categories[sample_id] = str(source_type or "UNKNOWN")
    return categories


def teacher_sample_id(path: Path) -> str:
    record = torch.load(path, map_location="cpu", weights_only=False)
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str):
        raise RuntimeError(f"{path} has no string sample_id")
    return sample_id


def allocate_counts(groups: dict[str, list[Path]], target: int) -> dict[str, int]:
    counts = {category: 0 for category in groups}
    remaining = min(target, sum(len(paths) for paths in groups.values()))
    while remaining > 0:
        active = [
            category
            for category, paths in sorted(groups.items())
            if counts[category] < len(paths)
        ]
        if not active:
            break
        base = max(1, remaining // len(active))
        for category in active:
            capacity = len(groups[category]) - counts[category]
            take = min(base, capacity, remaining)
            counts[category] += take
            remaining -= take
            if remaining == 0:
                break
    return counts


def select_wiki_files(
    wiki_dir: Path,
    categories_by_id: dict[str, str],
    target: int,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(wiki_dir.glob("*.pt")):
        sample_id = teacher_sample_id(path)
        groups[categories_by_id.get(sample_id, "UNKNOWN")].append(path)

    shuffled: dict[str, list[Path]] = {}
    for category, paths in groups.items():
        rng = random.Random(f"{seed}:wiki:{category}")
        values = list(paths)
        rng.shuffle(values)
        shuffled[category] = values

    counts = allocate_counts(shuffled, target)
    selected: list[Path] = []
    for category, count in counts.items():
        selected.extend(shuffled[category][:count])
    random.Random(f"{seed}:wiki:final").shuffle(selected)

    selected_categories = Counter(
        categories_by_id.get(teacher_sample_id(path), "UNKNOWN") for path in selected
    )
    manifest = {
        "source_dir": str(wiki_dir.resolve()),
        "available_files": sum(len(paths) for paths in groups.values()),
        "target_files": target,
        "selected_files": len(selected),
        "category_field": "source_type",
        "available_category_counts": {
            category: len(paths) for category, paths in sorted(groups.items())
        },
        "selected_category_counts": dict(sorted(selected_categories.items())),
    }
    return selected, manifest


def link_files(output_dir: Path, paths: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(paths):
        name = src.name
        dst = output_dir / name
        if dst.exists() or dst.is_symlink():
            name = f"{src.stem}.{index:05d}{src.suffix}"
            dst = output_dir / name
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "per_dataset_count": args.per_dataset_count,
        "seed": args.seed,
        "datasets": {},
    }

    wiki_dir = args.wiki_teacher_root / args.wiki_dataset
    wiki_categories = load_wiki_categories(args.wiki_data)
    wiki_files, wiki_manifest = select_wiki_files(
        wiki_dir=wiki_dir,
        categories_by_id=wiki_categories,
        target=args.per_dataset_count,
        seed=args.seed,
    )
    link_files(output_root / args.wiki_dataset, wiki_files)
    manifest["datasets"][args.wiki_dataset] = wiki_manifest

    for dataset_dir in sorted(path for path in args.balanced_teacher_root.iterdir() if path.is_dir()):
        files = sorted(dataset_dir.glob("*.pt"))
        if len(files) > args.per_dataset_count:
            rng = random.Random(f"{args.seed}:{dataset_dir.name}")
            files = list(files)
            rng.shuffle(files)
            files = files[: args.per_dataset_count]
        link_files(output_root / dataset_dir.name, files)
        manifest["datasets"][dataset_dir.name] = {
            "source_dir": str(dataset_dir.resolve()),
            "available_files": len(list(dataset_dir.glob("*.pt"))),
            "target_files": args.per_dataset_count,
            "selected_files": len(files),
        }

    total = 0
    for info in manifest["datasets"].values():
        total += int(info["selected_files"])
    manifest["total_selected_files"] = total
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote symlinked teacher root: {output_root}")
    print(f"Total selected files: {total}")
    for dataset, info in sorted(manifest["datasets"].items()):
        print(f"- {dataset}: {info['selected_files']}/{info['available_files']}")


if __name__ == "__main__":
    main()
