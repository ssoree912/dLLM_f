#!/usr/bin/env python3
"""Build a balanced train subset from local LongBench-format train JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CATEGORY_FIELDS: dict[str, tuple[str, ...]] = {
    "2wikimultihopqa": ("source_type",),
    "hotpotqa": ("source_type", "source_level"),
    "trec": ("answers[0]",),
    "triviaqa": ("question_source",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/train_balanced_2k_excl_2wikimultihopqa"),
    )
    parser.add_argument("--target-per-dataset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=345)
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=["2wikimultihopqa"],
        help="Dataset directory name to skip. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_field(row: dict[str, Any], field: str) -> Any:
    if field == "answers[0]":
        answers = row.get("answers")
        if isinstance(answers, list) and answers:
            return answers[0]
        return None
    return row.get(field)


def category_for_row(dataset: str, row: dict[str, Any]) -> str:
    fields = DEFAULT_CATEGORY_FIELDS.get(dataset)
    if not fields:
        return "__all__"
    values = [str(get_field(row, field) or "UNKNOWN") for field in fields]
    return " | ".join(values)


def iter_dataset_files(data_root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        matches = sorted(dataset_dir.glob("*_train_longbench_format.jsonl"))
        if not matches:
            continue
        if len(matches) > 1:
            raise RuntimeError(f"Multiple train JSONL files found under {dataset_dir}")
        files.append((dataset_dir.name, matches[0]))
    return files


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_no"] = line_no
            rows.append(row)
    return rows


def allocate_counts(groups: dict[str, list[int]], target: int) -> dict[str, int]:
    selected = {category: 0 for category in groups}
    remaining = min(target, sum(len(indices) for indices in groups.values()))

    while remaining > 0:
        active = [
            category
            for category, indices in sorted(groups.items())
            if selected[category] < len(indices)
        ]
        if not active:
            break

        base = max(1, remaining // len(active))
        progressed = False
        for category in active:
            capacity = len(groups[category]) - selected[category]
            take = min(base, capacity, remaining)
            if take <= 0:
                continue
            selected[category] += take
            remaining -= take
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    return selected


def sample_dataset(
    dataset: str,
    rows: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[category_for_row(dataset, row)].append(index)

    shuffled_groups: dict[str, list[int]] = {}
    for category, indices in sorted(groups.items()):
        rng = random.Random(f"{seed}:{dataset}:{category}")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        shuffled_groups[category] = shuffled

    counts = allocate_counts(shuffled_groups, target)
    selected_indices: list[int] = []
    for category, count in counts.items():
        selected_indices.extend(shuffled_groups[category][:count])

    rng = random.Random(f"{seed}:{dataset}:final_order")
    rng.shuffle(selected_indices)

    selected_rows = []
    for index in selected_indices:
        row = dict(rows[index])
        row.pop("_source_line_no", None)
        selected_rows.append(row)

    category_counts = Counter(category_for_row(dataset, row) for row in selected_rows)
    manifest = {
        "dataset": dataset,
        "source_rows": len(rows),
        "target_rows": target,
        "selected_rows": len(selected_rows),
        "category_fields": list(DEFAULT_CATEGORY_FIELDS.get(dataset, ())),
        "source_category_counts": {
            category: len(indices) for category, indices in sorted(groups.items())
        },
        "selected_category_counts": dict(sorted(category_counts.items())),
    }
    return selected_rows, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    excluded = set(args.exclude_dataset or [])

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_selected: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "target_per_dataset": args.target_per_dataset,
        "seed": args.seed,
        "excluded_datasets": sorted(excluded),
        "datasets": {},
    }

    for dataset, path in iter_dataset_files(data_root):
        if dataset in excluded:
            manifest["datasets"][dataset] = {
                "dataset": dataset,
                "source_file": str(path.resolve()),
                "excluded": True,
            }
            continue

        rows = load_rows(path)
        selected_rows, dataset_manifest = sample_dataset(
            dataset=dataset,
            rows=rows,
            target=args.target_per_dataset,
            seed=args.seed,
        )
        dataset_manifest["source_file"] = str(path.resolve())
        dataset_manifest["excluded"] = False

        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_file = dataset_dir / f"{dataset}_train_longbench_format.jsonl"
        write_jsonl(out_file, selected_rows)
        dataset_manifest["output_file"] = str(out_file)

        all_selected.extend(selected_rows)
        manifest["datasets"][dataset] = dataset_manifest

    all_file = output_root / "all_selected_train_longbench_format.jsonl"
    write_jsonl(all_file, all_selected)
    manifest["all_selected_file"] = str(all_file)
    manifest["total_selected_rows"] = len(all_selected)

    manifest_file = output_root / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(all_selected)} rows to {output_root}")
    for dataset, info in sorted(manifest["datasets"].items()):
        if info.get("excluded"):
            print(f"- {dataset}: excluded")
        else:
            category = ",".join(info["category_fields"]) or "none"
            print(
                f"- {dataset}: {info['selected_rows']}/{info['source_rows']} "
                f"(category={category})"
            )


if __name__ == "__main__":
    main()
