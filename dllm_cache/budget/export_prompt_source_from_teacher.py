"""Strip teacher shards down to reusable prompt-only source records.

This preserves the exact balanced 300-per-dataset prompt population while removing
all legacy teacher labels.  The resulting shards can be consumed by
``extract_offline_hybrid_from_shards`` because they retain the prompt and metadata
fields that extractor needs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REQUIRED_FIELDS = (
    "sample_id",
    "dataset",
    "prompt_input_ids",
    "question_token_indices",
)

OPTIONAL_FIELDS = (
    "task",
    "question",
    "answers",
    "prompt_format",
    "prompt_length",
    "truncation_offset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_root.is_dir():
        raise RuntimeError(f"source root is not a directory: {args.source_root}")
    files = sorted(args.source_root.glob("*/*.pt"))
    if not files:
        raise RuntimeError(f"no teacher shards found under {args.source_root}")

    saved = 0
    for index, source_path in enumerate(files, start=1):
        output_path = args.output_root / source_path.relative_to(args.source_root)
        if output_path.exists():
            saved += 1
            continue
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        if missing:
            raise RuntimeError(f"{source_path} is missing required fields: {missing}")
        record = {
            "prompt_source_kind": "prompt_only_from_teacher_shard",
            "source_teacher_kind": source.get("teacher_kind", ""),
            **{field: source[field] for field in REQUIRED_FIELDS},
            **{field: source[field] for field in OPTIONAL_FIELDS if field in source},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, output_path)
        saved += 1
        if index % 100 == 0 or index == len(files):
            print(f"[prompt-source {index}/{len(files)}] saved={saved}", flush=True)

    print(f"[done] prompt-only shards={saved} output={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
