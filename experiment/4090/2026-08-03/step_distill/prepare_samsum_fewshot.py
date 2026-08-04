from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from transformers import AutoTokenizer

from .samsum_fewshot_data import (
    FewshotBuildSpec,
    FewshotDataError,
    FewshotPackRequest,
    FewshotSplit,
    SamsumFewshotRecord,
    packed_samsum_record,
    pack_samsum_target,
    select_samsum_targets,
)
from .task_data import OffsetTokenizer, load_teacher_samples


@dataclass(frozen=True, slots=True)
class PrepareFewshotConfig:
    source_path: Path
    eval_path: Path | None
    output_root: Path
    tokenizer_id: str
    spec: FewshotBuildSpec


@dataclass(frozen=True, slots=True)
class PreparedFewshotPaths:
    train_path: Path
    validation_path: Path
    manifest_path: Path


class FewshotManifest(TypedDict):
    source_path: str
    source_sha256: str
    eval_path: str | None
    tokenizer_id: str
    seed: int
    max_length: int
    reserve_length: int
    prompt_cap: int
    train_count: int
    validation_count: int
    demo_pool_count: int
    target_demo_overlap_count: int
    eval_target_overlap_count: int
    train_prompt_length_min: int
    train_prompt_length_max: int
    validation_prompt_length_min: int
    validation_prompt_length_max: int


def run_prepare(
    config: PrepareFewshotConfig,
    tokenizer: OffsetTokenizer,
) -> PreparedFewshotPaths:
    """Build and atomically publish train-only LongBench-style SAMSum splits."""
    samples = load_teacher_samples(config.source_path)
    selected = select_samsum_targets(samples, config.spec)
    target_ids = {
        sample.sample_id for sample in (*selected.train, *selected.validation)
    }
    eval_ids = (
        {sample.sample_id for sample in load_teacher_samples(config.eval_path)}
        if config.eval_path is not None
        else set()
    )
    eval_overlap = target_ids & eval_ids
    if eval_overlap:
        raise FewshotDataError(
            f"source targets overlap eval IDs: {sorted(eval_overlap)[:3]}"
        )

    train_records = [
        packed_samsum_record(
            pack_samsum_target(
                FewshotPackRequest(tokenizer, target, selected.demo_pool, config.spec)
            ),
            FewshotSplit.TRAIN,
            config.spec,
        )
        for target in selected.train
    ]
    validation_records = [
        packed_samsum_record(
            pack_samsum_target(
                FewshotPackRequest(tokenizer, target, selected.demo_pool, config.spec)
            ),
            FewshotSplit.VALIDATION,
            config.spec,
        )
        for target in selected.validation
    ]
    paths = PreparedFewshotPaths(
        train_path=config.output_root / "samsum_train_fewshot_2048.jsonl.xz",
        validation_path=(
            config.output_root / "samsum_validation_fewshot_2048.jsonl.xz"
        ),
        manifest_path=config.output_root / "manifest.json",
    )
    _write_jsonl_atomic(paths.train_path, train_records)
    _write_jsonl_atomic(paths.validation_path, validation_records)
    manifest = FewshotManifest(
        source_path=str(config.source_path.resolve()),
        source_sha256=_sha256(config.source_path),
        eval_path=(
            str(config.eval_path.resolve()) if config.eval_path is not None else None
        ),
        tokenizer_id=config.tokenizer_id,
        seed=config.spec.seed,
        max_length=config.spec.max_length,
        reserve_length=config.spec.reserve_length,
        prompt_cap=config.spec.prompt_cap,
        train_count=len(train_records),
        validation_count=len(validation_records),
        demo_pool_count=len(selected.demo_pool),
        target_demo_overlap_count=0,
        eval_target_overlap_count=len(eval_overlap),
        train_prompt_length_min=min(
            row["prompt_token_length"] for row in train_records
        ),
        train_prompt_length_max=max(
            row["prompt_token_length"] for row in train_records
        ),
        validation_prompt_length_min=min(
            row["prompt_token_length"] for row in validation_records
        ),
        validation_prompt_length_max=max(
            row["prompt_token_length"] for row in validation_records
        ),
    )
    _write_json_atomic(paths.manifest_path, manifest)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> PrepareFewshotConfig:
    parser = argparse.ArgumentParser(
        description="Build train-only LongBench-style SAMSum few-shot prompts."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--reserve-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4090)
    args = parser.parse_args(argv)
    return PrepareFewshotConfig(
        source_path=args.source,
        eval_path=args.eval_data,
        output_root=args.output_root,
        tokenizer_id=str(args.tokenizer.resolve()),
        spec=FewshotBuildSpec(
            max_length=args.max_length,
            reserve_length=args.reserve_length,
            train_count=args.train_count,
            validation_count=args.validation_count,
            seed=args.seed,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_id,
        trust_remote_code=True,
    )
    paths = run_prepare(config, tokenizer)
    print(f"train={paths.train_path}")
    print(f"validation={paths.validation_path}")
    print(f"manifest={paths.manifest_path}")
    return 0


def _write_jsonl_atomic(path: Path, records: Sequence[SamsumFewshotRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    with lzma.open(temporary_path, mode="wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, record: FewshotManifest) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
