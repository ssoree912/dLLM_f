from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from dllm_cache.budget.extract_teacher import load_model_and_tokenizer, parse_dtype
from dllm_cache.budget.pool_active_prompt_kv import (
    build_pool_active_prompt_kv_cache,
    generate_with_pool_active_prompt_kv,
)


@dataclass(frozen=True, slots=True)
class TrecOraclePoolConfig:
    model_path: Path
    teacher_root: Path
    data_path: Path
    output_root: Path
    budgets: list[int]
    n_samples: int
    device: str
    dtype: torch.dtype
    gen_length: int
    block_length: int
    steps: int
    selection_mode: str
    mask_id: int


def parse_args() -> TrecOraclePoolConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 512, 128])
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--selection-mode", choices=["global", "layer_union"], default="global")
    parser.add_argument("--mask-id", type=int, default=126336)
    args = parser.parse_args()
    return TrecOraclePoolConfig(
        model_path=args.model,
        teacher_root=args.teacher_root,
        data_path=args.data,
        output_root=args.output_root,
        budgets=args.budgets,
        n_samples=args.n_samples,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        selection_mode=args.selection_mode,
        mask_id=args.mask_id,
    )


def main() -> int:
    config = parse_args()
    records = collect_teacher_records(config)
    data_by_id = load_trec_data(config.data_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    samples_path = config.output_root / "samples.jsonl"
    done = load_done(samples_path)
    model, tokenizer = load_model_and_tokenizer(config)
    started = time.time()
    for index, record_path in enumerate(records, start=1):
        rec = torch.load(record_path, map_location="cpu", weights_only=False)
        sample_id = str(rec["sample_id"])
        data = data_by_id.get(sample_id)
        if data is None:
            raise RuntimeError(f"missing TREC data row for sample_id={sample_id}")
        for budget in config.budgets:
            key = (sample_id, budget)
            if key in done:
                continue
            row = run_one(model, tokenizer, rec, data, record_path, budget, config)
            append_jsonl(samples_path, row)
            done.add(key)
            print(
                f"[trec-oracle-token-prune {index}/{len(records)}] budget={budget} "
                f"score={row['score']:.4f} elapsed={row['elapsed_seconds']:.2f}s "
                f"reduced_prompt={row['reduced_prompt_length']}",
                flush=True,
            )
        write_summary(samples_path, config)
    write_summary(samples_path, config)
    print(f"[done] records={len(records)} elapsed={time.time() - started:.1f}s output={config.output_root}", flush=True)
    return 0


def collect_teacher_records(config: TrecOraclePoolConfig) -> list[Path]:
    files = sorted((config.teacher_root / "trec").glob("*.pt"))
    if config.n_samples > 0:
        files = files[: config.n_samples]
    if not files:
        raise RuntimeError(f"no teacher records found under {config.teacher_root / 'trec'}")
    return files


def load_trec_data(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            sample_id = str(row.get("_id", f"sample_{index}"))
            rows[sample_id] = row
    return rows


def load_done(samples_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not samples_path.exists():
        return done
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            done.add((str(row["sample_id"]), int(row["budget"])))
    return done


@torch.inference_mode()
def run_one(
    model: torch.nn.Module,
    tokenizer,
    rec: dict,
    data: dict,
    record_path: Path,
    budget: int,
    config: TrecOraclePoolConfig,
) -> dict:
    if "future_frequency" not in rec:
        raise RuntimeError(f"teacher record lacks future_frequency: {record_path}")
    input_ids = rec["prompt_input_ids"].unsqueeze(0).to(config.device)
    teacher_scores = rec["future_frequency"].float()
    prompt_cache = build_pool_active_prompt_kv_cache(
        model,
        input_ids,
        pool_budget=budget,
        active_budget=budget,
        teacher_scores=teacher_scores,
        selection_mode=config.selection_mode,
    )
    torch.cuda.synchronize() if input_ids.device.type == "cuda" else None
    t0 = time.perf_counter()
    output_ids = generate_with_pool_active_prompt_kv(
        input_ids=input_ids,
        model=model,
        prompt_cache=prompt_cache,
        steps=config.steps,
        gen_length=config.gen_length,
        block_length=config.block_length,
        refresh_interval=1,
        temperature=0.0,
        mask_id=config.mask_id,
    )
    torch.cuda.synchronize() if input_ids.device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    answers = parse_answers(data.get("answers"))
    answer = answers[0]
    all_classes = parse_all_classes(data.get("all_classes"))
    score = classification_score(prediction, answer, all_classes)
    union_size = rec.get("future_union_size_by_layer")
    return {
        "sample_id": str(rec["sample_id"]),
        "teacher_record": str(record_path),
        "budget": int(budget),
        "selection_mode": config.selection_mode,
        "prompt_length": int(rec["prompt_length"]),
        "reduced_prompt_length": int(prompt_cache.reduced_prompt_length),
        "inner_active_selection": False,
        "teacher_target": "future_frequency_from_per_step_top128",
        "gen_length": int(config.gen_length),
        "steps": int(config.steps),
        "answer": answer,
        "prediction": prediction,
        "score": float(score),
        "classification_score": float(score),
        "union_size_mean": float(union_size.float().mean().item()) if torch.is_tensor(union_size) else None,
        "elapsed_seconds": float(elapsed),
    }


def parse_answers(value: object) -> list[str]:
    match value:
        case [*items] if items and all(isinstance(item, str) for item in items):
            return list(items)
        case str() as text:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
                return parsed
            raise RuntimeError("answers string must encode a non-empty string list")
        case _:
            raise RuntimeError("answers must be a non-empty string list")


def parse_all_classes(value: object) -> list[str]:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    raise RuntimeError("all_classes must be a non-empty string list")


def classification_score(prediction: str, ground_truth: str, all_classes: list[str]) -> float:
    matches = [class_name for class_name in all_classes if class_name in prediction]
    filtered = [match for match in matches if not (match in ground_truth and match != ground_truth)]
    if ground_truth in filtered:
        return 1.0 / float(len(filtered))
    return 0.0


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(samples_path: Path, config: TrecOraclePoolConfig) -> None:
    rows = read_rows(samples_path)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget"])].append(row)
    summary = {
        "teacher_root": str(config.teacher_root),
        "data_path": str(config.data_path),
        "budgets": config.budgets,
        "selection_mode": config.selection_mode,
        "n_samples_requested": config.n_samples,
        "rows": len(rows),
        "by_budget": {
            str(budget): {
                "count": len(values),
                "score_mean": mean(float(row["score"]) for row in values),
                "elapsed_seconds_mean": mean(float(row["elapsed_seconds"]) for row in values),
                "reduced_prompt_length_mean": mean(float(row["reduced_prompt_length"]) for row in values),
            }
            for budget, values in sorted(grouped.items())
        },
    }
    (config.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
