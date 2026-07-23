from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Final

import torch

SCRIPT_ROOT: Final = Path(__file__).resolve().parents[1]
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
EXPERIMENT_345_ROOT: Final = REPO_ROOT / "experiment" / "345" / "2026-07-15"
for import_root in (SCRIPT_ROOT, EXPERIMENT_345_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.extract_online_self_teacher_balanced import (
    parse_balanced_sample,
    tokenize_prompt_only,
)
from revealed_answer.extract_teacher import load_model_and_tokenizer, parse_dtype
from revealed_answer_345.full_dynamic_trajectory_teacher import (
    FullDynamicTrajectoryConfig,
    build_transfer_index,
    full_sequence_logits_and_prompt_attention,
    install_suffix_prompt_attention_collector,
    select_candidates,
)
from utils.generate_function import get_num_transfer_tokens

MASK_ID: Final = 126336
DEFAULT_MODEL_PATH: Final = Path("/home/M2026107/dllm/model/LLaDA-8B-Instruct")
DEFAULT_TRAIN_DATA: Final = Path("/home/M2026107/dllm/data/train_balanced_2k/all_selected_train_longbench_format.jsonl")
DEFAULT_OUTPUT_DIR: Final = Path("experiment/2026-07-14/results/full_dynamic_importance_drift_balanced_g32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", dest="data_path", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--confidence-weight", action="store_true")
    args = parser.parse_args()
    args.dtype = parse_dtype(args.dtype)
    return args


def load_stratified_samples(config: argparse.Namespace):
    samples = []
    counts: dict[str, int] = {}
    with config.data_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            sample = parse_balanced_sample(json.loads(line), row_index)
            if counts.get(sample.dataset, 0) >= config.samples_per_dataset:
                continue
            samples.append(sample)
            counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
            if len(samples) >= config.max_samples:
                break
    return samples


def topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    keep = min(max(1, k), int(scores.numel()))
    return torch.topk(scores.float(), k=keep, largest=True).indices.sort().values


def jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left_set = set(int(item) for item in left.detach().cpu().tolist())
    right_set = set(int(item) for item in right.detach().cpu().tolist())
    return len(left_set & right_set) / len(left_set | right_set)


@torch.inference_mode()
def diagnose_sample(model: torch.nn.Module, sample_index: int, sample, example, config: argparse.Namespace) -> list[dict[str, object]]:
    teacher_config = FullDynamicTrajectoryConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        max_weight=0.0,
        confidence_weight=config.confidence_weight,
    )
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    suffix_ids = torch.full((1, config.gen_length), MASK_ID, dtype=torch.long, device=config.device)
    collector = install_suffix_prompt_attention_collector(
        model,
        prompt_length=int(prompt_ids.shape[1]),
        suffix_length=config.gen_length,
    )
    rows: list[dict[str, object]] = []
    previous: dict[int, torch.Tensor] = {}
    first: dict[int, torch.Tensor] = {}
    cumulative: dict[int, torch.Tensor] = {}
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    try:
        for block_id in range(num_blocks):
            start = block_id * config.block_length
            end = (block_id + 1) * config.block_length
            block_mask = suffix_ids[:, start:end] == MASK_ID
            transfer_counts = get_num_transfer_tokens(block_mask, steps_per_block)
            for local_step in range(steps_per_block):
                mask_index = suffix_ids == MASK_ID
                logits, prompt_attention = full_sequence_logits_and_prompt_attention(model, prompt_ids, suffix_ids, collector)
                x0, confidence = select_candidates(logits, suffix_ids, mask_index, teacher_config)
                confidence[:, end:] = -torch.inf
                transfer_index = build_transfer_index(confidence, transfer_counts[:, local_step])
                selected = transfer_index[0].nonzero(as_tuple=False).flatten()
                if selected.numel() > 0:
                    append_rows(
                        rows,
                        prompt_attention,
                        selected,
                        confidence[0].index_select(0, selected),
                        previous,
                        first,
                        cumulative,
                        {
                            "sample_index": sample_index,
                            "sample_id": sample.sample_id,
                            "dataset": sample.dataset,
                            "prompt_length": example.prompt_length,
                            "block": block_id,
                            "step": block_id * steps_per_block + local_step,
                            "filled_suffix": int((suffix_ids != MASK_ID).sum().item()),
                        },
                        config,
                    )
                suffix_ids[transfer_index] = x0[transfer_index]
    finally:
        collector.restore()
    return rows


def append_rows(
    rows: list[dict[str, object]],
    prompt_attention: dict[int, torch.Tensor],
    selected: torch.Tensor,
    confidence: torch.Tensor,
    previous: dict[int, torch.Tensor],
    first: dict[int, torch.Tensor],
    cumulative: dict[int, torch.Tensor],
    base_row: dict[str, object],
    config: argparse.Namespace,
) -> None:
    weights = confidence.clamp_min(0.0).float()
    if not config.confidence_weight:
        weights = torch.ones_like(weights)
    for layer_id, layer_attention in prompt_attention.items():
        scores = (layer_attention.index_select(0, selected).float() * weights.unsqueeze(-1)).sum(dim=0)
        current_topk = topk(scores, config.top_k)
        if layer_id not in cumulative:
            cumulative[layer_id] = torch.zeros_like(scores)
            first[layer_id] = current_topk
        cumulative[layer_id] += scores
        prev = previous.get(layer_id)
        row = dict(base_row)
        row.update(
            {
                "layer": layer_id,
                "selected_count": int(selected.numel()),
                "prev_jaccard": None if prev is None else jaccard(current_topk, prev),
                "first_jaccard": jaccard(current_topk, first[layer_id]),
                "cumulative_jaccard": jaccard(current_topk, topk(cumulative[layer_id], config.top_k)),
            }
        )
        rows.append(row)
        previous[layer_id] = current_topk


def summarize(rows: list[dict[str, object]], config: argparse.Namespace) -> dict[str, object]:
    return {
        "sample_count": len({row["sample_id"] for row in rows}),
        "row_count": len(rows),
        "top_k": config.top_k,
        "mean_prev_jaccard": mean(row["prev_jaccard"] for row in rows),
        "mean_first_jaccard": mean(row["first_jaccard"] for row in rows),
        "mean_cumulative_jaccard": mean(row["cumulative_jaccard"] for row in rows),
        "by_dataset": group_summary(rows, "dataset"),
        "by_layer": group_summary(rows, "layer"),
    }


def group_summary(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        key: {
            "mean_prev_jaccard": mean(row["prev_jaccard"] for row in values),
            "mean_first_jaccard": mean(row["first_jaccard"] for row in values),
            "mean_cumulative_jaccard": mean(row["cumulative_jaccard"] for row in values),
        }
        for key, values in groups.items()
    }


def mean(values) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else 0.0


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(config)
    samples = load_stratified_samples(config)
    rows: list[dict[str, object]] = []
    started = time.time()
    for sample_index, sample in enumerate(samples):
        example = tokenize_prompt_only(tokenizer, sample, config)
        sample_rows = diagnose_sample(model, sample_index, sample, example, config)
        rows.extend(sample_rows)
        print(
            f"[sample {sample_index + 1}/{len(samples)}] dataset={sample.dataset} "
            f"prompt={example.prompt_length} rows={len(sample_rows)}",
            flush=True,
        )
    rows_path = config.output_dir / "step_topk_drift.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows, config) | {"elapsed_sec": time.time() - started}
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] rows={rows_path} summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
