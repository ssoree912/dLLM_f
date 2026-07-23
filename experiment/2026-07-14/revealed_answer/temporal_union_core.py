from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_345_ROOT = REPO_ROOT / "experiment" / "345" / "2026-07-15"
for import_root in (SCRIPT_ROOT, EXPERIMENT_345_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.student_model import PromptUtilityStudent
from revealed_answer_345.full_dynamic_trajectory_teacher import (
    FullDynamicTrajectoryConfig,
    build_transfer_index,
    full_sequence_logits_and_prompt_attention,
    install_suffix_prompt_attention_collector,
    select_candidates,
)
from utils.generate_function import get_num_transfer_tokens

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
MASK_ID = 126336


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    keep = min(max(1, k), int(scores.numel()))
    return torch.topk(scores.float(), k=keep, largest=True).indices.sort().values


def jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left_set = set(int(item) for item in left.detach().cpu().tolist())
    right_set = set(int(item) for item in right.detach().cpu().tolist())
    return len(left_set & right_set) / len(left_set | right_set) if left_set or right_set else 1.0


def miss_ratio(needed: torch.Tensor, kept: torch.Tensor) -> float:
    needed_set = set(int(item) for item in needed.detach().cpu().tolist())
    kept_set = set(int(item) for item in kept.detach().cpu().tolist())
    return len(needed_set - kept_set) / len(needed_set) if needed_set else 0.0


def recall(needed: torch.Tensor, kept: torch.Tensor) -> float:
    return 1.0 - miss_ratio(needed, kept)


@torch.inference_mode()
def predict_student_scores(
    model: torch.nn.Module,
    student: PromptUtilityStudent,
    prompt_ids: torch.Tensor,
    question_window: int,
) -> torch.Tensor:
    out = model(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    prompt_length = int(prompt_ids.shape[1])
    question_count = min(max(1, question_window), prompt_length)
    prompt_indices = torch.arange(prompt_length, dtype=torch.long, device=prompt_ids.device)
    question_indices = torch.arange(prompt_length - question_count, prompt_length, dtype=torch.long, device=prompt_ids.device)
    scores = []
    for layer_id in student.layer_indices:
        layer_scores = student.forward_layer(layer_id, out.hidden_states[layer_id].float(), prompt_indices, question_indices)
        scores.append(torch.softmax(layer_scores.float(), dim=-1).squeeze(0).cpu())
    return torch.stack(scores)


@torch.inference_mode()
def diagnose_sample(model: torch.nn.Module, student: PromptUtilityStudent, sample_index: int, sample, example, config) -> list[dict[str, JsonValue]]:
    prompt_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=config.device)
    student_scores = predict_student_scores(model, student, prompt_ids, config.question_window)
    suffix_ids = torch.full((1, config.gen_length), MASK_ID, dtype=torch.long, device=config.device)
    teacher_config = FullDynamicTrajectoryConfig(
        gen_length=config.gen_length,
        block_length=config.block_length,
        steps=config.steps,
        temperature=config.temperature,
        max_weight=0.0,
        confidence_weight=config.confidence_weight,
    )
    collector = install_suffix_prompt_attention_collector(model, int(prompt_ids.shape[1]), config.gen_length)
    previous: dict[int, torch.Tensor] = {}
    first: dict[int, torch.Tensor] = {}
    temporal_sum: dict[int, torch.Tensor] = {}
    temporal_max: dict[int, torch.Tensor] = {}
    unions: dict[int, set[int]] = {}
    churns: dict[int, list[float]] = {}
    first_drifts: dict[int, list[float]] = {}
    step_sets: dict[int, list[torch.Tensor]] = {}
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    try:
        for block_id in range(num_blocks):
            end = (block_id + 1) * config.block_length
            block_mask = suffix_ids[:, block_id * config.block_length : end] == MASK_ID
            transfer_counts = get_num_transfer_tokens(block_mask, steps_per_block)
            for step_id in range(steps_per_block):
                mask_index = suffix_ids == MASK_ID
                logits, prompt_attention = full_sequence_logits_and_prompt_attention(model, prompt_ids, suffix_ids, collector)
                x0, confidence = select_candidates(logits, suffix_ids, mask_index, teacher_config)
                confidence[:, end:] = -torch.inf
                transfer_index = build_transfer_index(confidence, transfer_counts[:, step_id])
                selected = transfer_index[0].nonzero(as_tuple=False).flatten()
                if selected.numel() > 0:
                    update_temporal_state(
                        prompt_attention,
                        selected,
                        confidence[0].index_select(0, selected),
                        previous,
                        first,
                        temporal_sum,
                        temporal_max,
                        unions,
                        churns,
                        first_drifts,
                        step_sets,
                        config,
                    )
                suffix_ids[transfer_index] = x0[transfer_index]
    finally:
        collector.restore()
    return summarize_sample(
        sample_index,
        sample,
        example,
        student_scores,
        temporal_sum,
        temporal_max,
        unions,
        churns,
        first_drifts,
        step_sets,
        config,
    )


def update_temporal_state(
    prompt_attention: dict[int, torch.Tensor],
    selected: torch.Tensor,
    confidence: torch.Tensor,
    previous: dict[int, torch.Tensor],
    first: dict[int, torch.Tensor],
    temporal_sum: dict[int, torch.Tensor],
    temporal_max: dict[int, torch.Tensor],
    unions: dict[int, set[int]],
    churns: dict[int, list[float]],
    first_drifts: dict[int, list[float]],
    step_sets: dict[int, list[torch.Tensor]],
    config,
) -> None:
    weights = confidence.clamp_min(0.0).float() if config.confidence_weight else torch.ones_like(confidence)
    for layer_id, layer_attention in prompt_attention.items():
        scores = (layer_attention.index_select(0, selected).float() * weights.unsqueeze(-1)).sum(dim=0)
        current = topk_indices(scores, config.top_k)
        temporal_sum[layer_id] = temporal_sum.get(layer_id, torch.zeros_like(scores)) + scores
        temporal_max[layer_id] = torch.maximum(temporal_max.get(layer_id, torch.zeros_like(scores)), scores)
        unions.setdefault(layer_id, set()).update(int(item) for item in current.detach().cpu().tolist())
        step_sets.setdefault(layer_id, []).append(current.cpu())
        if layer_id not in first:
            first[layer_id] = current
        if layer_id in previous:
            churns.setdefault(layer_id, []).append(1.0 - jaccard(current, previous[layer_id]))
        first_drifts.setdefault(layer_id, []).append(1.0 - jaccard(current, first[layer_id]))
        previous[layer_id] = current


def summarize_sample(
    sample_index: int,
    sample,
    example,
    student_scores: torch.Tensor,
    temporal_sum: dict[int, torch.Tensor],
    temporal_max: dict[int, torch.Tensor],
    unions: dict[int, set[int]],
    churns: dict[int, list[float]],
    first_drifts: dict[int, list[float]],
    step_sets: dict[int, list[torch.Tensor]],
    config,
) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for layer_id in sorted(temporal_sum):
        temporal = temporal_max[layer_id] if config.target_aggregation == "max" else temporal_sum[layer_id]
        oracle_keep = topk_indices(temporal, config.budget).cpu()
        student_keep = topk_indices(student_scores[layer_id], config.budget).cpu()
        union_tensor = torch.tensor(sorted(unions.get(layer_id, set())), dtype=torch.long)
        row: dict[str, JsonValue] = {
            "sample_index": sample_index,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "layer": layer_id,
            "prompt_length": example.prompt_length,
            "top_k": config.top_k,
            "budget": config.budget,
            "union_size": int(union_tensor.numel()),
            "union_over_topk": float(union_tensor.numel() / max(1, config.top_k)),
            "union_over_budget": float(union_tensor.numel() / max(1, config.budget)),
            "mean_step_churn": mean(churns.get(layer_id, [])),
            "mean_first_drift": mean(first_drifts.get(layer_id, [])),
            "student_oracle_jaccard": jaccard(student_keep, oracle_keep),
            "student_union_jaccard": jaccard(student_keep, union_tensor),
            "evicted_then_needed_union": miss_ratio(union_tensor, student_keep),
            "evicted_then_needed_oracle_budget": miss_ratio(oracle_keep, student_keep),
        }
        add_pool_metrics(row, temporal, student_scores[layer_id], union_tensor, step_sets.get(layer_id, []), config)
        rows.append(row)
    return rows


def add_pool_metrics(
    row: dict[str, JsonValue],
    temporal: torch.Tensor,
    student_scores: torch.Tensor,
    union_tensor: torch.Tensor,
    layer_step_sets: list[torch.Tensor],
    config,
) -> None:
    prompt_length = max(1, int(row["prompt_length"]))
    for pool_budget in getattr(config, "pool_budgets", ()):
        pool_size = min(pool_budget, prompt_length)
        oracle_pool = topk_indices(temporal, pool_budget).cpu()
        student_pool = topk_indices(student_scores, pool_budget).cpu()
        first_pool = torch.arange(pool_size, dtype=torch.long)
        last_pool = torch.arange(prompt_length - pool_size, prompt_length, dtype=torch.long)
        seed = 1729 + int(row["sample_index"]) * 1009 + int(row["layer"]) * 17 + int(pool_budget)
        generator = torch.Generator().manual_seed(seed)
        random_pool = torch.randperm(prompt_length, generator=generator)[:pool_size].sort().values
        row[f"oracle_pool{pool_budget}_step_recall"] = mean([recall(step_set, oracle_pool) for step_set in layer_step_sets])
        row[f"student_pool{pool_budget}_step_recall"] = mean([recall(step_set, student_pool) for step_set in layer_step_sets])
        row[f"first_pool{pool_budget}_step_recall"] = mean([recall(step_set, first_pool) for step_set in layer_step_sets])
        row[f"last_pool{pool_budget}_step_recall"] = mean([recall(step_set, last_pool) for step_set in layer_step_sets])
        row[f"random_pool{pool_budget}_step_recall"] = mean([recall(step_set, random_pool) for step_set in layer_step_sets])
        row[f"oracle_pool{pool_budget}_union_recall"] = recall(union_tensor, oracle_pool)
        row[f"student_pool{pool_budget}_union_recall"] = recall(union_tensor, student_pool)
        row[f"first_pool{pool_budget}_union_recall"] = recall(union_tensor, first_pool)
        row[f"last_pool{pool_budget}_union_recall"] = recall(union_tensor, last_pool)
        row[f"random_pool{pool_budget}_union_recall"] = recall(union_tensor, random_pool)
        row[f"student_pool{pool_budget}_oracle_jaccard"] = jaccard(student_pool, oracle_pool)
        row[f"pool{pool_budget}_resident_fraction"] = pool_size / prompt_length


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def is_number(value: JsonValue) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize(rows: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    metadata_fields = {"sample_index", "sample_id", "dataset", "layer", "prompt_length", "top_k", "budget"}
    numeric_fields = sorted(
        {field for row in rows for field, value in row.items() if field not in metadata_fields and is_number(value)}
    )
    summary: dict[str, JsonValue] = {"sample_count": len({str(row["sample_id"]) for row in rows}), "row_count": len(rows)}
    for field in numeric_fields:
        summary[f"mean_{field}"] = mean([float(row[field]) for row in rows])
    return summary
