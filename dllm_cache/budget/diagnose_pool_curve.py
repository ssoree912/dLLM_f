from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModel

from dllm_cache.budget.common import DEFAULT_MODEL_PATH
from dllm_cache.budget.student_model import PromptUtilityStudent, StudentConfig
from dllm_cache.budget.train_config import parse_dtype


@dataclass(frozen=True, slots=True)
class PoolCurveConfig:
    teacher_root: Path
    datasets: list[str]
    budgets: list[int]
    n_samples: int
    seed: int
    target_mode: str
    student_path: Path | None
    model_path: Path
    question_window: int
    student_score_activation: str
    device: str
    dtype: torch.dtype
    output: Path | None


def parse_args() -> PoolCurveConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 512, 256, 128])
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mode", choices=["auto", "frequency", "score", "union"], default="auto")
    parser.add_argument("--student-path", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--student-score-activation", choices=["softmax", "sigmoid", "raw"], default="sigmoid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    return PoolCurveConfig(
        teacher_root=args.teacher_root,
        datasets=args.datasets,
        budgets=args.budgets,
        n_samples=args.n_samples,
        seed=args.seed,
        target_mode=args.target_mode,
        student_path=args.student_path,
        model_path=args.model,
        question_window=args.question_window,
        student_score_activation=args.student_score_activation,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        output=args.output,
    )


def main() -> int:
    config = parse_args()
    files = collect_teacher_files(config)
    model = student = None
    if config.student_path is not None:
        model = load_frozen_model(config)
        student = load_student(config)
    rows = []
    for index, path in enumerate(files, start=1):
        rec = torch.load(path, map_location="cpu", weights_only=False)
        student_scores = None
        if model is not None and student is not None:
            student_scores = predict_student_scores(model, student, rec, config)
        rows.extend(analyze_record(rec, path, config, student_scores))
        if index % 25 == 0 or index == len(files):
            print(f"[pool-curve] processed {index}/{len(files)}", flush=True)
    summary = summarize(rows, config)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(text + "\n", encoding="utf-8")
    return 0


def collect_teacher_files(config: PoolCurveConfig) -> list[Path]:
    files: list[Path] = []
    for dataset in config.datasets:
        files.extend(sorted((config.teacher_root / dataset).glob("*.pt")))
    if not files:
        raise RuntimeError(f"no teacher files found under {config.teacher_root}")
    rng = random.Random(config.seed)
    rng.shuffle(files)
    if config.n_samples > 0:
        files = files[: config.n_samples]
    return files


def load_frozen_model(config: PoolCurveConfig) -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_student(config: PoolCurveConfig) -> PromptUtilityStudent:
    if config.student_path is None:
        raise RuntimeError("student_path is required")
    raw = json.loads((config.student_path / "config.json").read_text(encoding="utf-8"))
    student = PromptUtilityStudent(StudentConfig(**raw))
    state = torch.load(config.student_path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    student.load_state_dict(state)
    student.to(config.device)
    student.eval()
    return student


@torch.no_grad()
def predict_student_scores(
    model: torch.nn.Module,
    student: PromptUtilityStudent,
    rec: dict,
    config: PoolCurveConfig,
) -> torch.Tensor:
    input_ids = rec["prompt_input_ids"].unsqueeze(0).to(config.device)
    out = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    prompt_length = int(input_ids.shape[1])
    prompt_indices = rec["prompt_token_indices"].to(config.device)
    if "question_token_indices" in rec:
        question_indices = rec["question_token_indices"].to(config.device)
    else:
        question_count = min(max(1, config.question_window), prompt_length)
        question_indices = torch.arange(
            prompt_length - question_count,
            prompt_length,
            dtype=torch.long,
            device=config.device,
        )
    scores = []
    for layer_id in student.layer_indices:
        logits = student.forward_layer(
            layer_id,
            out.hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        ).float()
        match config.student_score_activation:
            case "softmax":
                layer_scores = torch.softmax(logits, dim=-1)
            case "sigmoid":
                layer_scores = torch.sigmoid(logits)
            case "raw":
                layer_scores = logits
            case _:
                raise RuntimeError(f"unsupported activation: {config.student_score_activation}")
        scores.append(layer_scores.squeeze(0).detach().cpu())
    return torch.stack(scores)


def analyze_record(
    rec: dict,
    path: Path,
    config: PoolCurveConfig,
    student_scores: torch.Tensor | None,
) -> list[dict[str, int | float | str]]:
    target = target_scores(rec, config.target_mode)
    needed = needed_mask(rec, target)
    rows = []
    for layer_id in range(target.shape[0]):
        target_row = target[layer_id].float()
        needed_row = needed[layer_id].bool()
        prompt_length = int(target_row.numel())
        for budget in config.budgets:
            rows.append(metric_row(path, rec, layer_id, budget, "oracle", target_row, needed_row, target_row))
            if student_scores is not None:
                rows.append(
                    metric_row(
                        path,
                        rec,
                        layer_id,
                        budget,
                        "student",
                        student_scores[layer_id].float(),
                        needed_row,
                        target_row,
                    )
                )
            rows.append(metric_row(path, rec, layer_id, budget, "first", first_scores(prompt_length), needed_row, target_row))
            rows.append(metric_row(path, rec, layer_id, budget, "last", last_scores(prompt_length), needed_row, target_row))
    return rows


def metric_row(
    path: Path,
    rec: dict,
    layer_id: int,
    budget: int,
    selector: str,
    selector_scores: torch.Tensor,
    needed: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, int | float | str]:
    pool = topk_mask(selector_scores, budget)
    needed_count = int(needed.sum().item())
    hit_count = int((pool & needed).sum().item())
    target_mass = float(target.sum().item())
    hit_mass = float(target[pool].sum().item())
    return {
        "path": str(path),
        "sample_id": str(rec.get("sample_id", path.stem)),
        "dataset": str(rec.get("dataset", path.parent.name)),
        "layer": layer_id,
        "budget": budget,
        "pool_size": int(pool.sum().item()),
        "selector": selector,
        "prompt_length": int(target.numel()),
        "needed_count": needed_count,
        "union_recall": hit_count / needed_count if needed_count else 1.0,
        "target_mass_recall": hit_mass / target_mass if target_mass > 0.0 else 1.0,
    }


def target_scores(rec: dict, target_mode: str) -> torch.Tensor:
    mode = target_mode
    if mode == "auto":
        if "future_frequency" in rec:
            mode = "frequency"
        elif "future_union_mask" in rec:
            mode = "union"
        else:
            mode = "score"
    match mode:
        case "frequency":
            if "future_frequency" in rec:
                return rec["future_frequency"].float()
            if "future_union_mask" in rec:
                return rec["future_union_mask"].float()
            raise RuntimeError("frequency target requires future_frequency or future_union_mask")
        case "score":
            return rec["teacher_norm"].float()
        case "union":
            if "future_union_mask" not in rec:
                raise RuntimeError("union target requires future_union_mask")
            return rec["future_union_mask"].float()
        case _:
            raise RuntimeError(f"unsupported target_mode: {target_mode}")


def needed_mask(rec: dict, target: torch.Tensor) -> torch.Tensor:
    if "future_union_mask" in rec:
        return rec["future_union_mask"].bool()
    return target > 0


def first_scores(length: int) -> torch.Tensor:
    return torch.arange(length, 0, -1, dtype=torch.float32)


def last_scores(length: int) -> torch.Tensor:
    return torch.arange(length, dtype=torch.float32)


def topk_mask(scores: torch.Tensor, budget: int) -> torch.Tensor:
    count = min(max(1, int(budget)), int(scores.numel()))
    indices = torch.topk(scores.float(), k=count, largest=True).indices
    mask = torch.zeros(scores.shape, dtype=torch.bool)
    mask.scatter_(dim=0, index=indices.cpu(), value=True)
    return mask


def summarize(rows: list[dict[str, int | float | str]], config: PoolCurveConfig) -> dict:
    grouped: dict[tuple[str, int], list[dict[str, int | float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["selector"]), int(row["budget"]))].append(row)
    curves = {}
    for (selector, budget), values in sorted(grouped.items()):
        curves.setdefault(selector, {})[str(budget)] = {
            "rows": len(values),
            "union_recall_mean": mean(float(row["union_recall"]) for row in values),
            "union_recall_p10": percentile((float(row["union_recall"]) for row in values), 10),
            "target_mass_recall_mean": mean(float(row["target_mass_recall"]) for row in values),
            "target_mass_recall_p10": percentile((float(row["target_mass_recall"]) for row in values), 10),
            "pool_size_mean": mean(float(row["pool_size"]) for row in values),
            "needed_count_mean": mean(float(row["needed_count"]) for row in values),
        }
    return {
        "teacher_root": str(config.teacher_root),
        "datasets": config.datasets,
        "budgets": config.budgets,
        "target_mode": config.target_mode,
        "student_path": str(config.student_path) if config.student_path is not None else None,
        "sample_count": len({str(row["path"]) for row in rows}),
        "layer_rows": len(rows),
        "curves": curves,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


if __name__ == "__main__":
    raise SystemExit(main())
