from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoModel

from dllm_cache.budget.student_model import (
    PromptUtilityStudent,
    StudentConfig,
    student_loss,
)
from dllm_cache.budget.train_config import TrainConfig


class TextSink(Protocol):
    def write(self, text: str) -> int:
        ...

    def flush(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class TrainingRuntime:
    config: TrainConfig
    model: torch.nn.Module
    student: PromptUtilityStudent
    optimizer: torch.optim.Optimizer
    best_metric: float | None


@dataclass(frozen=True, slots=True)
class TeacherSplit:
    train_files: list[Path]
    val_files: list[Path]


def build_runtime(config: TrainConfig, best_metric: float | None = None) -> TrainingRuntime:
    model = load_frozen_model(config)
    student = build_student(model, config)
    if config.resume_from is not None:
        load_student_checkpoint(student, config.resume_from)
    student = student.to(config.device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return TrainingRuntime(
        config=config,
        model=model,
        student=student,
        optimizer=optimizer,
        best_metric=best_metric,
    )


def load_frozen_model(config: TrainConfig) -> torch.nn.Module:
    print(f"[load] model={config.model_path} dtype={config.dtype} device={config.device}", flush=True)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_student(model: torch.nn.Module, config: TrainConfig) -> PromptUtilityStudent:
    model_config = getattr(model, "config")
    student_config = StudentConfig(
        layer_count=int(getattr(model_config, "n_layers", 32)),
        hidden_dim=int(getattr(model_config, "d_model", 4096)),
        proj_dim=config.proj_dim,
        mlp_dim=config.mlp_dim,
        heads=("attention", "delta") if config.target_mode == "attention_delta" else ("score",),
    )
    student = PromptUtilityStudent(student_config)
    params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] layers={student_config.layer_count} params={params:,}", flush=True)
    return student


def load_student_checkpoint(student: PromptUtilityStudent, checkpoint_dir: Path) -> None:
    state_path = checkpoint_dir / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    student.load_state_dict(state)
    print(f"[resume] loaded student={checkpoint_dir}", flush=True)


def split_teacher_files(config: TrainConfig) -> TeacherSplit:
    files: list[Path] = []
    for dataset in config.datasets:
        files.extend(sorted((config.teacher_root / dataset).glob("*.pt")))
    if config.n_samples > 0:
        files = files[: config.n_samples]
    if not files:
        raise RuntimeError(f"no teacher files found under {config.teacher_root}")
    rng = random.Random(config.seed)
    rng.shuffle(files)
    val_count = int(round(len(files) * config.val_ratio))
    val_count = min(max(0, val_count), max(0, len(files) - 1))
    return TeacherSplit(train_files=files[val_count:], val_files=files[:val_count])


def run_training(runtime: TrainingRuntime, split: TeacherSplit, log_file: TextSink) -> None:
    t0 = time.time()
    best_metric = runtime.best_metric
    if best_metric is not None:
        print(f"[resume] previous best_metric={best_metric:.6f}", flush=True)
    for epoch in range(1, runtime.config.epochs + 1):
        runtime.student.train()
        train_loss = 0.0
        for step, path in enumerate(split.train_files, start=1):
            loss, primary, rank, topk = train_one_file(runtime, path)
            train_loss += loss
            if step % runtime.config.log_every == 0 or step == len(split.train_files):
                print_train_step(runtime, epoch, step, train_loss, primary, rank, topk, split)
        val_loss = evaluate(runtime, split.val_files)
        train_mean = train_loss / max(1, len(split.train_files))
        metric = val_loss if val_loss is not None else train_mean
        if best_metric is None or metric < best_metric:
            best_metric = metric
            runtime.student.save_pretrained(runtime.config.output_dir / "checkpoint-best")
            print(f"[epoch {epoch}] saved checkpoint-best metric={metric:.6f}", flush=True)
        runtime.student.save_pretrained(runtime.config.output_dir / "checkpoint-last")
        print(f"[epoch {epoch}] saved checkpoint-last", flush=True)
        record = {
            "epoch": epoch,
            "train_loss": train_mean,
            "val_loss": val_loss,
            "best_metric": best_metric,
            "elapsed": time.time() - t0,
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(f"[epoch {epoch}] {record}", flush=True)


def print_train_step(
    runtime: TrainingRuntime,
    epoch: int,
    step: int,
    train_loss: float,
    primary: float,
    rank: float,
    topk: float,
    split: TeacherSplit,
) -> None:
    print(
        f"[epoch {epoch}/{runtime.config.epochs} step {step}/{len(split.train_files)}] "
        f"loss={train_loss / step:.6f} primary={primary:.6f} rank={rank:.6f} topk={topk:.6f}",
        flush=True,
    )


def train_one_file(runtime: TrainingRuntime, path: Path) -> tuple[float, float, float, float]:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    hidden_states = compute_prompt_hidden_states(runtime, rec["prompt_input_ids"])
    prompt_indices = rec["prompt_token_indices"].to(runtime.config.device)
    question_indices = rec["question_token_indices"].to(runtime.config.device)
    if runtime.config.target_mode == "attention_delta":
        return train_one_file_attention_delta(runtime, rec, hidden_states, prompt_indices, question_indices)
    teacher_targets = teacher_targets_from_record(rec, runtime.config).to(runtime.config.device)
    target_count = int(teacher_targets.shape[0])
    runtime.optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), dtype=torch.float32, device=runtime.config.device)
    primary_total = rank_total = topk_total = 0.0
    for layer_id in runtime.student.layer_indices:
        scores = runtime.student.forward_layer(
            layer_id,
            hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        )
        for teacher_target in teacher_targets:
            target = teacher_target[layer_id].float().unsqueeze(0)
            loss, primary, rank, topk = student_loss_from_runtime(runtime, scores, target)
            loss_total = loss_total + (loss / target_count)
            primary_total += float(primary.detach().cpu())
            rank_total += float(rank.detach().cpu())
            topk_total += float(topk.detach().cpu())
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(runtime.student.parameters(), runtime.config.max_grad_norm)
    runtime.optimizer.step()
    loss_count = len(runtime.student.layer_indices) * target_count
    return (
        float(loss_total.detach().cpu()),
        primary_total / loss_count,
        rank_total / loss_count,
        topk_total / loss_count,
    )


def train_one_file_attention_delta(
    runtime: TrainingRuntime,
    rec: dict,
    hidden_states: tuple[torch.Tensor, ...],
    prompt_indices: torch.Tensor,
    question_indices: torch.Tensor,
) -> tuple[float, float, float, float]:
    attention_target = rec["teacher_norm"].float().to(runtime.config.device)
    delta_target = rec["delta_norm"].float().to(runtime.config.device)
    runtime.optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), dtype=torch.float32, device=runtime.config.device)
    primary_total = rank_total = topk_total = 0.0
    for layer_id in runtime.student.layer_indices:
        hidden = hidden_states[layer_id].float()
        for head, target_by_layer in (
            ("attention", attention_target),
            ("delta", delta_target),
        ):
            scores = runtime.student.forward_layer(
                layer_id,
                hidden,
                prompt_indices,
                question_indices,
                head=head,
            )
            target = target_by_layer[layer_id].float().unsqueeze(0)
            loss, primary, rank, topk = student_loss_from_runtime(runtime, scores, target)
            loss_total = loss_total + (loss / 2.0)
            primary_total += float(primary.detach().cpu())
            rank_total += float(rank.detach().cpu())
            topk_total += float(topk.detach().cpu())
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(runtime.student.parameters(), runtime.config.max_grad_norm)
    runtime.optimizer.step()
    loss_count = len(runtime.student.layer_indices) * 2
    return (
        float(loss_total.detach().cpu()),
        primary_total / loss_count,
        rank_total / loss_count,
        topk_total / loss_count,
    )


@torch.no_grad()
def evaluate(runtime: TrainingRuntime, val_files: list[Path]) -> float | None:
    if not val_files:
        return None
    runtime.student.eval()
    total = 0.0
    for path in val_files:
        total += evaluate_one_file(runtime, path)
    return total / len(val_files)


def evaluate_one_file(runtime: TrainingRuntime, path: Path) -> float:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    hidden_states = compute_prompt_hidden_states(runtime, rec["prompt_input_ids"])
    prompt_indices = rec["prompt_token_indices"].to(runtime.config.device)
    question_indices = rec["question_token_indices"].to(runtime.config.device)
    if runtime.config.target_mode == "attention_delta":
        return evaluate_one_file_attention_delta(runtime, rec, hidden_states, prompt_indices, question_indices)
    teacher_targets = teacher_targets_from_record(rec, runtime.config).to(runtime.config.device)
    target_count = int(teacher_targets.shape[0])
    file_loss = 0.0
    for layer_id in runtime.student.layer_indices:
        scores = runtime.student.forward_layer(
            layer_id,
            hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        )
        for teacher_target in teacher_targets:
            target = teacher_target[layer_id].float().unsqueeze(0)
            loss, _mse, _rank, _topk = student_loss_from_runtime(runtime, scores, target)
            file_loss += float((loss / target_count).detach().cpu())
    return file_loss


def evaluate_one_file_attention_delta(
    runtime: TrainingRuntime,
    rec: dict,
    hidden_states: tuple[torch.Tensor, ...],
    prompt_indices: torch.Tensor,
    question_indices: torch.Tensor,
) -> float:
    attention_target = rec["teacher_norm"].float().to(runtime.config.device)
    delta_target = rec["delta_norm"].float().to(runtime.config.device)
    file_loss = 0.0
    for layer_id in runtime.student.layer_indices:
        hidden = hidden_states[layer_id].float()
        for head, target_by_layer in (
            ("attention", attention_target),
            ("delta", delta_target),
        ):
            scores = runtime.student.forward_layer(
                layer_id,
                hidden,
                prompt_indices,
                question_indices,
                head=head,
            )
            target = target_by_layer[layer_id].float().unsqueeze(0)
            loss, _primary, _rank, _topk = student_loss_from_runtime(runtime, scores, target)
            file_loss += float((loss / 2.0).detach().cpu())
    return file_loss


@torch.no_grad()
def compute_prompt_hidden_states(
    runtime: TrainingRuntime,
    prompt_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    input_ids = prompt_ids.unsqueeze(0).to(runtime.config.device)
    out = runtime.model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return tuple(state.detach() for state in out.hidden_states)


def student_loss_from_runtime(
    runtime: TrainingRuntime,
    scores: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return student_loss(
        scores,
        target,
        runtime.config.rank_weight,
        runtime.config.rank_margin,
        runtime.config.rank_top_ratio,
        runtime.config.rank_bottom_ratio,
        runtime.config.topk_weight,
        runtime.config.topk_k,
        runtime.config.topk_positive_weight,
        runtime.config.loss_mode,
        runtime.config.bce_positive_weight,
        runtime.config.rank_input,
    )


def teacher_targets_from_record(rec: dict, config: TrainConfig) -> torch.Tensor:
    match config.target_mode:
        case "score":
            return rec["teacher_norm"].float().unsqueeze(0)
        case "frequency":
            if "future_frequency" in rec:
                return rec["future_frequency"].float().unsqueeze(0)
            if "future_union_mask" in rec:
                return rec["future_union_mask"].float().unsqueeze(0)
            raise RuntimeError("target-mode=frequency requires future_frequency or future_union_mask in teacher record")
        case "union":
            if "future_union_mask" not in rec:
                raise RuntimeError("target-mode=union requires future_union_mask in teacher record")
            return rec["future_union_mask"].float().unsqueeze(0)
        case "drift":
            # How far a prompt position's value vector travels once the suffix
            # starts filling in; the signal that says which positions a cache may
            # freeze. Weighted by how much the suffix actually reads the position.
            if "drift_weighted" in rec:
                return rec["drift_weighted"].float().unsqueeze(0)
            if "drift_cumulative" in rec:
                return rec["drift_cumulative"].float().unsqueeze(0)
            raise RuntimeError("target-mode=drift requires drift_weighted or drift_cumulative in teacher record")
        case "hybrid_mask":
            if "hybrid_mask" not in rec:
                raise RuntimeError("target-mode=hybrid_mask requires hybrid_mask in label record")
            target = rec["hybrid_mask"].float()
            if target.ndim != 2:
                raise RuntimeError("hybrid_mask must have shape [layer, prompt]")
            return target.unsqueeze(0)
        case "delta":
            if "delta_norm" not in rec:
                raise RuntimeError("target-mode=delta requires delta_norm in teacher record")
            target = rec["delta_norm"].float()
            if target.ndim != 2:
                raise RuntimeError("delta_norm must have shape [layer, prompt]")
            return target.unsqueeze(0)
        case "step":
            if "future_step_masks" not in rec:
                raise RuntimeError("target-mode=step requires future_step_masks in teacher record")
            step_masks = rec["future_step_masks"].float()
            if step_masks.ndim != 3:
                raise RuntimeError("future_step_masks must have shape [step, layer, prompt]")
            if int(step_masks.shape[0]) == 0:
                raise RuntimeError("future_step_masks must contain at least one denoising step")
            return step_masks
        case _:
            raise RuntimeError(f"unsupported target_mode: {config.target_mode}")
