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
            loss, mse, rank, topk = train_one_file(runtime, path)
            train_loss += loss
            if step % runtime.config.log_every == 0 or step == len(split.train_files):
                print_train_step(runtime, epoch, step, train_loss, mse, rank, topk, split)
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
    mse: float,
    rank: float,
    topk: float,
    split: TeacherSplit,
) -> None:
    print(
        f"[epoch {epoch}/{runtime.config.epochs} step {step}/{len(split.train_files)}] "
        f"loss={train_loss / step:.6f} mse={mse:.6f} rank={rank:.6f} topk={topk:.6f}",
        flush=True,
    )


def train_one_file(runtime: TrainingRuntime, path: Path) -> tuple[float, float, float, float]:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    hidden_states = compute_prompt_hidden_states(runtime, rec["prompt_input_ids"])
    prompt_indices = rec["prompt_token_indices"].to(runtime.config.device)
    question_indices = rec["question_token_indices"].to(runtime.config.device)
    teacher_norm = rec["teacher_norm"].to(runtime.config.device)
    runtime.optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), dtype=torch.float32, device=runtime.config.device)
    mse_total = rank_total = topk_total = 0.0
    for layer_id in runtime.student.layer_indices:
        scores = runtime.student.forward_layer(
            layer_id,
            hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        )
        target = teacher_norm[layer_id].float().unsqueeze(0)
        loss, mse, rank, topk = student_loss_from_runtime(runtime, scores, target)
        loss_total = loss_total + loss
        mse_total += float(mse.detach().cpu())
        rank_total += float(rank.detach().cpu())
        topk_total += float(topk.detach().cpu())
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(runtime.student.parameters(), runtime.config.max_grad_norm)
    runtime.optimizer.step()
    layer_count = len(runtime.student.layer_indices)
    return (
        float(loss_total.detach().cpu()),
        mse_total / layer_count,
        rank_total / layer_count,
        topk_total / layer_count,
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
    teacher_norm = rec["teacher_norm"].to(runtime.config.device)
    file_loss = 0.0
    for layer_id in runtime.student.layer_indices:
        scores = runtime.student.forward_layer(
            layer_id,
            hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        )
        target = teacher_norm[layer_id].float().unsqueeze(0)
        loss, _mse, _rank, _topk = student_loss_from_runtime(runtime, scores, target)
        file_loss += float(loss.detach().cpu())
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
    )
