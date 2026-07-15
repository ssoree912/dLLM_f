from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (SCRIPT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.common import DEFAULT_MODEL_PATH
from revealed_answer.imitation_metrics import score_topk_metrics
from revealed_answer.student_model import PromptUtilityStudent, StudentConfig
from revealed_answer.train_config import parse_dtype

DEFAULT_TEACHER_ROOT = Path(
    "experiment/2026-07-14/results/revealed_answer_teacher_train_n5000"
)
DEFAULT_STUDENT_PATH = Path(
    "experiment/2026-07-14/results/"
    "revealed_answer_student_train_n5000_e30_lr5e-5/checkpoint-best"
)


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    teacher_root: Path
    student_path: Path
    model_path: Path
    output_dir: Path
    dataset: str
    split: str
    n_samples: int
    val_ratio: float
    seed: int
    limit: int
    top_k: list[int]
    device: str
    dtype: torch.dtype
    log_every: int


@dataclass(frozen=True, slots=True)
class ImitationRuntime:
    model: torch.nn.Module
    student: PromptUtilityStudent


def parse_args() -> ImitationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="2wikimultihopqa_train")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    return ImitationConfig(
        teacher_root=args.teacher_root,
        student_path=args.student,
        model_path=args.model,
        output_dir=args.output_dir,
        dataset=args.dataset,
        split=args.split,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit,
        top_k=args.top_k,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        log_every=args.log_every,
    )


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(config)
    files = select_teacher_files(config)
    sample_path = config.output_dir / "samples.jsonl"
    t0 = time.time()
    totals = make_metric_sums(config.top_k)
    layer_totals = [make_metric_sums(config.top_k) for _ in runtime.student.layer_indices]
    with sample_path.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(files, start=1):
            sample_metrics, layer_metrics = score_one_file(config, runtime, path)
            add_metric_sums(totals, sample_metrics)
            for layer_id, metrics in enumerate(layer_metrics):
                add_metric_sums(layer_totals[layer_id], metrics)
            handle.write(json.dumps({"path": str(path), "metrics": sample_metrics}) + "\n")
            handle.flush()
            if index % config.log_every == 0 or index == len(files):
                recall = totals[f"recall@{primary_k(config)}"] / index
                print(f"[imitation {index}/{len(files)}] recall@{primary_k(config)}={recall:.4f}", flush=True)
    summary = {
        "student": str(config.student_path),
        "teacher_root": str(config.teacher_root),
        "dataset": config.dataset,
        "split": config.split,
        "samples": len(files),
        "top_k": config.top_k,
        "metrics": mean_metric_sums(totals, len(files)),
        "layer_metrics": [mean_metric_sums(layer, len(files)) for layer in layer_totals],
        "elapsed": time.time() - t0,
        "sample_file": str(sample_path),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {summary}", flush=True)
    return 0


def build_runtime(config: ImitationConfig) -> ImitationRuntime:
    print(f"[load] model={config.model_path} dtype={config.dtype} device={config.device}", flush=True)
    model = AutoModel.from_pretrained(
        str(config.model_path),
        trust_remote_code=True,
        torch_dtype=config.dtype,
    ).to(config.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    student = load_student(config.student_path, config.device)
    return ImitationRuntime(model=model, student=student)


def load_student(checkpoint_dir: Path, device: str) -> PromptUtilityStudent:
    raw_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    student = PromptUtilityStudent(StudentConfig(**raw_config))
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    student.load_state_dict(state)
    student.to(device)
    student.eval()
    print(f"[student] checkpoint={checkpoint_dir}", flush=True)
    return student


def select_teacher_files(config: ImitationConfig) -> list[Path]:
    files = sorted((config.teacher_root / config.dataset).glob("*.pt"))
    if config.n_samples > 0:
        files = files[: config.n_samples]
    if not files:
        raise RuntimeError(f"no teacher files under {config.teacher_root / config.dataset}")
    rng = random.Random(config.seed)
    rng.shuffle(files)
    val_count = int(round(len(files) * config.val_ratio))
    val_count = min(max(0, val_count), max(0, len(files) - 1))
    match config.split:
        case "train":
            selected = files[val_count:]
        case "val":
            selected = files[:val_count]
        case "all":
            selected = files
        case unreachable:
            raise RuntimeError(f"unsupported split: {unreachable}")
    if config.limit > 0:
        selected = selected[: config.limit]
    return selected


@torch.inference_mode()
def score_one_file(
    config: ImitationConfig,
    runtime: ImitationRuntime,
    path: Path,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    hidden_states = compute_prompt_hidden_states(config, runtime, rec["prompt_input_ids"])
    prompt_indices = rec["prompt_token_indices"].to(config.device)
    question_indices = rec["question_token_indices"].to(config.device)
    teacher_norm = rec["teacher_norm"].float()
    layer_metrics: list[dict[str, float]] = []
    for layer_id in runtime.student.layer_indices:
        scores = runtime.student.forward_layer(
            layer_id,
            hidden_states[layer_id].float(),
            prompt_indices,
            question_indices,
        ).squeeze(0).cpu()
        layer_metrics.append(score_layer(scores, teacher_norm[layer_id], config.top_k))
    return mean_layer_metrics(layer_metrics, config.top_k), layer_metrics


@torch.inference_mode()
def compute_prompt_hidden_states(
    config: ImitationConfig,
    runtime: ImitationRuntime,
    prompt_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    input_ids = prompt_ids.unsqueeze(0).to(config.device)
    out = runtime.model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return tuple(state.detach() for state in out.hidden_states)


def score_layer(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    top_k: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in top_k:
        result = score_topk_metrics(student_scores, teacher_scores, k)
        for name, value in result.to_dict().items():
            metrics[f"{name}@{k}"] = value
    return metrics


def mean_layer_metrics(layer_metrics: list[dict[str, float]], top_k: list[int]) -> dict[str, float]:
    totals = make_metric_sums(top_k)
    for metrics in layer_metrics:
        add_metric_sums(totals, metrics)
    return mean_metric_sums(totals, len(layer_metrics))


def make_metric_sums(top_k: list[int]) -> dict[str, float]:
    sums: dict[str, float] = {}
    for k in top_k:
        for name in ("recall", "ndcg", "teacher_mass"):
            sums[f"{name}@{k}"] = 0.0
    return sums


def add_metric_sums(totals: dict[str, float], metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        totals[name] += value


def mean_metric_sums(totals: dict[str, float], count: int) -> dict[str, float]:
    denominator = max(1, count)
    return {name: value / denominator for name, value in totals.items()}


def primary_k(config: ImitationConfig) -> int:
    return 128 if 128 in config.top_k else config.top_k[0]


if __name__ == "__main__":
    raise SystemExit(main())
