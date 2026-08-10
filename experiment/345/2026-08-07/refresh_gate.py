"""Refresh-axis accuracy gate — custom harness (NOT lm-eval).

Why separate:
  The refresh oracle/teacher needs per-step control (a full shadow forward each
  step to compute r* = importance x value-staleness). lm-eval's generate_until
  gives no per-step hook, so this runs its own loop in the 345 (this-server) tree.

Consistency:
  The importance student was trained on PLAIN prompts (no chat template), so this
  harness builds prompts exactly like the task doc_to_text WITHOUT --apply_chat_template.
  Numbers here are therefore a SEPARATE axis from the lm-eval report numbers; compare
  methods WITHIN this harness (relative gate), not against the 0.38xx lm-eval values.

v0 (this file): full / frozen / periodic(dynkv) baselines + scoring, to validate the
harness. The oracle + uniform + cosine + student refresh modes are added on top next.

Keep set: v1 uses a shared keep set across layers (selection_mode=global), budget = a
ratio of the (max_length - gen_length) prompt. Refresh is the axis under study.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.measure_decode_latency import load_student, student_scores
from dllm_cache.budget.prompt_kv_cache import build_prompt_kv_cache
from dllm_cache.budget.prompt_kv_generate import generate_with_prompt_kv
from dllm_cache.budget.dynamic_prompt_kv import (
    build_dynamic_prompt_kv_cache,
    generate_with_dynamic_prompt_kv,
)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_oracle import generate_with_oracle_refresh
from lm_eval.tasks.longbench.metrics import (
    get_rouge_with_score,
    get_classification_with_score,
)
from utils.generate_function import generate

DATA_ROOT = "/mnt/srv/home/dlpcg.325/dllm/data/longbench"

# doc_to_text templates copied verbatim from the task yamls (plain, no chat template).
TASKS = {
    "samsum": {
        "data": f"{DATA_ROOT}/samsum.jsonl",
        "instruction": "Summarize the dialogue into a few short sentences. The following are some examples.",
        "query_field": "input",          # samsum yaml uses {{input}}
        "gen_length": 128,
        "metric": get_rouge_with_score,
        "metric_key": "rouge_score",
    },
    "trec": {
        "data": f"{DATA_ROOT}/trec.jsonl",
        "instruction": "Please determine the type of the question below. Here are some examples of questions.",
        "query_field": "question",       # trec yaml uses {{question}}
        "gen_length": 64,
        "metric": get_classification_with_score,
        "metric_key": "classification_score",
    },
}


@dataclass(frozen=True, slots=True)
class GateConfig:
    model_path: Path
    student_path: Path
    output_path: Path
    task: str
    methods: tuple[str, ...]
    samples: int
    budget_ratio: float
    refresh_intervals: tuple[int, ...]
    refresh_tokens: tuple[int, ...]
    student_refresh_ckpt: Path | None
    block_length: int
    max_length: int
    device: str


def build_prompt(row: dict, task_cfg: dict) -> str:
    # Mirrors doc_to_text: "<instruction>\n\n{{context}}\n{{query}}", then the
    # dataset's answer_prefix (e.g. trec "Type:") so the model completes the answer.
    # samsum has no answer_prefix (its input already ends with "Summary: ").
    context = str(row["context"]).strip()
    query = str(row.get(task_cfg["query_field"], "")).strip()
    prompt = f"{task_cfg['instruction']}\n\n{context}\n{query}"
    prefix = str(row.get("answer_prefix", "") or "").strip()
    if prefix:
        prompt = f"{prompt}\n{prefix}"
    return prompt


def score_one(task_cfg: dict, row: dict, gen_tokens: torch.Tensor, tokenizer) -> float:
    text = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True)
    pred = text.split("\n")[0]          # emulate until=["\n"]
    out = task_cfg["metric"](row, [pred])
    return float(out["score"])


_REFRESH_STUDENT = None


def run_method(name, model, student, prompt_ids, gen, budget, refresh):
    if name == "full":
        return generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            model=model, **gen,
        )
    scores = student_scores(model, student, prompt_ids)
    if name == "frozen":
        cache = build_prompt_kv_cache(model, prompt_ids, budget=budget, teacher_scores=scores)
        return generate_with_prompt_kv(input_ids=prompt_ids, model=model, prompt_cache=cache, **gen)
    if name == "dynkv":
        cache = build_dynamic_prompt_kv_cache(
            model, prompt_ids, budget=budget, teacher_scores=scores, selection_mode="global",
        )
        return generate_with_dynamic_prompt_kv(
            input_ids=prompt_ids, model=model, prompt_cache=cache,
            refresh_interval=refresh, **gen,
        )
    if name == "oracle":
        # shared keep set (global): top-budget by layer-mean importance
        keep = torch.topk(scores.mean(dim=0), k=budget, largest=True).indices.sort().values
        return generate_with_oracle_refresh(
            input_ids=prompt_ids, model=model, keep_indices=keep, refresh_tokens=refresh,
            steps=gen["steps"], gen_length=gen["gen_length"], block_length=gen["block_length"],
            temperature=gen["temperature"], mask_id=126336,
        )
    if name == "student_refresh":
        from refresh_student_infer import generate_with_student_refresh
        keep = torch.topk(scores.mean(dim=0), k=budget, largest=True).indices.sort().values
        return generate_with_student_refresh(
            prompt_ids, model, keep, _REFRESH_STUDENT, refresh_tokens=refresh,
            steps=gen["steps"], gen_length=gen["gen_length"], block_length=gen["block_length"],
            mask_id=126336,
        )
    raise RuntimeError(f"unsupported method: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    task_cfg = TASKS[cfg.task]
    gen_length = task_cfg["gen_length"]
    cap = cfg.max_length - gen_length
    budget = max(1, round(cap * cfg.budget_ratio))
    gen = dict(
        steps=gen_length, gen_length=gen_length, block_length=cfg.block_length,
        temperature=0.0, cfg_scale=0.0,
    )

    model = AutoModel.from_pretrained(
        str(cfg.model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(cfg.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_path), trust_remote_code=True)
    student = load_student(cfg.student_path, cfg.device)

    global _REFRESH_STUDENT
    if "student_refresh" in cfg.methods:
        from refresh_student_infer import load_refresh_student
        _REFRESH_STUDENT = load_refresh_student(cfg.student_refresh_ckpt, cfg.device)

    rows: list[dict] = []
    with open(task_cfg["data"], encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= cfg.samples:
                break
            rows.append(json.loads(line))
    prompts = []
    for row in rows:
        ids = tokenizer(build_prompt(row, task_cfg), add_special_tokens=False)["input_ids"][-cap:]
        prompts.append(torch.tensor([ids], dtype=torch.long, device=cfg.device))

    print(f"task={cfg.task}  samples={len(rows)}  cap={cap}  budget={budget} ({cfg.budget_ratio:.0%})  gen={gen_length}", flush=True)

    # expand dynkv into one entry per refresh interval
    plan = []
    for m in cfg.methods:
        if m == "dynkv":
            for k in cfg.refresh_intervals:
                plan.append((f"dynkv_r{k}", "dynkv", k))
        elif m == "oracle":
            for rt in cfg.refresh_tokens:
                plan.append((f"oracle_R{rt}", "oracle", rt))
        elif m == "student_refresh":
            for rt in cfg.refresh_tokens:
                plan.append((f"student_R{rt}", "student_refresh", rt))
        else:
            plan.append((m, m, 0))

    summary = {}
    for label, method, refresh in plan:
        vals = []
        with torch.inference_mode():
            for row, pid in zip(rows, prompts):
                out = run_method(method, model, student, pid, gen, budget, refresh)
                vals.append(score_one(task_cfg, row, out, tokenizer))
                torch.cuda.empty_cache()
        mean = sum(vals) / len(vals)
        se = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        summary[label] = {"n": len(vals), task_cfg["metric_key"]: mean, "stderr": se, "budget": budget}
        print(f"{label:14s}  {task_cfg['metric_key']} {mean:.4f} ± {se:.4f}  (n={len(vals)})", flush=True)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {cfg.output_path}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> GateConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--methods", nargs="+", default=["full", "frozen", "dynkv"])
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--budget-ratio", type=float, default=0.5)
    p.add_argument("--refresh-intervals", type=int, nargs="+", default=[1, 4])
    p.add_argument("--refresh-tokens", type=int, nargs="+", default=[240])
    p.add_argument("--student-refresh-ckpt", type=Path, default=None)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    return GateConfig(
        model_path=a.model, student_path=a.student, output_path=a.output,
        task=a.task, methods=tuple(a.methods), samples=a.samples,
        budget_ratio=a.budget_ratio, refresh_intervals=tuple(a.refresh_intervals),
        refresh_tokens=tuple(a.refresh_tokens),
        student_refresh_ckpt=a.student_refresh_ckpt,
        block_length=a.block_length, max_length=a.max_length, device=a.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
