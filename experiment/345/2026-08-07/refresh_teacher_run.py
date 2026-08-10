"""Driver: collect refresh (drift) teacher shards for samsum/trec.

Plain prompts + existing importance student (300each) for the keep set -- same regime as
refresh_gate.py, so teacher/student/eval stay consistent. One .pt per sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from dllm_cache.budget.measure_decode_latency import load_student, student_scores
from refresh_gate import TASKS, build_prompt
from refresh_teacher import collect_refresh_teacher


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--skip", type=int, default=0, help="skip first N rows (disjoint train/eval split)")
    p.add_argument("--budget-ratio", type=float, default=0.5)
    p.add_argument("--refresh-tokens", type=int, default=0, help="behavior-policy R (0 -> budget/4)")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    task_cfg = TASKS[a.task]
    gen_length = task_cfg["gen_length"]
    cap = a.max_length - gen_length
    budget = max(1, round(cap * a.budget_ratio))
    R = a.refresh_tokens if a.refresh_tokens > 0 else max(1, budget // 4)

    model = AutoModel.from_pretrained(
        str(a.model), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(a.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), trust_remote_code=True)
    student = load_student(a.student, a.device)

    rows = []
    with open(task_cfg["data"], encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < a.skip:
                continue
            if len(rows) >= a.samples:
                break
            rows.append(json.loads(line))

    out_dir = a.output_root / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"task={a.task} samples={len(rows)} cap={cap} budget={budget} R={R} gen={gen_length}", flush=True)

    saved = 0
    for idx, row in enumerate(rows):
        sid = str(row.get("_id", idx))
        out_path = out_dir / f"{idx}-{sid[:16]}.pt"
        if out_path.exists():
            saved += 1
            continue
        ids = tokenizer(build_prompt(row, task_cfg), add_special_tokens=False)["input_ids"][-cap:]
        pid = torch.tensor([ids], dtype=torch.long, device=a.device)
        with torch.inference_mode():
            scores = student_scores(model, student, pid)
            keep = torch.topk(scores.mean(dim=0), k=budget, largest=True).indices.sort().values
            shard = collect_refresh_teacher(
                pid, model, keep, refresh_tokens=R,
                steps=gen_length, gen_length=gen_length, block_length=a.block_length,
            )
        shard["dataset"] = a.task
        shard["sample_id"] = sid
        torch.save(shard, out_path)
        saved += 1
        print(f"[refresh-teacher {idx+1}/{len(rows)}] saved={out_path.name} "
              f"r*_nz={int((shard['r_star']>0).sum())}", flush=True)
    print(f"[done] saved={saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
