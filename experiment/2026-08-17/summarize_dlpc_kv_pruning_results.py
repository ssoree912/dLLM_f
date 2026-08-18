from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: summarize_dlpc_kv_pruning_results.py <eval_root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    rows: list[tuple[str, str, int | None, float, float | None, str]] = []
    for path in sorted(root.glob("*/*/*/results_*.json")):
        category = path.relative_to(root).parts[0]
        task_dir = path.relative_to(root).parts[1]
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results") or {}
        if not results:
            continue
        task_name, metrics = next(iter(results.items()))
        score_key = next((key for key in metrics if key.endswith(",none") and "stderr" not in key), None)
        stderr_key = next((key for key in metrics if key.endswith("_stderr,none")), None)
        if score_key is None:
            continue
        rows.append(
            (
                category,
                task_dir or task_name.removeprefix("local_longbench_"),
                metrics.get("sample_len"),
                float(metrics[score_key]),
                float(metrics[stderr_key]) if stderr_key is not None else None,
                path.as_posix(),
            )
        )

    if not rows:
        print(f"[summary] no result json found under {root}")
        return 0

    print("| category | task | samples | score | stderr | result |")
    print("|---|---|---:|---:|---:|---|")
    for category, task, sample_len, score, stderr, path in rows:
        stderr_text = "" if stderr is None else f"{stderr:.6f}"
        samples_text = "" if sample_len is None else str(sample_len)
        print(f"| {category} | {task} | {samples_text} | {score:.6f} | {stderr_text} | `{path}` |")
    macro = sum(row[3] for row in rows) / len(rows)
    weighted_den = sum(row[2] or 0 for row in rows)
    weighted = (
        sum(row[3] * (row[2] or 0) for row in rows) / weighted_den
        if weighted_den > 0
        else macro
    )
    print(f"\n[summary] tasks={len(rows)} macro={macro:.6f} weighted={weighted:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
