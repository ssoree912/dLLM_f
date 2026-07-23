from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "origin_cache_budget_1024_no_cache.md"

DATASETS = [
    ("Single-doc QA", "qasper", 128, 200, "qa_f1"),
    ("Single-doc QA", "multifieldqa_en", 64, 150, "qa_f1"),
    ("Single-doc QA", "narrativeqa", 128, 200, "qa_f1"),
    ("Multi-doc QA", "hotpotqa", 32, 200, "qa_f1"),
    ("Multi-doc QA", "2wikimqa", 32, 200, "qa_f1"),
    ("Multi-doc QA", "musique", 32, 200, "qa_f1"),
    ("Summarization", "gov_report", 512, 200, "rouge"),
    ("Summarization", "qmsum", 512, 200, "rouge"),
    ("Summarization", "multi_news", 512, 200, "rouge"),
    ("Few-shot", "trec", 64, 200, "classification"),
    ("Few-shot", "triviaqa", 32, 200, "qa_f1"),
    ("Few-shot", "samsum", 128, 200, "rouge"),
    ("Synthetic", "passage_count", 32, 200, "count"),
    ("Synthetic", "passage_retrieval_en", 32, 200, "retrieval"),
    ("Code", "lcc", 64, 500, "code_sim"),
    ("Code", "repobench-p", 64, 500, "code_sim"),
]

SCORE_KEYS = {
    "qa_f1": ("score,none", "qa_f1_score,none"),
    "rouge": ("score,none", "rouge_score,none"),
    "code_sim": ("score,none", "code_sim_score,none"),
    "classification": ("score,none", "classification_score,none"),
    "retrieval": ("score,none", "retrieval_score,none"),
    "count": ("score,none", "count_score,none"),
}

COLUMNS = {
    "origin": [
        "experiment/345/2026-07-15/results/llada_true_full_*_mlen2048_full/**/results_*.json",
    ],
    "128_cache": [
        "experiment/2026-07-14/results/"
        "lm_eval_student_balanced_2k_g32_e10_lr2e-5_chat_promptkv_b128_*_mlen2048_full/"
        "**/results_*.json",
    ],
    "budget_1024_no_cache": [
        "results/budget/"
        "lm_eval_student_poolactive_futurepool_train_300each_p1024_a128_mlen2048_all/"
        "**/results_*.json",
    ],
}


def dataset_name(task_name: str) -> str:
    return task_name.removeprefix("local_longbench_")


def score_value(row: dict, metric: str) -> float | None:
    for key in SCORE_KEYS[metric]:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def load_column(patterns: list[str]) -> dict[str, dict]:
    values: dict[str, dict] = {}
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    for path in sorted(files, key=lambda item: item.stat().st_mtime):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results") or {}
        configs = payload.get("configs") or {}
        for task, result in results.items():
            name = dataset_name(task)
            metric = metric_for_dataset(name)
            score = score_value(result, metric)
            if score is None:
                continue
            config = configs.get(task) or {}
            gen_kwargs = config.get("generation_kwargs") or {}
            values[name] = {
                "score": score,
                "n": result.get("sample_len"),
                "gen": gen_kwargs.get("gen_length") or gen_kwargs.get("max_gen_toks"),
                "path": str(path.relative_to(ROOT)),
            }
    return values


def metric_for_dataset(dataset: str) -> str:
    for _, name, _, _, metric in DATASETS:
        if name == dataset:
            return metric
    raise KeyError(dataset)


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> int:
    columns = {name: load_column(patterns) for name, patterns in COLUMNS.items()}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# LongBench Results: Origin vs 128 Cache vs Budget 1024 No Cache",
        "",
        f"Updated: {now}.",
        "",
        "- `origin`: origin LLaDA true-full, `is_feature_cache=False`, `is_cfg_cache=False`, `max_length=2048`.",
        "- `128_cache`: previous student prompt-KV B=128 results.",
        "- `budget_1024_no_cache`: student pool-active results, `student_pool_budget=1024`, `student_budget=128`, `is_feature_cache=False`.",
        "- `-`: no completed result JSON found for that column yet.",
        "",
        "| category | dataset | gen | n | metric | origin | 128_cache | budget_1024_no_cache |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]

    for category, dataset, gen, n, metric in DATASETS:
        origin = columns["origin"].get(dataset, {}).get("score")
        cache128 = columns["128_cache"].get(dataset, {}).get("score")
        budget = columns["budget_1024_no_cache"].get(dataset, {}).get("score")
        lines.append(
            f"| {category} | {dataset} | {gen} | {n} | {metric} | "
            f"{fmt(origin)} | {fmt(cache128)} | {fmt(budget)} |"
        )

    budget_done = [dataset for _, dataset, _, _, _ in DATASETS if dataset in columns["budget_1024_no_cache"]]
    budget_pending = [dataset for _, dataset, _, _, _ in DATASETS if dataset not in columns["budget_1024_no_cache"]]
    lines.extend(
        [
            "",
            "## Budget Progress",
            "",
            f"Completed: {len(budget_done)}/16.",
            "",
            "Done:",
            "",
            "- " + ", ".join(budget_done) if budget_done else "- none",
            "",
            "Pending:",
            "",
            "- " + ", ".join(budget_pending) if budget_pending else "- none",
            "",
        ]
    )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
