#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiment/345/2026-07-15"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/model/LLaDA-8B-Instruct}"
LOCAL_TASK_PATH="${LOCAL_TASK_PATH:-${EXPERIMENT_ROOT}/tasks/longbench_local}"
DATASETS_CACHE_DIR="${DATASETS_CACHE_DIR:-${TMPDIR:-/tmp}/dllm_hf_datasets_cache}"
MODULES_CACHE_DIR="${MODULES_CACHE_DIR:-${TMPDIR:-/tmp}/dllm_hf_modules_cache}"
REPORT_PATH="${REPORT_PATH:-${REPO_ROOT}/results/origin_2048_notemp.md}"
mkdir -p "${DATASETS_CACHE_DIR}" "${MODULES_CACHE_DIR}"

export HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}"
export HF_MODULES_CACHE="${MODULES_CACHE_DIR}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASKKV_ENABLED=0

MODEL="${MODEL:-${MODEL_DIR}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/accelerate_config_single_gpu.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ -x "${PYTHON_BIN}" ]]; then
  ACCELERATE_CMD=("${PYTHON_BIN}" -m accelerate.commands.accelerate_cli)
else
  ACCELERATE_CMD=(accelerate)
fi

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
LIMIT="${LIMIT:-full}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TASKS="${TASKS:-2wikimqa hotpotqa musique triviaqa passage_count passage_retrieval_en multifieldqa_en trec lcc repobench-p qasper narrativeqa samsum gov_report qmsum multi_news}"
COMMON_MODEL_ARGS="pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=${MODEL_MAX_LENGTH}"

gen_length_for_task() {
  case "$1" in
    2wikimqa|hotpotqa|musique|passage_count|passage_retrieval_en|triviaqa) echo 32 ;;
    lcc|repobench-p|trec|multifieldqa_en) echo 64 ;;
    narrativeqa|samsum|qasper) echo 128 ;;
    gov_report|multi_news|qmsum) echo 512 ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}

update_report() {
  REPORT_PATH="${REPORT_PATH}" \
  REPO_ROOT="${REPO_ROOT}" \
  RESULTS_ROOT="${EXPERIMENT_ROOT}/results" \
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
results_root = Path(os.environ["RESULTS_ROOT"])
report_path = Path(os.environ["REPORT_PATH"])
max_length = os.environ["MODEL_MAX_LENGTH"]

tasks = [
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


def latest_result(dataset: str) -> Path | None:
    pattern = f"llada_true_full_nochat_{dataset}_mlen{max_length}_full/**/results_*.json"
    matches = list(results_root.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def result_score(path: Path, dataset: str) -> float | None:
    data = json.loads(path.read_text())
    task_result = data.get("results", {}).get(f"local_longbench_{dataset}", {})
    score = task_result.get("score,none")
    if isinstance(score, (int, float)):
        return float(score)
    for key, value in task_result.items():
        if key.endswith(",none") and not key.endswith("_stderr,none") and isinstance(value, (int, float)):
            return float(value)
    return None


rows = []
json_rows = []
done = 0
for category, dataset, gen, sample_count, metric in tasks:
    path = latest_result(dataset)
    score = None if path is None else result_score(path, dataset)
    status = "pending" if score is None else "done"
    if score is not None:
        done += 1
    score_text = "-" if score is None else f"{score:.4f}"
    rows.append(
        f"| {category} | {dataset} | {gen} | {sample_count} | {metric} | {score_text} | {status} |"
    )
    if path is None:
        rel_path = "-"
    else:
        rel_path = f"`{path.relative_to(repo_root)}`"
    json_rows.append(f"| {dataset} | {rel_path} |")

now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(
        [
            "# Origin True-Full Results (2048, No Chat Template)",
            "",
            "기준 설정: origin LLaDA, `is_feature_cache=False`, `is_cfg_cache=False`, `max_length=2048`.",
            "`--apply_chat_template`와 `--fewshot_as_multiturn`는 사용하지 않았습니다.",
            f"최종 업데이트: {now}. 완료: {done}/16.",
            "",
            "| category | dataset | gen | n | metric | score | status |",
            "|---|---|---:|---:|---|---:|---|",
            *rows,
            "",
            "결과 JSON 기준:",
            "",
            "| dataset | result json |",
            "|---|---|",
            *json_rows,
            "",
        ]
    ),
    encoding="utf-8",
)
PY
}

has_completed_result() {
  local task="$1"
  find "${EXPERIMENT_ROOT}/results/llada_true_full_nochat_${task}_mlen${MODEL_MAX_LENGTH}_full" \
    -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .
}

run_task() {
  local task="$1"
  local gen_length
  gen_length="$(gen_length_for_task "${task}")"
  local limit_label="limit${LIMIT}"
  local limit_args=(--limit "${LIMIT}")
  if [[ "${LIMIT}" == "full" || "${LIMIT}" == "none" || "${LIMIT}" == "0" ]]; then
    limit_label="full"
    limit_args=()
  fi
  local output_path="${EXPERIMENT_ROOT}/results/llada_true_full_nochat_${task}_mlen${MODEL_MAX_LENGTH}_${limit_label}"
  if [[ "${limit_label}" == "full" && "${SKIP_COMPLETED}" == "1" ]] && has_completed_result "${task}"; then
    printf '[skip] true_full_nochat task=%s existing output=%s\n' "${task}" "${output_path}"
    update_report
    return 0
  fi
  local cmd=(
    "${ACCELERATE_CMD[@]}" launch --config_file "${ACCELERATE_CONFIG}" evaluation_script.py run
    --model LLaDA
    --tasks "local_longbench_${task}"
    --include_path "${LOCAL_TASK_PATH}"
    --batch_size 1
    --model_args "${COMMON_MODEL_ARGS}"
    --gen_kwargs "block_length=${BLOCK_LENGTH},gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0"
    "${limit_args[@]}"
    --num_fewshot 0
    --output_path "${output_path}"
    --log_samples
    --trust_remote_code
  )
  printf '[start] true_full_nochat task=%s gen_length=%s limit=%s output=%s\n' "${task}" "${gen_length}" "${limit_label}" "${output_path}"
  cd "${REPO_ROOT}"
  "${cmd[@]}"
  update_report
}

update_report
for task in ${TASKS}; do
  run_task "${task}"
done
update_report
