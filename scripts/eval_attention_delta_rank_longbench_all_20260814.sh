#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# The workspace policy reserves physical GPU 2. Inside this process it is cuda:0.
PHYSICAL_GPU=2
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/opt/conda/envs/dllm/bin/python
MODEL=/workspace/dllm/model/LLaDA-8B-Instruct
TASK_ROOT=experiment/345/2026-08-13/tasks/longbench_full_local
KEEP_STUDENT=results/budget/student_keep_score_300each_chat_rank_20260814/checkpoint-best
DELTA_STUDENT=results/budget/student_delta_300each_chat_rank_20260814/checkpoint-best
OUTPUT_ROOT="${OUTPUT_ROOT:-results/budget/longbench_en16_attention_chat_delta_chat_drift_once_t1_20260814}"
LOG_ROOT=results/budget/attention_template_ablation_drift_once_t1_20260814/run_logs/chat_resume
TASK_FILTER="${TASK_FILTER:-all}"

# The standard 16-dataset English LongBench evaluation suite. Chinese tasks and
# the separate LongBench-E configs are intentionally excluded. Fields are task
# name, official generation length, and the exact Parquet row count.
TASK_SPECS=(
  "longbench_qasper:128:200"
  "longbench_narrativeqa:128:200"
  "longbench_multifieldqa_en:64:150"
  "longbench_hotpotqa:32:200"
  "longbench_2wikimqa:32:200"
  "longbench_musique:32:200"
  "longbench_gov_report:512:200"
  "longbench_qmsum:512:200"
  "longbench_multi_news:512:200"
  "longbench_trec:64:200"
  "longbench_triviaqa:32:200"
  "longbench_samsum:128:200"
  "longbench_passage_count:32:200"
  "longbench_passage_retrieval_en:32:200"
  "longbench_lcc:64:500"
  "longbench_repobench-p:64:500"
)

ACTIVE_PID=""
FAILURES=()

report_gpu_processes() {
  echo "[gpu-check] physical GPU ${PHYSICAL_GPU} compute processes:"
  nvidia-smi -i "${PHYSICAL_GPU}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader || true
}

cleanup() {
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
  report_gpu_processes
}

on_signal() {
  cleanup
  exit 130
}

trap on_signal INT TERM
trap cleanup EXIT

for required in \
  "${KEEP_STUDENT}/pytorch_model.bin" \
  "${DELTA_STUDENT}/pytorch_model.bin"; do
  if [[ ! -f "${required}" ]]; then
    echo "[error] missing checkpoint file: ${required}" >&2
    exit 1
  fi
done

nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
GPU_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${GPU_PIDS}" ]]; then
  echo "[error] physical GPU ${PHYSICAL_GPU} is busy; refusing to launch (PIDs=${GPU_PIDS})" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

for spec in "${TASK_SPECS[@]}"; do
  IFS=: read -r task gen_length expected_rows <<<"${spec}"
  short="${task#longbench_}"
  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${task}" && "${TASK_FILTER}" != "${short}" ]]; then
    continue
  fi

  task_output="${OUTPUT_ROOT}/${short}"
  task_log="${LOG_ROOT}/${short}.log"
  completed="$(find "${task_output}" -type f -name 'results_*.json' -print -quit 2>/dev/null || true)"
  if [[ -n "${completed}" ]]; then
    echo "[skip] ${task}: completed result=${completed}"
    continue
  fi

  echo "[eval] ${task}: all ${expected_rows} rows, gen=${gen_length}, keep=960, fixed refresh=480, interval=1"
  mkdir -p "${task_output}"
  "${PYTHON}" evaluation_script.py run \
    --model LLaDA \
    --tasks "${task}" \
    --include_path "${TASK_ROOT}" \
    --batch_size 1 \
    --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${KEEP_STUDENT},student_refresh_path=${DELTA_STUDENT},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=960,student_refresh_tokens=480,student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
    --gen_kwargs "block_length=32,gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --num_fewshot 0 \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code \
    --output_path "${task_output}" >"${task_log}" 2>&1 &
  ACTIVE_PID=$!
  wait "${ACTIVE_PID}"
  status=$?
  ACTIVE_PID=""
  if [[ "${status}" -ne 0 ]]; then
    echo "[failed] ${task}: exit=${status}, log=${task_log}" >&2
    FAILURES+=("${task}")
  else
    echo "[done] ${task}: output=${task_output}"
  fi
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "[incomplete] failed tasks: ${FAILURES[*]}" >&2
  exit 1
fi

echo "[done] all 16 standard English LongBench datasets completed: ${OUTPUT_ROOT}"
