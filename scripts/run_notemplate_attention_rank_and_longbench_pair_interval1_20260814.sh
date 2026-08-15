#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Workspace policy: physical GPU 2 only; it is cuda:0 inside this process.
PHYSICAL_GPU=2
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/opt/conda/envs/dllm/bin/python
MODEL=/workspace/dllm/model/LLaDA-8B-Instruct
TASK_ROOT=experiment/345/2026-08-13/tasks/longbench_full_local

# Complete 11 x 300 no-chat-template attention teacher. The similarly named
# offline_hybrid no-template directory is incomplete (323 shards), so it is not used.
NO_TEMPLATE_TEACHER=results/budget/future_pool_teacher_train_300each_g128_top128
NO_TEMPLATE_KEEP_OUT=results/budget/student_keep_score_300each_notemplate_rank_20260814
NO_TEMPLATE_KEEP="${NO_TEMPLATE_KEEP_OUT}/checkpoint-best"
CHAT_KEEP=results/budget/student_keep_score_300each_chat_rank_20260814/checkpoint-best
CHAT_DELTA=results/budget/student_delta_300each_chat_rank_20260814/checkpoint-best

CHAT_OUTPUT_ROOT=results/budget/longbench_en16_attention_chat_delta_chat_drift_once_t1_20260814
NO_TEMPLATE_OUTPUT_ROOT=results/budget/longbench_en16_attention_notemplate_delta_chat_drift_once_t1_20260814
RUN_LOG_ROOT=results/budget/attention_template_ablation_drift_once_t1_20260814/run_logs
TRAIN_LOG="${RUN_LOG_ROOT}/train_attention_notemplate_rank.log"
TASK_FILTER="${TASK_FILTER:-all}"

DATASETS=(
  2wikimultihopqa_train
  gov_report
  hotpotqa
  multi_news
  musique
  narrativeqa
  qasper
  qmsum
  samsum
  trec
  triviaqa
)

# SAMSum is first so the requested same-task comparison is available before the
# remaining full LongBench sweep. Fields: task, official gen length, row count.
TASK_SPECS=(
  "longbench_samsum:128:200"
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

stop_child() {
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
}

cleanup() {
  stop_child
  report_gpu_processes
}

on_signal() {
  cleanup
  exit 130
}

run_gpu() {
  "$@" &
  ACTIVE_PID=$!
  wait "${ACTIVE_PID}"
  local status=$?
  ACTIVE_PID=""
  return "${status}"
}

trap on_signal INT TERM
trap cleanup EXIT

for required in \
  "${CHAT_KEEP}/pytorch_model.bin" \
  "${CHAT_DELTA}/pytorch_model.bin"; do
  if [[ ! -f "${required}" ]]; then
    echo "[error] missing checkpoint file: ${required}" >&2
    exit 1
  fi
done

nvidia-smi -i "${PHYSICAL_GPU}" \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
GPU_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${GPU_PIDS}" ]]; then
  echo "[error] physical GPU ${PHYSICAL_GPU} is busy; refusing to launch (PIDs=${GPU_PIDS})" >&2
  exit 1
fi

mkdir -p "${RUN_LOG_ROOT}" "${CHAT_OUTPUT_ROOT}" "${NO_TEMPLATE_OUTPUT_ROOT}"

if [[ ! -f "${NO_TEMPLATE_KEEP}/pytorch_model.bin" ]]; then
  echo "[train] no-template attention student: continuous score + ranking, top-k loss disabled"
  run_gpu "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${NO_TEMPLATE_TEACHER}" \
    --output-dir "${NO_TEMPLATE_KEEP_OUT}" \
    --model "${MODEL}" \
    --datasets "${DATASETS[@]}" \
    --val-ratio 0.1 \
    --epochs 10 \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode score \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0 \
    --max-grad-norm 1 \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 >"${TRAIN_LOG}" 2>&1
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] no-template attention training failed: exit=${status}, log=${TRAIN_LOG}" >&2
    exit "${status}"
  fi
else
  echo "[skip] no-template attention checkpoint already exists: ${NO_TEMPLATE_KEEP}"
fi

if [[ ! -f "${NO_TEMPLATE_KEEP}/pytorch_model.bin" ]]; then
  echo "[error] training finished without checkpoint-best: ${NO_TEMPLATE_KEEP}" >&2
  exit 1
fi

evaluate_one() {
  local variant=$1
  local keep_student=$2
  local output_root=$3
  local task=$4
  local gen_length=$5
  local expected_rows=$6
  local short="${task#longbench_}"
  local task_output="${output_root}/${short}"
  local task_log="${RUN_LOG_ROOT}/${variant}_${short}.log"
  local completed

  completed="$(find "${task_output}" -type f -name 'results_*.json' -print -quit 2>/dev/null || true)"
  if [[ -n "${completed}" ]]; then
    echo "[skip] ${variant} ${task}: completed result=${completed}"
    return 0
  fi

  echo "[eval] ${variant} ${task}: rows=${expected_rows}, gen=${gen_length}, select_once=true, interval=1"
  mkdir -p "${task_output}"
  run_gpu "${PYTHON}" evaluation_script.py run \
    --model LLaDA \
    --tasks "${task}" \
    --include_path "${TASK_ROOT}" \
    --batch_size 1 \
    --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${keep_student},student_refresh_path=${CHAT_DELTA},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=960,student_refresh_tokens=480,student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
    --gen_kwargs "block_length=32,gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --num_fewshot 0 \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code \
    --output_path "${task_output}" >"${task_log}" 2>&1
}

for spec in "${TASK_SPECS[@]}"; do
  IFS=: read -r task gen_length expected_rows <<<"${spec}"
  short="${task#longbench_}"
  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${task}" && "${TASK_FILTER}" != "${short}" ]]; then
    continue
  fi

  if evaluate_one chat_attention "${CHAT_KEEP}" "${CHAT_OUTPUT_ROOT}" \
    "${task}" "${gen_length}" "${expected_rows}"; then
    :
  else
    status=$?
    echo "[failed] chat_attention ${task}: exit=${status}" >&2
    FAILURES+=("chat_attention:${task}")
  fi

  if evaluate_one notemplate_attention "${NO_TEMPLATE_KEEP}" "${NO_TEMPLATE_OUTPUT_ROOT}" \
    "${task}" "${gen_length}" "${expected_rows}"; then
    :
  else
    status=$?
    echo "[failed] notemplate_attention ${task}: exit=${status}" >&2
    FAILURES+=("notemplate_attention:${task}")
  fi
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "[incomplete] failed evaluations: ${FAILURES[*]}" >&2
  exit 1
fi

echo "[done] chat attention results: ${CHAT_OUTPUT_ROOT}"
echo "[done] no-template attention results: ${NO_TEMPLATE_OUTPUT_ROOT}"
