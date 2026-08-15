#!/usr/bin/env bash
set -euo pipefail

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
TEACHER=results/budget/offline_hybrid_teacher_300each_chat_20260811
KEEP_OUT=results/budget/student_keep_score_300each_chat_rank_20260814
DELTA_OUT=results/budget/student_delta_300each_chat_rank_20260814
KEEP_STUDENT="${KEEP_OUT}/checkpoint-best"
DELTA_STUDENT="${DELTA_OUT}/checkpoint-best"
LIMIT="${LIMIT:-200}"
EVAL_OUT="${EVAL_OUT:-results/budget/samsum_b960_refresh480_attention_rank_delta_rank_chat_drift_once_t1_20260815/limit${LIMIT}}"
RUN_LOG_DIR="${KEEP_OUT}/run_logs"
KEEP_TRAIN_LOG="${RUN_LOG_DIR}/train_keep.log"
DELTA_TRAIN_LOG="${RUN_LOG_DIR}/train_delta.log"
EVAL_LOG="${RUN_LOG_DIR}/eval_samsum_drift_once_t1_limit${LIMIT}.log"

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

ACTIVE_PID=""

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

on_signal() {
  stop_child
  report_gpu_processes
  exit 130
}

on_exit() {
  stop_child
  report_gpu_processes
}

run_gpu() {
  "$@" &
  ACTIVE_PID=$!
  set +e
  wait "${ACTIVE_PID}"
  local status=$?
  set -e
  ACTIVE_PID=""
  return "${status}"
}

trap on_signal INT TERM
trap on_exit EXIT

nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
GPU_PIDS="$(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${GPU_PIDS}" ]]; then
  echo "[error] physical GPU ${PHYSICAL_GPU} is busy; refusing to launch (PIDs=${GPU_PIDS})" >&2
  exit 1
fi

mkdir -p "${RUN_LOG_DIR}"

if [[ ! -f "${KEEP_STUDENT}/pytorch_model.bin" ]]; then
  echo "[train] chat-template attention keep student, continuous score + ranking only"
  run_gpu "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER}" \
    --output-dir "${KEEP_OUT}" \
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
    --mlp-dim 512 >"${KEEP_TRAIN_LOG}" 2>&1
else
  echo "[train] attention keep checkpoint already exists; skipping"
fi

if [[ ! -f "${KEEP_STUDENT}/pytorch_model.bin" ]]; then
  echo "[error] attention keep training finished without checkpoint-best" >&2
  exit 1
fi

if [[ ! -f "${DELTA_STUDENT}/pytorch_model.bin" ]]; then
  echo "[train] cumulative chat-template delta student, continuous score + ranking only"
  run_gpu "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER}" \
    --output-dir "${DELTA_OUT}" \
    --model "${MODEL}" \
    --datasets "${DATASETS[@]}" \
    --val-ratio 0.1 \
    --epochs 10 \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode delta \
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
    --mlp-dim 512 >"${DELTA_TRAIN_LOG}" 2>&1
else
  echo "[train] cumulative delta checkpoint already exists; skipping"
fi

if [[ ! -f "${DELTA_STUDENT}/pytorch_model.bin" ]]; then
  echo "[error] training finished without checkpoint-best" >&2
  exit 1
fi

echo "[eval] SAMSum: attention top-960, fixed delta top-480, refresh every step"
run_gpu "${PYTHON}" evaluation_script.py run \
  --model LLaDA \
  --tasks longbench_samsum \
  --include_path "${TASK_ROOT}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${KEEP_STUDENT},student_refresh_path=${DELTA_STUDENT},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=960,student_refresh_tokens=480,student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code \
  --output_path "${EVAL_OUT}" >"${EVAL_LOG}" 2>&1

echo "[done] attention keep checkpoint=${KEEP_STUDENT}"
echo "[done] cumulative delta checkpoint=${DELTA_STUDENT}"
echo "[done] evaluation output=${EVAL_OUT}"
