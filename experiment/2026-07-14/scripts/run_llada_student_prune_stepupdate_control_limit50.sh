#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiment/2026-07-14"
MODEL="${MODEL:-$(cd "${REPO_ROOT}/.." && pwd)/model/LLaDA-8B-Instruct}"
STUDENT="${STUDENT:-${EXPERIMENT_ROOT}/results/online_self_generated_student_balanced_2k_g32_topk128_e10_lr2e-5_tw0.02/checkpoint-best}"
LOCAL_TASK_PATH="${LOCAL_TASK_PATH:-${REPO_ROOT}/experiment/345/2026-07-15/tasks/longbench_local}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/accelerate_config_single_gpu.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
LIMIT="${LIMIT:-50}"
TASK_FILTER="${TASK_FILTER:-all}"
BUDGET="${BUDGET:-128}"
QUESTION_WINDOW="${QUESTION_WINDOW:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"

mkdir -p "${EXPERIMENT_ROOT}/results"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR:-/tmp}/dllm_hf_datasets_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${TMPDIR:-/tmp}/dllm_hf_modules_cache}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASKKV_ENABLED=0

COMMON_MODEL_ARGS="pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=${MODEL_MAX_LENGTH},student_path=${STUDENT},student_prompt_prune=True,student_budget=${BUDGET},student_question_window=${QUESTION_WINDOW}"

should_run_task() {
  local short_name="$1"
  [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "${short_name}" || "${TASK_FILTER}" == "local_longbench_${short_name}" ]]
}

run_task() {
  local short_name="$1"
  local gen_length="$2"
  local task_name="local_longbench_${short_name}"
  local output_path="${EXPERIMENT_ROOT}/results/lm_eval_student_prune_queryprune_b${BUDGET}_${short_name}_mlen${MODEL_MAX_LENGTH}_limit${LIMIT}"

  echo "[$(date -Is)] START ${task_name} gen_length=${gen_length} limit=${LIMIT} budget=${BUDGET}"
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch --config_file "${ACCELERATE_CONFIG}" evaluation_script.py run \
    --model LLaDA \
    --tasks "${task_name}" \
    --include_path "${LOCAL_TASK_PATH}" \
    --batch_size 1 \
    --model_args "${COMMON_MODEL_ARGS}" \
    --gen_kwargs "block_length=${BLOCK_LENGTH},gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --limit "${LIMIT}" \
    --num_fewshot 0 \
    --output_path "${output_path}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code
  echo "[$(date -Is)] DONE ${task_name}"
}

if should_run_task lcc; then
  run_task lcc 64
fi
if should_run_task repobench-p; then
  run_task repobench-p 64
fi
if should_run_task triviaqa; then
  run_task triviaqa 32
fi
if should_run_task gov_report; then
  run_task gov_report 512
fi

echo "[$(date -Is)] ALL_DONE"
