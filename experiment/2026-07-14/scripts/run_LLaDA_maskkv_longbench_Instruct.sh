#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiment/2026-07-14"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/model/LLaDA-8B-Instruct}"
LOCAL_TASK_PATH="${LOCAL_TASK_PATH:-${EXPERIMENT_ROOT}/tasks/longbench_local}"
DATASETS_CACHE_DIR="${DATASETS_CACHE_DIR:-${TMPDIR:-/tmp}/dllm_hf_datasets_cache}"
MODULES_CACHE_DIR="${MODULES_CACHE_DIR:-${TMPDIR:-/tmp}/dllm_hf_modules_cache}"
mkdir -p "${DATASETS_CACHE_DIR}" "${MODULES_CACHE_DIR}"

export HF_DATASETS_CACHE="${DATASETS_CACHE_DIR}"
export HF_MODULES_CACHE="${MODULES_CACHE_DIR}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASKKV_ENABLED="${MASKKV_ENABLED:-1}"
export MASKKV_BUDGET="${MASKKV_BUDGET:-256}"
export MASKKV_LAYER_BASE_RATE="${MASKKV_LAYER_BASE_RATE:-1.0}"
export MASKKV_HEAD_BASE_RATE="${MASKKV_HEAD_BASE_RATE:-0.2}"

MODEL="${MODEL:-${MODEL_DIR}}"
OUTPUT_PATH="${OUTPUT_PATH:-${EXPERIMENT_ROOT}/results/longbench_maskkv_log}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/accelerate_config_single_gpu.yaml}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${REPO_ROOT}/.venv/bin/accelerate}"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  ACCELERATE_BIN="accelerate"
fi
PROMPT_INTERVAL_STEPS="${PROMPT_INTERVAL_STEPS:-50}"
GEN_INTERVAL_STEPS="${GEN_INTERVAL_STEPS:-5}"
TRANSFER_RATIO="${TRANSFER_RATIO:-0.25}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
TASK_FILTER="${TASK_FILTER:-all}"
COMMON_MODEL_ARGS="pretrained=${MODEL},prompt_interval_steps=${PROMPT_INTERVAL_STEPS},gen_interval_steps=${GEN_INTERVAL_STEPS},cfg_interval_steps=1,transfer_ratio=${TRANSFER_RATIO},is_feature_cache=True,is_cfg_cache=False"
if [[ -n "${MODEL_MAX_LENGTH:-}" ]]; then
  COMMON_MODEL_ARGS="${COMMON_MODEL_ARGS},max_length=${MODEL_MAX_LENGTH}"
fi
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

should_run_task() {
  local task_name="$1"
  local short_name="${task_name#local_longbench_}"
  [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "${task_name}" || "${TASK_FILTER}" == "${short_name}" ]]
}

run_task() {
  local task_name="$1"
  local gen_length="$2"
  local cmd=(
    "${ACCELERATE_BIN}" launch --config_file "${ACCELERATE_CONFIG}" evaluation_script.py run
    --model LLaDA \
    --tasks "${task_name}" \
    --include_path "${LOCAL_TASK_PATH}" \
    --batch_size 1 \
    --model_args "${COMMON_MODEL_ARGS}" \
    --gen_kwargs "block_length=${BLOCK_LENGTH},gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    "${LIMIT_ARGS[@]}" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi
  cd "${REPO_ROOT}"
  "${cmd[@]}"
}

if should_run_task local_longbench_qasper; then
  run_task local_longbench_qasper 128
fi
if should_run_task local_longbench_2wikimqa; then
  run_task local_longbench_2wikimqa 32
fi
if should_run_task local_longbench_qmsum; then
  run_task local_longbench_qmsum 512
fi
