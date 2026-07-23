#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiment/345/2026-07-15"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
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
TASKS="${TASKS:-2wikimqa}"
COMMON_MODEL_ARGS="pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=${MODEL_MAX_LENGTH},truncation_strategy=middle"

gen_length_for_task() {
  case "$1" in
    2wikimqa|hotpotqa|musique|passage_count|passage_retrieval_en|triviaqa) echo 32 ;;
    lcc|repobench-p|trec|multifieldqa_en) echo 64 ;;
    narrativeqa|samsum|qasper) echo 128 ;;
    gov_report|multi_news|qmsum) echo 512 ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
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
  local output_path="${EXPERIMENT_ROOT}/results/llada_true_full_middle_${task}_mlen${MODEL_MAX_LENGTH}_${limit_label}"
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
    --apply_chat_template
    --fewshot_as_multiturn
    --trust_remote_code
  )
  printf '[start] true_full_middle task=%s gen_length=%s limit=%s output=%s\n' "${task}" "${gen_length}" "${limit_label}" "${output_path}"
  cd "${REPO_ROOT}"
  "${cmd[@]}"
}

for task in ${TASKS}; do
  run_task "${task}"
done
