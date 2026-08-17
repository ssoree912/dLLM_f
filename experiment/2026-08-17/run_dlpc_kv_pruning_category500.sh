#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ACTION="${1:-plan}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT}/.venv/bin/python"
elif [[ -x "/home/M2026107/dllm/dLLM-Cache/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="/home/M2026107/dllm/dLLM-Cache/.venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi

PYTHON="${DLPC_PYTHON:-${DEFAULT_PYTHON}}"
MODEL="${MODEL:-/home/M2026107/dllm/model/LLaDA-8B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASKKV_ENABLED="${MASKKV_ENABLED:-0}"

RUN_TAG="${RUN_TAG:-20260817}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-500}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-2e-5}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-0}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
QUESTION_WINDOW="${QUESTION_WINDOW:-128}"
TEACHER_GEN_LENGTH="${TEACHER_GEN_LENGTH:-128}"
TEACHER_BLOCK_LENGTH="${TEACHER_BLOCK_LENGTH:-8}"
TEACHER_STEPS="${TEACHER_STEPS:-128}"
ACTIVE_TOP_K="${ACTIVE_TOP_K:-0}"

KEEP_BUDGET="${KEEP_BUDGET:-960}"
REFRESH_TOKENS="${REFRESH_TOKENS:-480}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-1}"
FROZEN_LAYERS="${FROZEN_LAYERS:-0}"

TASK_ROOT="${TASK_ROOT:-experiment/345/2026-08-13/tasks/longbench_full_local}"
LOG_DIR="${LOG_DIR:-logs/dlpc_kv_pruning_category500_${RUN_TAG}}"
CHAT_TEACHER_ROOT="${CHAT_TEACHER_ROOT:-/home/M2026107/.cache/dlpc_kv_pruning_teacher500_delta_chat_${RUN_TAG}}"
ATTN_TEACHER_ROOT="${ATTN_TEACHER_ROOT:-${CHAT_TEACHER_ROOT}}"
DELTA_TEACHER_ROOT="${DELTA_TEACHER_ROOT:-${CHAT_TEACHER_ROOT}}"
STUDENT_ROOT="${STUDENT_ROOT:-results/budget/dlpc_kv_pruning_category500_${RUN_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-results/dlpc_kv_pruning_category500_b${KEEP_BUDGET}_r${REFRESH_TOKENS}_${RUN_TAG}}"

CATEGORIES=(
  single_doc_qa
  multi_doc_qa
  summarization
  few_shot
  synthetic
  code
)

declare -A CATEGORY_TRAIN_DATASETS=(
  [single_doc_qa]="qasper narrativeqa"
  [multi_doc_qa]="2wikimultihopqa_train hotpotqa musique"
  [summarization]="gov_report multi_news qmsum"
  [few_shot]="samsum trec triviaqa"
  [synthetic]=""
  [code]="repobench-p"
)

declare -A CATEGORY_EVAL_TASKS=(
  [single_doc_qa]="qasper narrativeqa multifieldqa_en"
  [multi_doc_qa]="2wikimqa hotpotqa musique"
  [summarization]="gov_report multi_news qmsum"
  [few_shot]="trec triviaqa samsum"
  [synthetic]="passage_count passage_retrieval_en"
  [code]="lcc repobench-p"
)

data_path_for_dataset() {
  case "$1" in
    2wikimultihopqa_train)
      echo "/home/M2026107/dllm/data/train/2wikimultihopqa/2wikimultihopqa_train_longbench_format.jsonl"
      ;;
    hotpotqa)
      echo "/home/M2026107/dllm/data/train/musique/hotpotqa/hotpotqa_train_longbench_format.jsonl"
      ;;
    *)
      echo "/home/M2026107/dllm/data/train/$1/$1_train_longbench_format.jsonl"
      ;;
  esac
}

gen_length_for_task() {
  case "$1" in
    hotpotqa|2wikimqa|musique|triviaqa|passage_count|passage_retrieval_en)
      echo 32
      ;;
    multifieldqa_en|trec|lcc|repobench-p)
      echo 64
      ;;
    qasper|narrativeqa|samsum)
      echo 128
      ;;
    gov_report|qmsum|multi_news)
      echo 512
      ;;
    *)
      echo "[error] unknown eval task: $1" >&2
      exit 2
      ;;
  esac
}

eval_harness_task_for_task() {
  local task="$1"
  local yaml_path="${TASK_ROOT}/${task}.yaml"
  if [[ -f "${yaml_path}" ]]; then
    local task_name
    task_name="$(awk -F': *' '$1 == "task" {print $2; exit}' "${yaml_path}")"
    if [[ -n "${task_name}" ]]; then
      echo "${task_name}"
      return 0
    fi
  fi
  echo "${TASK_PREFIX:-longbench}_${task}"
}

teacher_files_for_category() {
  local root="$1"
  local category="$2"
  local count=0
  for dataset in ${CATEGORY_TRAIN_DATASETS[${category}]}; do
    if [[ -d "${root}/${dataset}" ]]; then
      count=$((count + $(find "${root}/${dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)))
    fi
  done
  echo "${count}"
}

attention_student_dir() {
  echo "${STUDENT_ROOT}/$1/attention_score_rank0p1_topk0_e${EPOCHS}"
}

delta_student_dir() {
  echo "${STUDENT_ROOT}/$1/delta_score_rank0p1_topk0_e${EPOCHS}"
}

print_plan() {
  echo "[plan] branch=$(git branch --show-current)"
  echo "[plan] python=${PYTHON}"
  echo "[plan] model=${MODEL}"
  echo "[plan] device=${DEVICE}, visible_gpu=${CUDA_VISIBLE_DEVICES}"
  echo "[plan] samples_per_dataset=${SAMPLES_PER_DATASET}, epochs=${EPOCHS}"
  echo "[plan] shared chat teacher root=${CHAT_TEACHER_ROOT}"
  echo "[plan] attention teacher root=${ATTN_TEACHER_ROOT} (chat template, teacher_norm)"
  echo "[plan] delta teacher root=${DELTA_TEACHER_ROOT} (chat template, delta_norm)"
  echo "[plan] student root=${STUDENT_ROOT}"
  echo "[plan] eval root=${EVAL_ROOT}"
  for category in "${CATEGORIES[@]}"; do
    echo "[plan] ${category}: train=[${CATEGORY_TRAIN_DATASETS[${category}]}] eval=[${CATEGORY_EVAL_TASKS[${category}]}]"
  done
}

extract_one_dataset() {
  local root="$1"
  local dataset="$2"
  local chat_flag="$3"
  local data_path
  data_path="$(data_path_for_dataset "${dataset}")"
  if [[ ! -f "${data_path}" ]]; then
    echo "[skip] missing train data for dataset=${dataset}: ${data_path}"
    return 0
  fi
  local existing_count=0
  if [[ -d "${root}/${dataset}" ]]; then
    existing_count="$(find "${root}/${dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  fi
  if (( existing_count >= SAMPLES_PER_DATASET )); then
    echo "[skip] dataset=${dataset} root=${root}: existing=${existing_count}/${SAMPLES_PER_DATASET}"
    return 0
  fi
  mkdir -p "${LOG_DIR}"
  local log_suffix="nochat"
  local extra_args=()
  if [[ "${chat_flag}" == "chat" ]]; then
    log_suffix="chat"
    extra_args+=(--apply-chat-template)
  fi
  echo "[extract] dataset=${dataset} root=${root} template=${log_suffix}"
  "${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_teacher \
    --model "${MODEL}" \
    --data "${data_path}" \
    --output-root "${root}" \
    --datasets "${dataset}" \
    --samples-per-dataset "${SAMPLES_PER_DATASET}" \
    --max-length "${MAX_LENGTH}" \
    --question-window "${QUESTION_WINDOW}" \
    --gen-length "${TEACHER_GEN_LENGTH}" \
    --block-length "${TEACHER_BLOCK_LENGTH}" \
    --steps "${TEACHER_STEPS}" \
    --active-top-k "${ACTIVE_TOP_K}" \
    --temperature 0 \
    --confidence-weight \
    --target-aggregation max \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --prompt-format train \
    "${extra_args[@]}" \
    >"${LOG_DIR}/extract_${dataset}_${log_suffix}.log" 2>&1
}

extract_teachers() {
  for category in "${CATEGORIES[@]}"; do
    for dataset in ${CATEGORY_TRAIN_DATASETS[${category}]}; do
      if [[ "${ATTN_TEACHER_ROOT}" == "${DELTA_TEACHER_ROOT}" ]]; then
        extract_one_dataset "${CHAT_TEACHER_ROOT}" "${dataset}" chat
      else
        extract_one_dataset "${ATTN_TEACHER_ROOT}" "${dataset}" chat
        extract_one_dataset "${DELTA_TEACHER_ROOT}" "${dataset}" chat
      fi
    done
  done
}

train_one_student() {
  local category="$1"
  local target_mode="$2"
  local teacher_root="$3"
  local output_dir="$4"
  shift 4
  local datasets=("$@")
  if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "[skip] category=${category} target=${target_mode}: no train datasets"
    return 0
  fi
  local file_count
  file_count="$(teacher_files_for_category "${teacher_root}" "${category}")"
  if [[ "${file_count}" == "0" ]]; then
    echo "[skip] category=${category} target=${target_mode}: no teacher shards under ${teacher_root}"
    return 0
  fi
  if [[ -f "${output_dir}/checkpoint-best/pytorch_model.bin" ]]; then
    echo "[train] category=${category} target=${target_mode}: checkpoint exists; skip ${output_dir}"
    return 0
  fi
  mkdir -p "${LOG_DIR}" "${output_dir}"
  echo "[train] category=${category} target=${target_mode} files=${file_count} output=${output_dir}"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${teacher_root}" \
    --output-dir "${output_dir}" \
    --model "${MODEL}" \
    --datasets "${datasets[@]}" \
    --val-ratio 0.1 \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay 0 \
    --target-mode "${target_mode}" \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0 \
    --max-grad-norm 1 \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --seed "${SEED}" \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 \
    >"${LOG_DIR}/train_${category}_${target_mode}.log" 2>&1
}

train_students() {
  for category in "${CATEGORIES[@]}"; do
    local dataset_line="${CATEGORY_TRAIN_DATASETS[${category}]}"
    local datasets=()
    if [[ -n "${dataset_line}" ]]; then
      read -r -a datasets <<<"${dataset_line}"
    fi
    train_one_student "${category}" score "${ATTN_TEACHER_ROOT}" "$(attention_student_dir "${category}")" "${datasets[@]}"
    train_one_student "${category}" delta "${DELTA_TEACHER_ROOT}" "$(delta_student_dir "${category}")" "${datasets[@]}"
  done
}

eval_one_task() {
  local category="$1"
  local task="$2"
  local attention_student
  local delta_student
  attention_student="$(attention_student_dir "${category}")/checkpoint-best"
  delta_student="$(delta_student_dir "${category}")/checkpoint-best"
  if [[ ! -f "${attention_student}/pytorch_model.bin" || ! -f "${delta_student}/pytorch_model.bin" ]]; then
    echo "[skip] eval category=${category} task=${task}: missing trained scorer"
    return 0
  fi
  local gen_length
  local harness_task
  gen_length="$(gen_length_for_task "${task}")"
  harness_task="$(eval_harness_task_for_task "${task}")"
  local out_dir="${EVAL_ROOT}/${category}/${task}"
  local eval_args=()
  if [[ "${LIMIT}" != "0" ]]; then
    eval_args+=(--limit "${LIMIT}")
  fi
  if [[ "${task}" != "trec" ]]; then
    eval_args+=(--apply_chat_template --fewshot_as_multiturn)
  fi
  mkdir -p "${LOG_DIR}" "${out_dir}"
  echo "[eval] category=${category} task=${task} harness_task=${harness_task} gen=${gen_length} output=${out_dir}"
  "${PYTHON}" evaluation_script.py run \
    --model LLaDA \
    --tasks "${harness_task}" \
    --include_path "${TASK_ROOT}" \
    --batch_size 1 \
    --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=${MAX_LENGTH},student_path=${attention_student},student_refresh_path=${delta_student},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=${KEEP_BUDGET},student_refresh_tokens=${REFRESH_TOKENS},student_refresh_interval=${REFRESH_INTERVAL},student_drift_frozen_layers=${FROZEN_LAYERS},student_question_window=${QUESTION_WINDOW},student_score_activation=softmax" \
    --gen_kwargs "block_length=32,gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --num_fewshot 0 \
    --log_samples \
    --trust_remote_code \
    --output_path "${out_dir}" \
    "${eval_args[@]}" \
    >"${LOG_DIR}/eval_${category}_${task}.log" 2>&1
}

eval_categories() {
  for category in "${CATEGORIES[@]}"; do
    for task in ${CATEGORY_EVAL_TASKS[${category}]}; do
      eval_one_task "${category}" "${task}"
    done
  done
}

summarize_results() {
  "${PYTHON}" experiment/2026-08-17/summarize_dlpc_kv_pruning_results.py "${EVAL_ROOT}"
}

case "${ACTION}" in
  plan)
    print_plan
    ;;
  extract)
    print_plan
    extract_teachers
    ;;
  train)
    print_plan
    train_students
    ;;
  eval)
    print_plan
    eval_categories
    ;;
  summarize)
    summarize_results
    ;;
  all)
    print_plan
    extract_teachers
    train_students
    eval_categories
    summarize_results
    ;;
  *)
    echo "usage: $0 {plan|extract|train|eval|summarize|all}" >&2
    exit 2
    ;;
esac
