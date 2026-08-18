#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Workspace policy: expose physical GPU 2 only; it becomes cuda:0 in this process.
PHYSICAL_GPU=2
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/opt/conda/envs/dllm/bin/python
MODEL=/workspace/dllm/model/LLaDA-8B-Instruct
SOURCE_ROOT=results/budget/prompt_source_300each_11dataset_20260818
OUTPUT_ROOT=results/budget/offline_commit_attention_delta_teacher_300each_chat_notopk_20260818
LOG_ROOT=results/budget/teacher_run_logs
LOG_FILE="${LOG_ROOT}/commit_attention_delta_300each_chat_notopk_20260818.log"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
MAX_STALLED_RETRIES="${MAX_STALLED_RETRIES:-3}"

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

for dataset in "${DATASETS[@]}"; do
  dataset_root="${SOURCE_ROOT}/${dataset}"
  if [[ ! -d "${dataset_root}" ]]; then
    echo "[error] missing prompt-source dataset: ${dataset_root}" >&2
    exit 1
  fi
  count="$(find "${dataset_root}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  if [[ "${count}" -ne 300 ]]; then
    echo "[error] ${dataset} prompt-source count=${count}, expected=300" >&2
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

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

echo "[extract] commit-time continuous Attention + cumulative Delta in one trajectory"
echo "[extract] chat_template=true active_top_k=0 aggregation=max samples=11x300"
echo "[extract] chunk_size=${CHUNK_SIZE} timeout=${CHUNK_TIMEOUT_SECONDS}s resume=true"

run_chunk() {
  local dataset=$1
  local target=$2
  timeout --signal=TERM --kill-after=30s "${CHUNK_TIMEOUT_SECONDS}s" \
    "${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_from_shards \
      --model "${MODEL}" \
      --source-root "${SOURCE_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --datasets "${dataset}" \
      --n-samples "${target}" \
      --device cuda:0 \
      --dtype bfloat16 \
      --gen-length 128 \
      --block-length 8 \
      --steps 128 \
      --active-top-k 0 \
      --confidence-weight \
      --target-aggregation max \
      --apply-chat-template >>"${LOG_FILE}" 2>&1 &
  ACTIVE_PID=$!
  set +e
  wait "${ACTIVE_PID}"
  local status=$?
  set -e
  ACTIVE_PID=""
  return "${status}"
}

for dataset in "${DATASETS[@]}"; do
  output_dataset="${OUTPUT_ROOT}/${dataset}"
  mkdir -p "${output_dataset}"
  stalled_retries=0
  while true; do
    before="$(find "${output_dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
    if [[ "${before}" -ge 300 ]]; then
      echo "[dataset-done] ${dataset}=300/300" | tee -a "${LOG_FILE}"
      break
    fi
    target=$((before + CHUNK_SIZE))
    if [[ "${target}" -gt 300 ]]; then
      target=300
    fi
    echo "[chunk] dataset=${dataset} before=${before} target=${target}" | tee -a "${LOG_FILE}"
    if run_chunk "${dataset}" "${target}"; then
      status=0
    else
      status=$?
      echo "[chunk-exit] dataset=${dataset} status=${status}" | tee -a "${LOG_FILE}"
    fi
    after="$(find "${output_dataset}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
    if [[ "${after}" -gt "${before}" ]]; then
      stalled_retries=0
      echo "[chunk-progress] dataset=${dataset} ${before}->${after}" | tee -a "${LOG_FILE}"
      continue
    fi
    stalled_retries=$((stalled_retries + 1))
    echo "[chunk-stalled] dataset=${dataset} retry=${stalled_retries}/${MAX_STALLED_RETRIES}" \
      | tee -a "${LOG_FILE}"
    if [[ "${stalled_retries}" -ge "${MAX_STALLED_RETRIES}" ]]; then
      echo "[error] ${dataset} made no progress after ${stalled_retries} attempts" >&2
      exit 1
    fi
  done
done

saved="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.pt' | wc -l)"
if [[ "${saved}" -ne 3300 ]]; then
  echo "[error] extraction ended with ${saved}/3300 shards; log=${LOG_FILE}" >&2
  exit 1
fi

echo "[done] teacher shards=${saved} output=${OUTPUT_ROOT}"
echo "[done] log=${LOG_FILE}"
