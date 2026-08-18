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
OUTPUT_ROOT=results/budget/offline_lifetime_attention_delta_teacher_300each_chat_notopk_20260818
LOG_ROOT=results/budget/teacher_run_logs
LOG_FILE="${LOG_ROOT}/lifetime_attention_delta_300each_chat_notopk_20260818.log"

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

echo "[extract] lifetime Attention + cumulative Delta in one trajectory"
echo "[extract] chat_template=true active_top_k=0 aggregation=sum samples=11x300"
"${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_from_shards \
  --model "${MODEL}" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --n-samples 300 \
  --device cuda:0 \
  --dtype bfloat16 \
  --gen-length 128 \
  --block-length 8 \
  --steps 128 \
  --active-top-k 0 \
  --reference-query-mode lifetime_mask \
  --confidence-weight \
  --target-aggregation sum \
  --apply-chat-template >"${LOG_FILE}" 2>&1 &
ACTIVE_PID=$!
wait "${ACTIVE_PID}"
ACTIVE_PID=""

saved="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.pt' | wc -l)"
if [[ "${saved}" -ne 3300 ]]; then
  echo "[error] extraction ended with ${saved}/3300 shards; log=${LOG_FILE}" >&2
  exit 1
fi

echo "[done] teacher shards=${saved} output=${OUTPUT_ROOT}"
echo "[done] log=${LOG_FILE}"
