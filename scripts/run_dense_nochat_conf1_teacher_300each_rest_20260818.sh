#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/M2026107/dllm/dLLM-Cache/.venv/bin/python"
MODEL="${MODEL:-/home/M2026107/dllm/model/LLaDA-8B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/M2026107/dllm/dLLM-Cache/results/budget/future_pool_teacher_train_300each_g128_top128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/M2026107/.cache/offline_hybrid_teacher_300each_dense_nochat_conf1_20260818}"
LOG_ROOT="${LOG_ROOT:-logs/dense_nochat_conf1_teacher_300each_20260818}"

DATASETS=(
  2wikimultihopqa_train
  gov_report
  hotpotqa
  multi_news
  musique
  narrativeqa
  qasper
  qmsum
  trec
  triviaqa
)

mkdir -p "${LOG_ROOT}"

echo "[config] root=${ROOT}"
echo "[config] source_root=${SOURCE_ROOT}"
echo "[config] output_root=${OUTPUT_ROOT}"
echo "[config] datasets=${DATASETS[*]}"
echo "[config] n_samples=300 active_top_k=0 confidence_weight=1 apply_chat_template=0"

for dataset in "${DATASETS[@]}"; do
  if [[ ! -d "${SOURCE_ROOT}/${dataset}" ]]; then
    echo "[error] missing source shards: ${SOURCE_ROOT}/${dataset}" >&2
    exit 1
  fi
done

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
  --confidence-weight \
  --target-aggregation max \
  >"${LOG_ROOT}/extract_teacher_rest.log" 2>&1

echo "[done] output_root=${OUTPUT_ROOT}"
for dataset in "${DATASETS[@]}"; do
  count="$(find "${OUTPUT_ROOT}/${dataset}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
  echo "[done] ${dataset} shards=${count}"
done
