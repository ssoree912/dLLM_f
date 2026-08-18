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
TEACHER_ROOT="${TEACHER_ROOT:-/home/M2026107/.cache/offline_hybrid_teacher_samsum300_dense_nochat_conf0_20260818}"
OUT_DIR="${OUT_DIR:-results/budget/student_samsum300_dense_nochat_conf0_attention_delta_2head_rank0p1_topk0_e20_20260818}"
LOG_ROOT="${LOG_ROOT:-logs/samsum300_dense_nochat_conf0_2head_20260818}"

mkdir -p "${LOG_ROOT}"

echo "[config] root=${ROOT}"
echo "[config] source_root=${SOURCE_ROOT}"
echo "[config] teacher_root=${TEACHER_ROOT}"
echo "[config] out_dir=${OUT_DIR}"

if [[ ! -d "${SOURCE_ROOT}/samsum" ]]; then
  echo "[error] missing source samsum shards: ${SOURCE_ROOT}/samsum" >&2
  exit 1
fi

existing_teacher=0
if [[ -d "${TEACHER_ROOT}/samsum" ]]; then
  existing_teacher="$(find "${TEACHER_ROOT}/samsum" -maxdepth 1 -type f -name '*.pt' | wc -l)"
fi

if [[ "${existing_teacher}" -lt 300 ]]; then
  echo "[extract] no-chat dense attention+delta teacher, samsum 300, active_top_k=0, confidence_weight=0"
  "${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_from_shards \
    --model "${MODEL}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${TEACHER_ROOT}" \
    --datasets samsum \
    --n-samples 300 \
    --device cuda:0 \
    --dtype bfloat16 \
    --gen-length 128 \
    --block-length 8 \
    --steps 128 \
    --active-top-k 0 \
    --no-confidence-weight \
    --target-aggregation max \
    >"${LOG_ROOT}/extract_teacher.log" 2>&1
else
  echo "[extract] skip existing teacher shards=${existing_teacher}"
fi

teacher_count="$(find "${TEACHER_ROOT}/samsum" -maxdepth 1 -type f -name '*.pt' | wc -l)"
if [[ "${teacher_count}" -lt 300 ]]; then
  echo "[error] teacher shards incomplete: ${teacher_count}/300" >&2
  exit 1
fi
echo "[extract] teacher shards=${teacher_count}"

if [[ ! -f "${OUT_DIR}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[train] dense no-chat conf0 attention_delta 2-head scorer"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --datasets samsum \
    --n-samples 300 \
    --val-ratio 0.1 \
    --epochs 20 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --target-mode attention_delta \
    --loss-mode mse \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input prob \
    --topk-weight 0.0 \
    --topk-k 128 \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 \
    >"${LOG_ROOT}/train_attention_delta_2head.log" 2>&1
else
  echo "[train] skip existing checkpoint"
fi

if [[ ! -f "${OUT_DIR}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[error] missing checkpoint after train" >&2
  exit 1
fi

echo "[done] teacher=${TEACHER_ROOT}"
echo "[done] student=${OUT_DIR}/checkpoint-best"
