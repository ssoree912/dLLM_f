#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 SOURCE_ROOT TEACHER_ROOT STUDENT_OUT DATASET [DATASET ...]" >&2
  exit 2
fi

SOURCE_ROOT=$1
TEACHER_ROOT=$2
STUDENT_OUT=$3
shift 3
DATASETS=("$@")

# One physical GPU only.  Inside the isolated process it is cuda:0.
export CUDA_VISIBLE_DEVICES=2
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PYTHON=/opt/conda/envs/dllm/bin/python
MODEL=/workspace/dllm/model/LLaDA-8B-Instruct

gpu_pids="$(nvidia-smi -i 2 --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${gpu_pids}" ]]; then
  echo "[error] physical GPU 2 is busy; PIDs=${gpu_pids}" >&2
  exit 1
fi

echo "[teacher] no-chat dense confidence-weighted Attention + cumulative Delta"
taskset --cpu-list 0-7 "${PYTHON}" -m dllm_cache.budget.extract_prune_cache_teacher \
  --model "${MODEL}" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${TEACHER_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --device cuda:0 \
  --dtype bfloat16

echo "[train] one shared scorer with attention and delta heads"
taskset --cpu-list 0-7 "${PYTHON}" -m dllm_cache.budget.train_prune_cache_student \
  --teacher-root "${TEACHER_ROOT}" \
  --output-dir "${STUDENT_OUT}" \
  --model "${MODEL}" \
  --datasets "${DATASETS[@]}" \
  --device cuda:0 \
  --dtype bfloat16

echo "[done] teacher=${TEACHER_ROOT}"
echo "[done] student=${STUDENT_OUT}/checkpoint-best"
nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
