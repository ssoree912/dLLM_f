#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_DIR [LONGBENCH_TASK]" >&2
  exit 2
fi

CHECKPOINT=$1
OUTPUT_DIR=$2
TASK=${3:-longbench_samsum}

export CUDA_VISIBLE_DEVICES=2
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/opt/conda/envs/dllm/bin/python
MODEL=/workspace/dllm/model/LLaDA-8B-Instruct
TASK_ROOT=experiment/345/2026-08-13/tasks/longbench_full_local

gpu_pids="$(nvidia-smi -i 2 --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${gpu_pids}" ]]; then
  echo "[error] physical GPU 2 is busy; PIDs=${gpu_pids}" >&2
  exit 1
fi

# prune_cache_path is the entire cache configuration.  There is intentionally
# no chat-template flag and no budget/refresh/select-once setting here.
taskset --cpu-list 0-7 "${PYTHON}" evaluation_script.py run \
  --model LLaDA \
  --tasks "${TASK}" \
  --include_path "${TASK_ROOT}" \
  --batch_size 1 \
  --model_args "pretrained=${MODEL},prune_cache_path=${CHECKPOINT},trust_remote_code=True" \
  --num_fewshot 0 \
  --log_samples \
  --trust_remote_code \
  --output_path "${OUTPUT_DIR}"

nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
