#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the LLaDA-8B-Instruct checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to the original 2Wiki train JSONL}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for schema-v2 shards}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
export PYTHONPATH="${EXPERIMENT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN:-python}" -m step_distill.extract_teacher \
  --model "${MODEL_PATH}" \
  --data "${DATA_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --sample-limit "${SAMPLE_LIMIT:-2}" \
  --gen-length "${GEN_LENGTH:-32}" \
  --steps "${STEPS:-32}" \
  --block-length "${BLOCK_LENGTH:-8}" \
  --max-target-k "${MAX_TARGET_K:-512}" \
  --diversity-gamma "${DIVERSITY_GAMMA:-0.1}" \
  --seed "${SEED:-4090}"
