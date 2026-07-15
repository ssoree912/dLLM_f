#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/LLaDA-8B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/train/2wikimultihopqa/2wikimultihopqa_train_longbench_format.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiment/2026-07-14/results/revealed_answer_teacher}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
N_SAMPLES="${N_SAMPLES:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
QUESTION_WINDOW="${QUESTION_WINDOW:-128}"

"${PYTHON_BIN}" experiment/2026-07-14/revealed_answer/extract_teacher.py \
  --model "${MODEL_PATH}" \
  --data "${DATA_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-length "${MAX_LENGTH}" \
  --n-samples "${N_SAMPLES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --question-window "${QUESTION_WINDOW}"
