#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/LLaDA-8B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/longbench/2wikimqa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiment/2026-07-14/results/revealed_answer_oracle_2wikimqa_b128}"
LIMIT="${LIMIT:-20}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BUDGET="${BUDGET:-128}"
GEN_LENGTH="${GEN_LENGTH:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
STEPS="${STEPS:-32}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
QUESTION_WINDOW="${QUESTION_WINDOW:-128}"

"${PYTHON_BIN}" experiment/2026-07-14/revealed_answer/eval_teacher_oracle_2wikimqa.py \
  --model "${MODEL_PATH}" \
  --data "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --max-length "${MAX_LENGTH}" \
  --budget "${BUDGET}" \
  --gen-length "${GEN_LENGTH}" \
  --block-length "${BLOCK_LENGTH}" \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --question-window "${QUESTION_WINDOW}"
