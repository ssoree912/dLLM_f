#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/LLaDA-8B-Instruct}"
TEACHER_ROOT="${TEACHER_ROOT:-${REPO_ROOT}/experiment/2026-07-14/results/revealed_answer_teacher}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiment/2026-07-14/results/revealed_answer_student}"
N_SAMPLES="${N_SAMPLES:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-1}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
PROJ_DIM="${PROJ_DIM:-256}"
MLP_DIM="${MLP_DIM:-512}"
DATASETS="${DATASETS:-2wikimultihopqa_train}"

"${PYTHON_BIN}" experiment/2026-07-14/revealed_answer/train_student.py \
  --teacher-root "${TEACHER_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL_PATH}" \
  --datasets ${DATASETS} \
  --n-samples "${N_SAMPLES}" \
  --val-ratio "${VAL_RATIO}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --proj-dim "${PROJ_DIM}" \
  --mlp-dim "${MLP_DIM}"
