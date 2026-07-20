#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/M2026107/.conda/envs/dave-llada/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct}"
TEACHER_ROOT="${TEACHER_ROOT:-experiment/345/2026-07-15/results/full_dynamic_teacher_balanced_plus_wiki_2k}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment/345/2026-07-15/results/full_dynamic_student_balanced_plus_wiki_2k_n21238_topk128_e10_lr5e-5_tw0.02}"
EPOCHS="${EPOCHS:-10}"
RESUME_FROM=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-last)
      RESUME_FROM="${OUTPUT_DIR}/checkpoint-last"
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  experiment/2026-07-14/revealed_answer/train_student.py
  --teacher-root "${TEACHER_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_PATH}"
  --datasets
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
  --n-samples 0
  --val-ratio 0.1
  --epochs "${EPOCHS}"
  --lr 5e-5
  --weight-decay 0.0
  --rank-weight 0.1
  --rank-margin 0.05
  --rank-top-ratio 0.2
  --rank-bottom-ratio 0.4
  --topk-weight 0.02
  --topk-k 128
  --topk-positive-weight 8.0
  --max-grad-norm 1.0
  --device cuda:0
  --dtype bfloat16
  --seed 0
  --log-every 100
  --proj-dim 256
  --mlp-dim 512
)

if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

exec "${CMD[@]}" "${EXTRA_ARGS[@]}"
