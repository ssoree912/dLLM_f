#!/usr/bin/env bash
set -euo pipefail

cd /mnt/srv/home/dlpcg.325/dllm/dLLM-Cache

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/.venv/bin/python
MODEL=/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct
DATA=/mnt/srv/home/dlpcg.325/dllm/data/train_mixed_8task_300/mixed_train_longbench_format.jsonl
TEACHER=/home/M2026107/.cache/offline_hybrid_teacher_mixed8_chat_20260811
LABELS=/home/M2026107/.cache/offline_hybrid_teacher_mixed8_chat_20260811_labels_b960_r0p50
KEEP_STUDENT=/home/M2026107/.cache/student_offline_hybrid_mixed8_b960_r480_20260811
DELTA_STUDENT=/home/M2026107/.cache/student_offline_delta_mixed8_refresh480_20260811

DATASETS=(
  2wikimultihopqa_train
  hotpotqa
  gov_report
  multi_news
  musique
  narrativeqa
  qasper
  qmsum
)

echo "[stage teacher] extracting/resuming 300 samples for each mixed-8 dataset"
"${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_teacher \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output-root "${TEACHER}" \
  --datasets "${DATASETS[@]}" \
  --samples-per-dataset 300 \
  --max-length 2048 \
  --question-window 128 \
  --gen-length 128 \
  --block-length 32 \
  --steps 128 \
  --active-top-k 128 \
  --temperature 0 \
  --confidence-weight \
  --target-aggregation max \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt-format train \
  --apply-chat-template

echo "[stage labels] building/resuming B=960, K_R=480 hybrid labels"
"${PYTHON}" -m dllm_cache.budget.build_hybrid_labels \
  --teacher-root "${TEACHER}" \
  --datasets "${DATASETS[@]}" \
  --budget 960 \
  --ref-ratio 0.5 \
  --seed 1234

if [[ ! -f "${KEEP_STUDENT}/training.done" ]]; then
  echo "[stage keep-student] training the hybrid top-960 selector"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${LABELS}" \
    --output-dir "${KEEP_STUDENT}" \
    --model "${MODEL}" \
    --datasets "${DATASETS[@]}" \
    --val-ratio 0.1 \
    --epochs 10 \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode hybrid_mask \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0.02 \
    --topk-k 960 \
    --topk-positive-weight 1 \
    --max-grad-norm 1 \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 100 \
    --proj-dim 256 \
    --mlp-dim 512
  touch "${KEEP_STUDENT}/training.done"
else
  echo "[stage keep-student] already complete; skipping"
fi

if [[ ! -f "${DELTA_STUDENT}/training.done" ]]; then
  echo "[stage delta-student] training the top-480 refresh selector"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER}" \
    --output-dir "${DELTA_STUDENT}" \
    --model "${MODEL}" \
    --datasets "${DATASETS[@]}" \
    --val-ratio 0.1 \
    --epochs 10 \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode delta \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0.02 \
    --topk-k 480 \
    --topk-positive-weight 8 \
    --max-grad-norm 1 \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 100 \
    --proj-dim 256 \
    --mlp-dim 512
  touch "${DELTA_STUDENT}/training.done"
else
  echo "[stage delta-student] already complete; skipping"
fi

echo "[done] mixed-8 teacher, labels, keep student, and delta student are complete"
