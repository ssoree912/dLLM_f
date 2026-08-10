#!/usr/bin/env bash
set -uo pipefail
REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO"
M="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
S="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
OUT="/home/M2026107/.cache/refresh_gate_20260809"; mkdir -p "$OUT"
echo "=== START gate trec $(date -Is)"
.venv/bin/python experiment/345/2026-08-07/refresh_gate.py \
  --model "$M" --student "$S" --output "$OUT/trec.json" \
  --task trec --samples 30 \
  --methods full dynkv oracle --refresh-intervals 1 4 --refresh-tokens 248 0
echo "=== END gate trec $(date -Is)"
