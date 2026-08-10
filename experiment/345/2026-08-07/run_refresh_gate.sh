#!/usr/bin/env bash
# Phase 0 gate: does step-conditioned oracle refresh beat periodic/frozen at matched compute?
# samsum (gen128 -> B=960, R=240=B/4) then trec (gen64 -> B=992, R=248~B/4).
# methods: full, dynkv(r1=ceiling, r4=periodic k4), oracle(R=B/4, R0=frozen shared-keep)
set -uo pipefail
REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"
cd "$REPO"
M="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
S="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
OUT="/home/M2026107/.cache/refresh_gate_20260809"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO"

run () {
  local task="$1" rt="$2"
  echo "=== START gate ${task} $(date -Is)"
  .venv/bin/python experiment/345/2026-08-07/refresh_gate.py \
    --model "$M" --student "$S" --output "$OUT/${task}.json" \
    --task "$task" --samples 30 \
    --methods full dynkv oracle --refresh-intervals 1 4 --refresh-tokens "$rt" 0
  echo "=== END gate ${task} $(date -Is)"
}

run trec   248
run samsum 240
echo "=== GATE DONE $(date -Is)"
