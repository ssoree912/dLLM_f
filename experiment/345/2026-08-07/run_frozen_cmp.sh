#!/usr/bin/env bash
set -uo pipefail
REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO"
M="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
S="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
OUT="/home/M2026107/.cache/refresh_gate_20260809"; mkdir -p "$OUT"
echo "=== START frozen_cmp samsum $(date -Is)"
# frozen(kv_cache, per-layer keep) vs oracle_R0(shared-keep frozen), same 30 samples
.venv/bin/python experiment/345/2026-08-07/refresh_gate.py \
  --model "$M" --student "$S" --output "$OUT/frozen_cmp_samsum.json" \
  --task samsum --samples 30 \
  --methods frozen oracle --refresh-tokens 0
echo "=== END frozen_cmp samsum $(date -Is)"
