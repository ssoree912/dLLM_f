#!/usr/bin/env bash
# Decode latency (SAMSum, 50 samples, warmup 2) — 순서대로:
#   origin, dllm_cache(기본 full cache), dynkv refresh=1, dynkv refresh=4,
#   measured(measuredfix) fl=16 measured_tokens=480 (런타임 드리프트 선택)
# 각 설정을 별도 프로세스로 실행 (dllm_cache 훅이 다른 method 오염 안 하도록).
# 스크립트: experiment/345/2026-08-07/measure_decode_latency_dllm.py (dllm_cache method 추가본)
set -uo pipefail

REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"
cd "$REPO"

MODEL="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
DATA="/mnt/srv/home/dlpcg.325/dllm/data/longbench/samsum.jsonl"
STUDENT="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
SCRIPT="${REPO}/experiment/345/2026-08-07/measure_decode_latency_dllm.py"
OUTDIR="/home/M2026107/.cache/decode_latency_samsum_20260808"
mkdir -p "$OUTDIR"

COMMON="--model $MODEL --data $DATA --student $STUDENT --samples 50 --warmup 2 \
  --budget 960 --gen-length 128 --steps 128 --block-length 32 --max-length 2048"

run () {
  local label="$1"; shift
  echo "=== START ${label} $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$REPO" \
  .venv/bin/python "$SCRIPT" $COMMON --output "$OUTDIR/${label}.json" "$@"
  echo "=== END ${label} $(date -Is)"
}

run origin           --methods origin
run dllm_cache       --methods dllm_cache
run dynkv_refresh1   --methods dynkv --refresh-interval 1
run dynkv_refresh4   --methods dynkv --refresh-interval 4
run measuredfix_480  --methods measured --frozen-layers 16 --measured-tokens 480

echo "=== LATENCY DONE $(date -Is)"
