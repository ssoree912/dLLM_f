#!/usr/bin/env bash
# keep 480 (프롬프트 25%), R=120 (=B/4), fl=16 — random vs student
set -uo pipefail
D="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/experiment/345/2026-08-07"
OUT=/home/M2026107/.cache/official_drift_20260810
CK=/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/refresh_student_samsum_20260809/checkpoint-last.pt

TASK=samsum CHAT=1 LIMIT=100 GEN=128 BUDGET=480 NAME=samsum_random_B480_R120_fl16 OUT=$OUT \
  EXTRA="student_prompt_drift_refresh=True,student_drift_mode=random,student_refresh_tokens=120,student_drift_frozen_layers=16" \
  bash "$D/run_official_drift.sh"

TASK=samsum CHAT=1 LIMIT=100 GEN=128 BUDGET=480 NAME=samsum_student_B480_R120_fl16 OUT=$OUT \
  EXTRA="student_prompt_drift_refresh=True,student_drift_mode=student,student_refresh_tokens=120,student_drift_frozen_layers=16,student_drift_ckpt=${CK}" \
  bash "$D/run_official_drift.sh"

echo "=== B480 DONE $(date -Is)"
