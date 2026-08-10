#!/usr/bin/env bash
# 공식 lm-eval, samsum 200행: drift oracle R=240 -> drift student R=240
# 비교 기준(동일 프로토콜, 기존 공식 수치): dynkv refresh=4 = 0.3822, origin = 0.3835
set -uo pipefail
R="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/experiment/345/2026-08-07"
OUTROOT=/home/M2026107/.cache/official_drift_20260810
CK=/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/refresh_student_samsum_20260809/checkpoint-last.pt

LIMIT=200 NAME=oracle_R240 OUT=$OUTROOT \
  EXTRA="student_prompt_drift_refresh=True,student_drift_mode=oracle,student_refresh_tokens=240" \
  bash "$R/run_official_drift.sh"

LIMIT=200 NAME=student_R240 OUT=$OUTROOT \
  EXTRA="student_prompt_drift_refresh=True,student_drift_mode=student,student_refresh_tokens=240,student_drift_ckpt=${CK}" \
  bash "$R/run_official_drift.sh"

echo "=== OFFICIAL DRIFT DONE $(date -Is)"
