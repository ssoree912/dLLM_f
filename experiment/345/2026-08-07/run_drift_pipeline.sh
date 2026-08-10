#!/usr/bin/env bash
# 우리 방법 드리프트 훈련 end-to-end: teacher(samsum test[50:200]) -> student -> eval(test[0:50])
# disjoint split (오염 없음), 동일 test 포맷.
set -uo pipefail
REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"; D="$REPO/experiment/345/2026-08-07"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:$D"
PY="$REPO/.venv/bin/python"
M="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
S="$REPO/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
TROOT="/home/M2026107/.cache/refresh_teacher_samsum_20260809"
SCK="$REPO/results/refresh_student_samsum_20260809"

echo "=== [1/3] teacher 추출 (samsum test[50:200], 150개) $(date -Is)"
$PY "$D/refresh_teacher_run.py" --model "$M" --student "$S" --task samsum \
  --output-root "$TROOT" --skip 50 --samples 150 --refresh-tokens 240 2>&1 | grep -E "task=|done|Error|Traceback|refresh-teacher 15?0/" | tail -20
echo "=== [2/3] student 학습 $(date -Is)"
$PY "$D/refresh_student_train.py" --model "$M" --teacher-root "$TROOT" \
  --output-dir "$SCK" --epochs 5 --refresh-tokens 240 --min-layer 0 2>&1 | grep -E "epoch|done|Error|Traceback" | tail -12
echo "=== [3/3] eval (samsum test[0:50]): dynkv(periodic) vs oracle vs student_refresh $(date -Is)"
$PY "$D/refresh_gate.py" --model "$M" --student "$S" --task samsum --samples 50 \
  --methods dynkv oracle student_refresh --refresh-intervals 4 --refresh-tokens 240 \
  --student-refresh-ckpt "$SCK/checkpoint-last.pt" \
  --output "/home/M2026107/.cache/drift_eval_samsum_20260809.json" 2>&1 | grep -E "task=|rouge_score|done|Error|Traceback" | tail -12
echo "=== DRIFT PIPELINE DONE $(date -Is)"
