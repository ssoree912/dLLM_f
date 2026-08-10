#!/usr/bin/env bash
# 공식 lm-eval 프로토콜로 drift refresh 평가.
#   TASK=samsum|trec|qasper|2wikimqa   CHAT=1|0   LIMIT=n   GEN=생성길이   BUDGET=keep
#   EXTRA="model_args 추가 인자"       NAME=출력이름   OUT=출력루트
# 기본은 samsum + chat template. trec 은 CHAT=0 으로 호출한다.
set -uo pipefail
REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
M="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
TASK_PATH="${REPO}/experiment/345/2026-08-07/tasks/longbench_local"
S="results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"

TASK="${TASK:-samsum}"
CHAT="${CHAT:-1}"
LIMIT="${LIMIT:-100}"
GEN="${GEN:-128}"
BUDGET="${BUDGET:-960}"
EXTRA="${EXTRA:-}"
NAME="${NAME:-run}"
OUT="${OUT:-/home/M2026107/.cache/official_drift_$(date +%Y%m%d)}"

CHAT_ARGS="--apply_chat_template --fewshot_as_multiturn"
[ "$CHAT" = "0" ] && CHAT_ARGS=""

echo "=== START ${NAME} task=${TASK} chat=${CHAT} limit=${LIMIT} gen=${GEN} budget=${BUDGET} $(date -Is)"
.venv/bin/python evaluation_script.py \
  --model LLaDA --tasks "local_longbench_${TASK}" \
  --include_path "$TASK_PATH" \
  --batch_size 1 --limit "$LIMIT" \
  --model_args "pretrained=${M},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${S},student_budget=${BUDGET},student_question_window=128,${EXTRA}" \
  --gen_kwargs "block_length=32,gen_length=${GEN},steps=${GEN},cfg_scale=0.0" \
  --num_fewshot 0 --log_samples ${CHAT_ARGS} \
  --trust_remote_code --output_path "${OUT}/${NAME}" 2>&1 | tail -6
echo "=== END ${NAME} $(date -Is)"
