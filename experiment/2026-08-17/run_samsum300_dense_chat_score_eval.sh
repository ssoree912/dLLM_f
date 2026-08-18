#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${ROOT}/.venv/bin/python"
elif [[ -x "/home/M2026107/dllm/dLLM-Cache/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="/home/M2026107/dllm/dLLM-Cache/.venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi

PYTHON="${DLPC_PYTHON:-${DEFAULT_PYTHON}}"
MODEL="${MODEL:-/home/M2026107/dllm/model/LLaDA-8B-Instruct}"
DATA="${DATA:-/home/M2026107/dllm/data/train/samsum/samsum_train_longbench_format.jsonl}"
TASK_ROOT="${TASK_ROOT:-experiment/345/2026-08-13/tasks/longbench_full_local}"
RUN_TAG="${RUN_TAG:-20260818}"
DEVICE="${DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASKKV_ENABLED="${MASKKV_ENABLED:-0}"

SAMPLES="${SAMPLES:-300}"
EPOCHS="${EPOCHS:-20}"
LIMIT="${LIMIT:-200}"
KEEP_BUDGET="${KEEP_BUDGET:-960}"
REFRESH_TOKENS="${REFRESH_TOKENS:-480}"

TEACHER="${TEACHER:-/home/M2026107/.cache/dlpc_kv_pruning_teacher_samsum${SAMPLES}_dense_chat_${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-results/budget/dlpc_kv_pruning_samsum${SAMPLES}_dense_chat_${RUN_TAG}}"
ATTN_OUT="${ATTN_OUT:-${OUT_ROOT}/attention_score_rank0p1_topk0_e${EPOCHS}}"
DELTA_OUT="${DELTA_OUT:-${OUT_ROOT}/delta_score_rank0p1_topk0_e${EPOCHS}}"
EVAL_OUT="${EVAL_OUT:-results/samsum${SAMPLES}_dense_chat_b${KEEP_BUDGET}_r${REFRESH_TOKENS}_once_${RUN_TAG}/limit${LIMIT}}"
LOG_DIR="${LOG_DIR:-logs/dlpc_kv_pruning_samsum${SAMPLES}_dense_chat_${RUN_TAG}}"

mkdir -p "${LOG_DIR}"

count_teacher() {
  find "${TEACHER}/samsum" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l
}

echo "[plan] teacher=${TEACHER}"
echo "[plan] attention=${ATTN_OUT}"
echo "[plan] delta=${DELTA_OUT}"
echo "[plan] eval=${EVAL_OUT}"
echo "[plan] active_top_k=0 dense attention, chat-template, samples=${SAMPLES}, epochs=${EPOCHS}"

if (( $(count_teacher) < SAMPLES )); then
  echo "[extract] samsum dense chat teacher $(count_teacher)/${SAMPLES}"
  "${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_teacher \
    --model "${MODEL}" \
    --data "${DATA}" \
    --output-root "${TEACHER}" \
    --datasets samsum \
    --samples-per-dataset "${SAMPLES}" \
    --max-length 2048 \
    --question-window 128 \
    --gen-length 128 \
    --block-length 8 \
    --steps 128 \
    --active-top-k 0 \
    --temperature 0 \
    --confidence-weight \
    --target-aggregation max \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --prompt-format train \
    --apply-chat-template \
    >"${LOG_DIR}/extract_samsum_dense_chat.log" 2>&1
else
  echo "[extract] teacher already complete: $(count_teacher)/${SAMPLES}"
fi

if [[ ! -f "${ATTN_OUT}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[train] attention scorer"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER}" \
    --output-dir "${ATTN_OUT}" \
    --model "${MODEL}" \
    --datasets samsum \
    --val-ratio 0.1 \
    --epochs "${EPOCHS}" \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode score \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0 \
    --max-grad-norm 1 \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 \
    >"${LOG_DIR}/train_attention.log" 2>&1
else
  echo "[train] attention checkpoint exists; skip"
fi

if [[ ! -f "${DELTA_OUT}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[train] delta scorer"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER}" \
    --output-dir "${DELTA_OUT}" \
    --model "${MODEL}" \
    --datasets samsum \
    --val-ratio 0.1 \
    --epochs "${EPOCHS}" \
    --lr 2e-5 \
    --weight-decay 0 \
    --target-mode delta \
    --loss-mode auto \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input auto \
    --topk-weight 0 \
    --max-grad-norm 1 \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 \
    >"${LOG_DIR}/train_delta.log" 2>&1
else
  echo "[train] delta checkpoint exists; skip"
fi

echo "[eval] samsum dense attention top-${KEEP_BUDGET}, delta refresh-${REFRESH_TOKENS}, select once"
"${PYTHON}" evaluation_script.py run \
  --model LLaDA \
  --tasks longbench_samsum \
  --include_path "${TASK_ROOT}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${ATTN_OUT}/checkpoint-best,student_refresh_path=${DELTA_OUT}/checkpoint-best,student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=${KEEP_BUDGET},student_refresh_tokens=${REFRESH_TOKENS},student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code \
  --output_path "${EVAL_OUT}" \
  >"${LOG_DIR}/eval_samsum.log" 2>&1

echo "[done] teacher=${TEACHER}"
echo "[done] attention=${ATTN_OUT}/checkpoint-best"
echo "[done] delta=${DELTA_OUT}/checkpoint-best"
echo "[done] eval=${EVAL_OUT}"
