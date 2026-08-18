#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/M2026107/dllm/dLLM-Cache/.venv/bin/python"
MODEL="${MODEL:-/home/M2026107/dllm/model/LLaDA-8B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/M2026107/dllm/dLLM-Cache/results/budget/future_pool_teacher_train_300each_g128_top128}"
TEACHER_ROOT="${TEACHER_ROOT:-/home/M2026107/.cache/offline_hybrid_teacher_samsum300_dense_nochat_20260818}"
ATTN_OUT="${ATTN_OUT:-results/budget/student_samsum300_dense_nochat_attention_score_rank0p1_topk0_e20_20260818}"
DELTA_STUDENT="${DELTA_STUDENT:-/home/M2026107/dllm/dLLM-Cache/results/budget/student_samsum300_cumdelta_chat_delta_rank0p1_topk0_20260817_cumbranch/checkpoint-best}"
TASK_ROOT="${TASK_ROOT:-experiment/345/2026-08-13/tasks/longbench_full_local}"
EVAL_OUT="${EVAL_OUT:-results/samsum_dense_nochat_attention_cumdelta_chat_b960_r480_once_20260818/limit200}"
LOG_ROOT="${LOG_ROOT:-logs/samsum300_dense_nochat_attention_20260818}"

mkdir -p "${LOG_ROOT}"

echo "[config] root=${ROOT}"
echo "[config] teacher=${TEACHER_ROOT}"
echo "[config] attention_student=${ATTN_OUT}"
echo "[config] delta_student=${DELTA_STUDENT}"
echo "[config] eval_out=${EVAL_OUT}"

if [[ ! -d "${SOURCE_ROOT}/samsum" ]]; then
  echo "[error] missing source samsum shards: ${SOURCE_ROOT}/samsum" >&2
  exit 1
fi

if [[ ! -f "${DELTA_STUDENT}/pytorch_model.bin" ]]; then
  echo "[error] missing delta checkpoint: ${DELTA_STUDENT}/pytorch_model.bin" >&2
  exit 1
fi

existing_teacher=0
if [[ -d "${TEACHER_ROOT}/samsum" ]]; then
  existing_teacher="$(find "${TEACHER_ROOT}/samsum" -maxdepth 1 -type f -name '*.pt' | wc -l)"
fi

if [[ "${existing_teacher}" -lt 300 ]]; then
  echo "[extract] no-chat dense attention+delta teacher, samsum 300, active_top_k=0"
  "${PYTHON}" -m dllm_cache.budget.extract_offline_hybrid_from_shards \
    --model "${MODEL}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${TEACHER_ROOT}" \
    --datasets samsum \
    --n-samples 300 \
    --device cuda:0 \
    --dtype bfloat16 \
    --gen-length 128 \
    --block-length 8 \
    --steps 128 \
    --active-top-k 0 \
    --confidence-weight \
    --target-aggregation max \
    >"${LOG_ROOT}/extract_teacher.log" 2>&1
else
  echo "[extract] skip existing teacher shards=${existing_teacher}"
fi

teacher_count="$(find "${TEACHER_ROOT}/samsum" -maxdepth 1 -type f -name '*.pt' | wc -l)"
if [[ "${teacher_count}" -lt 300 ]]; then
  echo "[error] teacher shards incomplete: ${teacher_count}/300" >&2
  exit 1
fi
echo "[extract] teacher shards=${teacher_count}"

if [[ ! -f "${ATTN_OUT}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[train] no-chat dense attention score-only scorer"
  "${PYTHON}" -m dllm_cache.budget.train_student \
    --teacher-root "${TEACHER_ROOT}" \
    --output-dir "${ATTN_OUT}" \
    --model "${MODEL}" \
    --datasets samsum \
    --n-samples 300 \
    --val-ratio 0.1 \
    --epochs 20 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --target-mode score \
    --loss-mode mse \
    --rank-weight 0.1 \
    --rank-margin 0.05 \
    --rank-top-ratio 0.2 \
    --rank-bottom-ratio 0.4 \
    --rank-input prob \
    --topk-weight 0.0 \
    --topk-k 128 \
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 0 \
    --log-every 10 \
    --proj-dim 256 \
    --mlp-dim 512 \
    >"${LOG_ROOT}/train_attention.log" 2>&1
else
  echo "[train] skip existing attention checkpoint"
fi

if [[ ! -f "${ATTN_OUT}/checkpoint-best/pytorch_model.bin" ]]; then
  echo "[error] missing attention checkpoint after train" >&2
  exit 1
fi

echo "[eval] SAMSum 200: no-chat dense attention top960 + chat cumulative delta refresh480"
mkdir -p "${EVAL_OUT}"
"${PYTHON}" evaluation_script.py run \
  --model LLaDA \
  --tasks longbench_samsum \
  --include_path "${TASK_ROOT}" \
  --batch_size 1 \
  --limit 200 \
  --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${ATTN_OUT}/checkpoint-best,student_refresh_path=${DELTA_STUDENT},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=960,student_refresh_tokens=480,student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code \
  --output_path "${EVAL_OUT}" \
  >"${LOG_ROOT}/eval_samsum.log" 2>&1

echo "[done] teacher=${TEACHER_ROOT}"
echo "[done] attention_student=${ATTN_OUT}/checkpoint-best"
echo "[done] delta_student=${DELTA_STUDENT}"
echo "[done] eval_out=${EVAL_OUT}"
