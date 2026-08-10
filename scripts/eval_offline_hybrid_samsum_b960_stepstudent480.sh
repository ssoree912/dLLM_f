#!/usr/bin/env bash
set -euo pipefail

cd /home/M2026107/dllm/dLLM-Cache

export CUDA_VISIBLE_DEVICES=0
export MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/home/M2026107/dllm/model/LLaDA-8B-Instruct
KEEP_STUDENT=/home/M2026107/.cache/student_offline_hybrid_samsum_b960_r480_20260811/checkpoint-best
DELTA_STUDENT=/home/M2026107/.cache/student_offline_delta_samsum_refresh480_20260811/checkpoint-best
LIMIT=${LIMIT:-100}
OUTPUT=${OUTPUT:-results/offline_hybrid_samsum_b960_stepstudent480_20260811/limit${LIMIT}}

.venv/bin/python evaluation_script.py \
  --model LLaDA \
  --tasks local_longbench_samsum \
  --include_path experiment/345/2026-07-15/tasks/longbench_local \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --model_args "pretrained=${MODEL},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${KEEP_STUDENT},student_refresh_path=${DELTA_STUDENT},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_budget=960,student_refresh_tokens=480,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code \
  --output_path "${OUTPUT}"
