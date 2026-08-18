#!/usr/bin/env bash
set -euo pipefail

cd /home/M2026107/dllm/dLLM-Cache-dlpc-kv-pruning

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN=/home/M2026107/dllm/dLLM-Cache/.venv/bin/python
CKPT=results/budget/student_samsum300_dense_nochat_conf0_attention_delta_2head_rank0p1_topk0_e20_20260818/checkpoint-best
OUT_DIR=results/samsum300_dense_nochat_conf0_attention_delta_2head_b960_r480_once_20260818/limit200

exec "${PYTHON_BIN}" evaluation_script.py \
  --model LLaDA \
  --model_args "pretrained=/home/M2026107/dllm/model/LLaDA-8B-Instruct,is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${CKPT},student_refresh_path=${CKPT},student_prompt_drift_refresh=True,student_drift_mode=delta_student,student_delta_select_once=True,student_budget=960,student_refresh_tokens=480,student_refresh_interval=1,student_drift_frozen_layers=0,student_question_window=128,student_score_activation=softmax,trust_remote_code=True" \
  --tasks longbench_samsum \
  --include_path experiment/345/2026-08-13/tasks/longbench_full_local \
  --batch_size 1 \
  --limit 200 \
  --device cuda:0 \
  --output_path "${OUT_DIR}" \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0"
