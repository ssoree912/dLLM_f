#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="results/budget/lm_eval_student_poolactive_futurepool_train_300each_p1024_a128_mlen2048_all"
STUDENT_PATH="results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
MODEL_PATH="../model/LLaDA-8B-Instruct"
TASK_PATH="experiment/345/2026-07-15/tasks/longbench_local"
TASKS="local_longbench_samsum,local_longbench_narrativeqa,local_longbench_qasper"

mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE="${TMPDIR:-/tmp}/dllm_hf_datasets_cache" \
HF_MODULES_CACHE="${TMPDIR:-/tmp}/dllm_hf_modules_cache" \
HF_ALLOW_CODE_EVAL=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASKKV_ENABLED=0 \
.venv/bin/python -m accelerate.commands.accelerate_cli launch \
  --config_file accelerate_config_single_gpu.yaml \
  evaluation_script.py run \
  --model LLaDA \
  --tasks "${TASKS}" \
  --include_path "${TASK_PATH}" \
  --batch_size 1 \
  --model_args "pretrained=${MODEL_PATH},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${STUDENT_PATH},student_prompt_pool_active=True,student_pool_budget=1024,student_budget=128,student_question_window=128" \
  --gen_kwargs "block_length=8,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code

.venv/bin/python experiment/345/2026-07-15/update_origin_cache_budget_1024_no_cache.py
