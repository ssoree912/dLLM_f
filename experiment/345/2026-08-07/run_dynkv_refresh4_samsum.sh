#!/usr/bin/env bash
# dynkv 960 (프롬프트 50%), refresh_interval=4
# 보고서 #7 (dynkv 960 refresh=2) 에서 refresh 만 4 로 변경. 그 외 설정 동일.
# report: experiment/report/2026-08-07-samsum-prompt-budget.md
set -euo pipefail

REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"
cd "$REPO"

MODEL_PATH="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
TASK_PATH="${REPO}/experiment/345/2026-08-07/tasks/longbench_local"
STUDENT="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
OUTPUT="/home/M2026107/.cache/lmeval_samsum_b960_20260807/dynkv_960_refresh4"

mkdir -p "$OUTPUT"
echo "=== START dynkv_960_refresh4 $(date -Is)"

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
  --tasks local_longbench_samsum \
  --include_path "${TASK_PATH}" \
  --batch_size 1 --limit 200 \
  --model_args "pretrained=${MODEL_PATH},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${STUDENT},student_prompt_dynamic_kv=True,student_budget=960,student_selection_mode=global,student_refresh_interval=4,student_question_window=128" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 \
  --output_path "${OUTPUT}" \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --trust_remote_code

echo "=== END dynkv_960_refresh4 $(date -Is)"
