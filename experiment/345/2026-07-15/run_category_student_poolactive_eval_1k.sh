#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
TASK_PATH="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/experiment/345/2026-07-15/tasks/longbench_local"
BASE_OUTPUT="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/budget"

SINGLE_DOC_STUDENT="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/budget/future_pool_student_single_doc_qa_train_1k_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
MULTI_DOC_STUDENT="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/budget/future_pool_student_multi_doc_qa_train_1k_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
SUMMARIZATION_STUDENT="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/budget/future_pool_student_summarization_train_1k_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
FEW_SHOT_STUDENT="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache/results/budget/future_pool_student_few_shot_train_1k_topk128_e10_lr2e-5_tw0.02/checkpoint-best"

run_eval() {
  local label="$1"
  local student_path="$2"
  local tasks="$3"
  local gen_length="$4"
  local output_path="${BASE_OUTPUT}/lm_eval_student_poolactive_category1k_${label}_p1024_a128_mlen2048"

  mkdir -p "${output_path}"
  echo "[start] label=${label} gen_length=${gen_length} tasks=${tasks} output=${output_path} $(date -Is)"

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
    --tasks "${tasks}" \
    --include_path "${TASK_PATH}" \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${student_path},student_prompt_pool_active=True,student_pool_budget=1024,student_budget=128,student_question_window=128" \
    --gen_kwargs "block_length=8,gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --num_fewshot 0 \
    --output_path "${output_path}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code

  echo "[done] label=${label} $(date -Is)"
}

run_eval "multi_doc_qa_g32" "${MULTI_DOC_STUDENT}" "local_longbench_2wikimqa,local_longbench_hotpotqa,local_longbench_musique" 32
run_eval "few_shot_triviaqa_g32" "${FEW_SHOT_STUDENT}" "local_longbench_triviaqa" 32
run_eval "single_doc_qa_multifieldqa_en_g64" "${SINGLE_DOC_STUDENT}" "local_longbench_multifieldqa_en" 64
run_eval "few_shot_trec_g64" "${FEW_SHOT_STUDENT}" "local_longbench_trec" 64
run_eval "single_doc_qa_qasper_narrativeqa_g128" "${SINGLE_DOC_STUDENT}" "local_longbench_qasper,local_longbench_narrativeqa" 128
run_eval "few_shot_samsum_g128" "${FEW_SHOT_STUDENT}" "local_longbench_samsum" 128
run_eval "summarization_g512" "${SUMMARIZATION_STUDENT}" "local_longbench_gov_report,local_longbench_multi_news,local_longbench_qmsum" 512
