#!/usr/bin/env bash
# 2wikimqa + qasper 에 dynkv 프롬프트 50% refresh sweep (2,4,8,16)
# 프롬프트 = max_length - gen_length, budget = 그 절반
#   2wikimqa: gen 32  -> prompt 2016 -> budget 1008,  steps=32
#   qasper  : gen 128 -> prompt 1920 -> budget 960,   steps=128
# 그 외 설정은 samsum 실험과 동일 (block=32, cfg=0.0, student=300each, selection=global)
# report: experiment/report/2026-08-07-samsum-prompt-budget.md
set -uo pipefail

REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"
cd "$REPO"

MODEL_PATH="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
TASK_PATH="${REPO}/experiment/345/2026-08-07/tasks/longbench_local"
STUDENT="${REPO}/results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best"
OUTROOT="/home/M2026107/.cache/lmeval_wiki_qasper_b50_20260808"

# run <task> <budget> <gen_length> <refresh>
run () {
  local task="$1" budget="$2" gen="$3" refresh="$4"
  local name="${task}_b${budget}_refresh${refresh}"
  local output="${OUTROOT}/${name}"
  rm -rf "$output"; mkdir -p "$output"
  echo "=== START ${name} $(date -Is)"

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
    --tasks "local_longbench_${task}" \
    --include_path "${TASK_PATH}" \
    --batch_size 1 --limit 200 \
    --model_args "pretrained=${MODEL_PATH},is_feature_cache=False,is_cfg_cache=False,max_length=2048,student_path=${STUDENT},student_prompt_dynamic_kv=True,student_budget=${budget},student_selection_mode=global,student_refresh_interval=${refresh},student_question_window=128" \
    --gen_kwargs "block_length=32,gen_length=${gen},steps=${gen},cfg_scale=0.0" \
    --num_fewshot 0 \
    --output_path "${output}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code
  echo "=== END ${name} $(date -Is)"
}

for r in 2 4 8 16; do run 2wikimqa 1008 32  "$r"; done
for r in 2 4 8 16; do run qasper   960  128 "$r"; done
echo "=== SWEEP DONE $(date -Is)"
