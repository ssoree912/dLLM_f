#!/usr/bin/env bash
# 2wikimqa + qasper origin baseline (pruning 없음, full 프롬프트)
# dynkv refresh sweep 와 동일한 생성 설정, student 만 제거.
#   2wikimqa: gen 32,  steps 32
#   qasper  : gen 128, steps 128
# report: experiment/report/2026-08-07-samsum-prompt-budget.md
set -uo pipefail

REPO="/mnt/srv/home/dlpcg.325/dllm/dLLM-Cache"
cd "$REPO"

MODEL_PATH="/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct"
TASK_PATH="${REPO}/experiment/345/2026-08-07/tasks/longbench_local"
OUTROOT="/home/M2026107/.cache/lmeval_wiki_qasper_b50_20260808"

# run <task> <gen_length>
run () {
  local task="$1" gen="$2"
  local name="${task}_origin"
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
    --model_args "pretrained=${MODEL_PATH},is_feature_cache=False,is_cfg_cache=False,max_length=2048" \
    --gen_kwargs "block_length=32,gen_length=${gen},steps=${gen},cfg_scale=0.0" \
    --num_fewshot 0 \
    --output_path "${output}" \
    --log_samples \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code
  echo "=== END ${name} $(date -Is)"
}

run 2wikimqa 32
run qasper   128
echo "=== ORIGIN DONE $(date -Is)"
