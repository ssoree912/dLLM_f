#!/usr/bin/env bash
# measured `step` baseline on 2WikiMultihopQA, then origin on the same 50 rows.
#
# 50 rows is a direction check, not a verdict: 2wikimqa QA-F1 carried +-0.03 at
# n=200, so the interval here is wide enough to hide any difference smaller than
# the pruning gain itself.
set -uo pipefail
cd /home/M2026107/dllm/dLLM-Cache
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/home/M2026107/dllm/model/LLaDA-8B-Instruct
S=results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best
OUT=results/measured_baseline_2wikimqa_20260810
LIMIT=50
mkdir -p $OUT logs

run () {
  local name=$1 extra=$2
  echo "=== START $name $(date +%H:%M:%S)"
  .venv/bin/python evaluation_script.py \
    --model LLaDA --tasks local_longbench_2wikimqa \
    --include_path experiment/345/2026-07-15/tasks/longbench_local \
    --batch_size 1 --limit $LIMIT \
    --model_args "pretrained=$M,is_feature_cache=False,is_cfg_cache=False,max_length=2048${extra}" \
    --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
    --num_fewshot 0 --log_samples --apply_chat_template \
    --fewshot_as_multiturn --trust_remote_code \
    --output_path $OUT/$name \
    > logs/2wiki_${name}.log 2>&1
  echo "=== END $name $(date +%H:%M:%S)"
  grep -E "\|score\|" logs/2wiki_${name}.log | tail -1
}

run measured480_step_n50 ",student_path=$S,student_prompt_layer_split=True,student_budget=960,student_frozen_layers=16,student_measured_tokens=480,student_measured_baseline=step,student_question_window=128"
run origin_n50 ""
echo "DONE $(date +%H:%M:%S)"
