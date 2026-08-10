#!/usr/bin/env bash
# New mixed-8-task chat teacher student, then the periodic-full-refresh sweep.
#
# Run 1 isolates the student: same measured-480 config the old students were
# scored under, so its delta is attributable to the teacher change alone. Runs
# 2-5 then vary only full_refresh_interval on top of it.
set -uo pipefail
cd /home/M2026107/dllm/dLLM-Cache

export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/home/M2026107/dllm/model/LLaDA-8B-Instruct
S=/home/M2026107/.cache/student_mixed8_chat_20260810_topk960/checkpoint-best
OUT=results/sweep_full_refresh_20260810
LIMIT=100
mkdir -p $OUT logs

run () {
  local name=$1 interval=$2
  echo "=== START $name (interval=$interval) $(date +%H:%M:%S)"
  .venv/bin/python evaluation_script.py \
    --model LLaDA --tasks local_longbench_samsum \
    --include_path experiment/345/2026-07-15/tasks/longbench_local \
    --batch_size 1 --limit $LIMIT \
    --model_args "pretrained=$M,is_feature_cache=False,is_cfg_cache=False,max_length=2048,\
student_path=$S,student_prompt_layer_split=True,student_budget=960,\
student_frozen_layers=16,student_measured_tokens=480,\
student_full_refresh_interval=$interval,student_question_window=128" \
    --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
    --num_fewshot 0 --log_samples --apply_chat_template \
    --fewshot_as_multiturn --trust_remote_code \
    --output_path $OUT/$name \
    > logs/sweep_${name}.log 2>&1
  echo "=== END $name $(date +%H:%M:%S)"
  grep -E "rouge|classification" logs/sweep_${name}.log | tail -3
}

run interval0   0
run interval16 16
run interval8   8
run interval4   4
run interval100 100
echo "SWEEP DONE $(date +%H:%M:%S)"
