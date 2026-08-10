#!/usr/bin/env bash
# measured-mode baseline comparison on 2WikiMultihopQA.
#
# SAMSum could not separate the two rules: 0.3761 vs 0.3688 at n=200 is inside
# the +-0.014 standard error. 2wikimqa is the one task where prompt pruning beat
# origin (0.1709 vs 0.1424), so it has headroom for the refresh rule to matter.
#
# Only one job fits on the 40GB card, so wait for whatever holds the GPU first.
set -uo pipefail
cd /home/M2026107/dllm/dLLM-Cache

if [ -n "${WAIT_PID:-}" ]; then
  echo "[wait] PID $WAIT_PID $(date +%H:%M:%S)"
  while [ -d /proc/$WAIT_PID ]; do sleep 60; done
  echo "[wait] released $(date +%H:%M:%S)"
fi

export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/home/M2026107/dllm/model/LLaDA-8B-Instruct
S=results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best
OUT=results/measured_baseline_2wikimqa_20260810
LIMIT=200
mkdir -p $OUT logs

run () {
  local name=$1 baseline=$2 meas=$3
  echo "=== START $name (baseline=$baseline meas=$meas) $(date +%H:%M:%S)"
  .venv/bin/python evaluation_script.py \
    --model LLaDA --tasks local_longbench_2wikimqa \
    --include_path experiment/345/2026-07-15/tasks/longbench_local \
    --batch_size 1 --limit $LIMIT \
    --model_args "pretrained=$M,is_feature_cache=False,is_cfg_cache=False,max_length=2048,\
student_path=$S,student_prompt_layer_split=True,student_budget=960,\
student_frozen_layers=16,student_measured_tokens=$meas,\
student_measured_baseline=$baseline,student_question_window=128" \
    --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
    --num_fewshot 0 --log_samples --apply_chat_template \
    --fewshot_as_multiturn --trust_remote_code \
    --output_path $OUT/$name \
    > logs/2wiki_${name}.log 2>&1
  echo "=== END $name $(date +%H:%M:%S)"
  grep -E "\|score\|" logs/2wiki_${name}.log | tail -2
}

run measured480_refresh refresh 480
run measured480_step    step    480
echo "2WIKI MEASURED BASELINE DONE $(date +%H:%M:%S)"
