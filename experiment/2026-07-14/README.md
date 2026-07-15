# MaskKV Reproduction Notes - 2026-07-14

This fork starts from `maomaocun/dLLM-Cache` because the official `jianuo-huang/MaskKV`
repository currently contains the paper assets and README but not the implementation.

## Implemented Scope

- Model: `/home/M2026107/dllm/model/LLaDA-8B-Instruct`
- Evaluation data: `/home/M2026107/dllm/data/longbench/{qasper,2wikimqa,qmsum}.jsonl`
- Evaluation tasks: `local_longbench_qasper`, `local_longbench_2wikimqa`, `local_longbench_qmsum`
- Cache method: dLLM-Cache feature reuse plus a MaskKV-style LLaDA attention wrapper.

## What Changed

- Cloned `maomaocun/dLLM-Cache` as the base because `jianuo-huang/MaskKV` has not released code.
- Added a MaskKV-style LLaDA attention wrapper in `dllm_cache/maskkv.py` and `dllm_cache/hooks/cache_hook_LLaDA_maskkv.py`.
- Added mask-position tracking in `utils/generate_function.py`.
- Added local LongBench task definitions under `experiment/2026-07-14/tasks/longbench_local`.
- Updated the LongBench runner to prefer `.venv/bin/accelerate` so it does not pick the system
  `accelerate` executable without `torch`.
- Moved the base repository demo scripts under `experiment/2026-07-14/demo`.
- Exported Qasper, 2WikiMultihopQA, and QMSum test splits as JSONL files under `/home/M2026107/dllm/data/longbench`.
- Copied the LLaDA model snapshot to `/home/M2026107/dllm/model/LLaDA-8B-Instruct`.
- Removed the project-specific shared Hugging Face cache entries after local data/model copies were ready.
- Collected smoke-test outputs under `experiment/2026-07-14/results`.

The wrapper uses mask-token queries from the current denoising state to score prompt
tokens, then applies per-head top-k prompt KV selection during attention. Layer and
head allocation are controlled by environment variables:

- `MASKKV_ENABLED=1`
- `MASKKV_BUDGET=256`
- `MASKKV_LAYER_BASE_RATE=1.0`
- `MASKKV_HEAD_BASE_RATE=0.2`

This is an initial reproduction scaffold, not the unreleased official implementation.
It wires Mask-Voting and head-aware prompt KV selection into the LLaDA attention path;
full peak-memory parity with the paper still requires the official cache storage
layout and offline calibration profile.

Current Mask-Voting coverage:

- Implemented: mask-query attention scoring with `softmax(Q_mask K^T / sqrt(d_k))`.
- Implemented: prompt-token importance by summing mask-query attention over mask tokens.
- Implemented: per-head prompt top-k KV selection from the importance scores.
- Partial: layer/head budget allocation. The current code uses online heuristic allocation
  controlled by `MASKKV_LAYER_BASE_RATE` and `MASKKV_HEAD_BASE_RATE`, not the paper's
  offline calibrated layer profile.
- Missing: the paper's offline cache-storage layout for peak-memory reduction.

## Run

Install dependencies in the environment you use for the 8B model, then run:

```bash
bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh
```

The runner uses the local JSONL files under `/home/M2026107/dllm/data` and the
local model snapshot under `/home/M2026107/dllm/model`. It does not use the
built-in Hugging Face LongBench task definitions. Runtime-only Hugging Face
processing caches default to `/tmp`, not the persistent `data` or `model`
directories.

Useful overrides:

```bash
MASKKV_BUDGET=128 OUTPUT_PATH=./longbench_maskkv_b128 \
  bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh
```

```bash
TASK_FILTER=2wikimqa MODEL_MAX_LENGTH=2048 MASKKV_BUDGET=128 \
  OUTPUT_PATH=experiment/2026-07-14/results/longbench_maskkv_b128_2wikimqa_full_mlen2048 \
  bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh
```

The script defaults to `accelerate_config_single_gpu.yaml` for the current
single-A100 workspace. To use the original multi-GPU config, set
`ACCELERATE_CONFIG=accelerate_config.yaml`.

The paper reports LongBench generation lengths of 128 for Qasper, 32 for
2WikiMultihopQA, and 512 for QMSum, so the script runs each task separately.
Use `TASK_FILTER=qasper`, `TASK_FILTER=2wikimqa`, or `TASK_FILTER=qmsum` to run
one task. `MODEL_MAX_LENGTH` is optional; it sets the LLaDA wrapper's token
limit before generation length is added.

## Demo Scripts

The original base-repo demos are archived in `experiment/2026-07-14/demo`.
They are kept separate from the reproduction path because they are interactive
or model-specific examples, not the LongBench reproduction driver. The demo
folder includes `demo_bootstrap.py`; each moved demo imports it before the
original repository imports so direct execution can still find repository
packages.

## Validation

- `lm_eval validate --include_path experiment/2026-07-14/tasks/longbench_local --tasks local_longbench_qasper,local_longbench_2wikimqa,local_longbench_qmsum`
- Qasper 1-sample smoke run using `/home/M2026107/dllm/model/LLaDA-8B-Instruct` and `/home/M2026107/dllm/data/longbench/qasper.jsonl`
- Budget-128 path check:
  `MASKKV_BUDGET=128 LIMIT=1 OUTPUT_PATH=experiment/2026-07-14/results/longbench_maskkv_b128_limit1 bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh`
- Baseline cache comparison:
  `MASKKV_ENABLED=0 LIMIT=1 OUTPUT_PATH=experiment/2026-07-14/results/longbench_baseline_cache_limit1 bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh`

### Budget-128 LIMIT=1 Results

These numbers are a pipeline sanity check, not statistically meaningful LongBench scores.
The full 200-sample run is required for paper-style reporting.

| Task | Gen length | MaskKV B=128 | Baseline cache |
| --- | ---: | ---: | ---: |
| Qasper | 128 | 0.04598 F1 | 0.07059 F1 |
| 2WikiMultihopQA | 32 | 0.00000 F1 | 0.13333 F1 |
| QMSum | 512 | 0.00000 Rouge-L | 0.10000 Rouge-L |

Observed sample quality under `MASKKV_BUDGET=128` is degraded versus the baseline 1-sample
run, especially for 2WikiMultihopQA and QMSum. Treat the current MaskKV path as a scaffold
until the offline allocation and cache-storage details are matched.

### 2WikiMultihopQA Budget-128 Full Run

Command:

```bash
TASK_FILTER=2wikimqa MODEL_MAX_LENGTH=2048 MASKKV_BUDGET=128 \
  OUTPUT_PATH=experiment/2026-07-14/results/longbench_maskkv_b128_2wikimqa_full_mlen2048 \
  bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh
```

Result:

| Task | Samples | Gen length | Model max length | MaskKV budget | F1 | Stderr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2WikiMultihopQA | 200 | 32 | 2048 | 128 | 0.10009 | 0.01252 |

Full-cache baseline under the same single-GPU truncation condition:

```bash
TASK_FILTER=2wikimqa MODEL_MAX_LENGTH=2048 MASKKV_ENABLED=0 \
  OUTPUT_PATH=experiment/2026-07-14/results/longbench_fullcache_2wikimqa_full_mlen2048 \
  bash experiment/2026-07-14/scripts/run_LLaDA_maskkv_longbench_Instruct.sh
```

| Run | Samples | Gen length | Model max length | F1 | Stderr | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full cache | 200 | 32 | 2048 | 0.12161 | 0.01029 | 487.90s |
| MaskKV B=128 | 200 | 32 | 2048 | 0.10009 | 0.01252 | 540.45s |

Output files:

- `experiment/2026-07-14/results/longbench_maskkv_b128_2wikimqa_full_mlen2048/__home__M2026107__dllm__model__LLaDA-8B-Instruct/results_2026-07-14T16-26-45.240786.json`
- `experiment/2026-07-14/results/longbench_maskkv_b128_2wikimqa_full_mlen2048/__home__M2026107__dllm__model__LLaDA-8B-Instruct/samples_local_longbench_2wikimqa_2026-07-14T16-26-45.240786.jsonl`
- `experiment/2026-07-14/results/longbench_fullcache_2wikimqa_full_mlen2048/__home__M2026107__dllm__model__LLaDA-8B-Instruct/results_2026-07-14T16-53-08.706986.json`
- `experiment/2026-07-14/results/longbench_fullcache_2wikimqa_full_mlen2048/__home__M2026107__dllm__model__LLaDA-8B-Instruct/samples_local_longbench_2wikimqa_2026-07-14T16-53-08.706986.jsonl`

Notes:

- LLaDA config reports `max_position_embeddings=4096`, but the current scaffold OOMs on
  A100 40GB at `MODEL_MAX_LENGTH=4096` during the full 2Wiki run. The failure occurred
  after 32/200 requests with about 36 GiB already allocated before the final logits
  projection.
- `MODEL_MAX_LENGTH=2048` keeps prompt length at `2048 - 32 = 2016` tokens for this task.
  This is a practical single-GPU scaffold setting, not a paper-identical long-context
  condition.
- Truncation is a likely contributor to the low baseline. The 2Wiki prompts average
  about 7.7k tokens; `MODEL_MAX_LENGTH=2048` keeps about 32.7% of tokens on average,
  while `MODEL_MAX_LENGTH=4096` keeps about 60.5%. A 40-sample full-cache ablation
  improved from 0.12117 F1 at 2048 to 0.13090 F1 at 4096, so longer context helps
  but does not fully explain the gap.

## Revealed-Answer Teacher Experiment

Added an oracle teacher path for the Q-ViK-style follow-up experiment. Instead of
using current mask queries, this teacher appends the gold answer to the prompt and
records how answer-token queries attend to prompt-token keys:

```text
score(prompt_i) = mean_heads sum_answer softmax(Q_answer K_prompt^T / sqrt(d))_i
```

This is a teacher-forced/oracle trajectory, not the model's actual generated
trajectory. It is intended as a stronger label for a student MLP that must predict
future-important prompt tokens from prompt-only hidden states.

Teacher extraction:

```bash
N_SAMPLES=0 MAX_LENGTH=2048 \
  OUTPUT_ROOT=experiment/2026-07-14/results/revealed_answer_teacher \
  bash experiment/2026-07-14/scripts/run_revealed_answer_teacher_2wikimqa.sh
```

Teacher dataset:

| Field | Value |
| --- | --- |
| Source | `framolfese/2WikiMultihopQA` |
| Split | `train` |
| Rows | 167,454 |
| Local file | `/home/M2026107/dllm/data/train/2wikimultihopqa/2wikimultihopqa_train_longbench_format.jsonl` |
| Eval overlap check | 0 overlapping `_id`s with `/home/M2026107/dllm/data/longbench/2wikimqa.jsonl` |

The first revealed-answer smoke run used one sample from
`/home/M2026107/dllm/data/longbench/2wikimqa.jsonl` only to verify the hook and
serialization path. Those eval-based smoke artifacts were removed. Use the
train-based teacher outputs below for student training.

Student training:

```bash
TEACHER_ROOT=experiment/2026-07-14/results/revealed_answer_teacher \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_student \
  EPOCHS=1 \
  bash experiment/2026-07-14/scripts/run_revealed_answer_student.sh
```

Train-based smoke QA completed with `N_SAMPLES=1 MAX_LENGTH=256`:

| Stage | Output | Observed |
| --- | --- | --- |
| Teacher extraction | `experiment/2026-07-14/results/revealed_answer_teacher_train_smoke/2wikimultihopqa_train/*.pt` | `prompt=255`, `answer=1`, shard saved from train split |
| Student training | `experiment/2026-07-14/results/revealed_answer_student_train_smoke/checkpoint-last` | 1-sample train completed, loss `0.169421` |

First real student tranche:

```bash
N_SAMPLES=1000 MAX_LENGTH=2048 \
  OUTPUT_ROOT=experiment/2026-07-14/results/revealed_answer_teacher_train_n1000 \
  bash experiment/2026-07-14/scripts/run_revealed_answer_teacher_2wikimqa.sh

.venv/bin/python experiment/2026-07-14/revealed_answer/train_student.py \
  --teacher-root experiment/2026-07-14/results/revealed_answer_teacher_train_n1000 \
  --output-dir experiment/2026-07-14/results/revealed_answer_student_train_n1000_e10_lr5e-5 \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --datasets 2wikimultihopqa_train \
  --val-ratio 0.1 \
  --epochs 10 \
  --lr 5e-5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --proj-dim 256 \
  --mlp-dim 512
```

The initial `lr=1e-4` check improved through epoch 2 but validation worsened by
epoch 3, so the retained run uses `lr=5e-5`. The student is an MLP, but each
training sample still recomputes frozen LLaDA hidden states, so epoch count is
not free. For this 1000-sample tranche, 10 epochs was still practical and kept
improving validation loss.

| Setting | Value |
| --- | --- |
| Teacher shards | 1000 train-split examples |
| Student split | 900 train / 100 validation |
| Epochs | 10 |
| Learning rate | `5e-5` |
| Output checkpoint | `experiment/2026-07-14/results/revealed_answer_student_train_n1000_e10_lr5e-5/checkpoint-last` |
| Final train loss | 0.143761 |
| Final / best val loss | 0.143931 at epoch 10 |
| Elapsed | 1669.3s |

Second student tranche:

```bash
N_SAMPLES=5000 MAX_LENGTH=2048 \
  OUTPUT_ROOT=experiment/2026-07-14/results/revealed_answer_teacher_train_n5000 \
  bash experiment/2026-07-14/scripts/run_revealed_answer_teacher_2wikimqa.sh

.venv/bin/python experiment/2026-07-14/revealed_answer/train_student.py \
  --teacher-root experiment/2026-07-14/results/revealed_answer_teacher_train_n5000 \
  --output-dir experiment/2026-07-14/results/revealed_answer_student_train_n5000_e30_lr5e-5 \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --datasets 2wikimultihopqa_train \
  --val-ratio 0.1 \
  --epochs 30 \
  --lr 5e-5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --proj-dim 256 \
  --mlp-dim 512 \
  --log-every 100
```

The 5000-shard run was started with a 30-epoch cap but stopped manually after
epoch 9 because the validation improvement had flattened below `1e-5` per
epoch. `checkpoint-best` is saved whenever validation improves, so the
interrupted epoch 10 did not overwrite the retained best model.

| Setting | Value |
| --- | --- |
| Teacher shards | 5000 train-split examples |
| Student split | 4500 train / 500 validation |
| Epochs completed | 9, then stopped for plateau |
| Learning rate | `5e-5` |
| Output checkpoint | `experiment/2026-07-14/results/revealed_answer_student_train_n5000_e30_lr5e-5/checkpoint-best` |
| Final / best train loss | 0.143793 at epoch 9 |
| Final / best val loss | 0.143723 at epoch 9 |
| Elapsed to epoch 9 | 7608.9s |

Compared with the first 1000-shard tranche, scaling teacher shards helped more
than simply extending epochs: the 5000-shard validation curve still improved
through epoch 9, but the final improvements were tiny, so continuing to the
planned 30 epochs was not worth the compute for this checkpoint.

Oracle teacher-score pruning eval:

```bash
LIMIT=20 MAX_LENGTH=2048 BUDGET=128 \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_oracle_2wikimqa_b128_limit20 \
  bash experiment/2026-07-14/scripts/run_revealed_answer_oracle_2wikimqa.sh
```

This eval uses the gold-answer revealed-attention score directly as the pruning
oracle during generation. It is not a trained student result. The first-20
comparison reuses the existing full-cache and MaskKV 200-sample outputs and
takes their first 20 rows to match the oracle run:

| Method | Samples | Budget | 2Wiki F1 |
| --- | ---: | ---: | ---: |
| Full cache | first 20 of 200 | full | 0.109925 |
| Existing MaskKV | first 20 of 200 | 128 | 0.070568 |
| Revealed-answer teacher oracle | 20 | 128 | 0.144814 |

This is enough to treat the teacher score as a plausible target for the student
MLP: on the same first 20 eval examples, the oracle-pruned run is close to and
slightly above the full-cache baseline despite keeping only 128 prompt KV
positions per layer.

Student-score pruning eval:

```bash
LIMIT=20 MAX_LENGTH=2048 BUDGET=128 \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_student_2wikimqa_b128_limit20 \
  bash experiment/2026-07-14/scripts/run_revealed_answer_student_prune_2wikimqa.sh
```

This eval uses prompt-only hidden states, predicts layer-wise prompt token
importance with the trained MLP, and then uses the same B=128 pruning hook as
the oracle eval.

| Method | Samples | Budget | 2Wiki F1 |
| --- | ---: | ---: | ---: |
| Full cache | first 20 of 200 | full | 0.109925 |
| Existing MaskKV | first 20 of 200 | 128 | 0.070568 |
| Revealed-answer teacher oracle | 20 | 128 | 0.144814 |
| Revealed-answer student, 1000 teacher shards | 20 | 128 | 0.093905 |
| Revealed-answer student, 5000 teacher shards, epoch-9 best | 20 | 128 | 0.116439 |

The 5000-shard student now beats both the existing B=128 MaskKV first-20 run and
the full-cache first-20 baseline, but it remains below the revealed-answer oracle
upper bound on this quick subset.

Full 200-sample pruning evals:

```bash
LIMIT=0 MAX_LENGTH=2048 BUDGET=128 \
  STUDENT_PATH=experiment/2026-07-14/results/revealed_answer_student_train_n5000_e30_lr5e-5/checkpoint-best \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_student_n5000_e9best_2wikimqa_b128_full \
  bash experiment/2026-07-14/scripts/run_revealed_answer_student_prune_2wikimqa.sh

LIMIT=0 MAX_LENGTH=2048 BUDGET=128 \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_oracle_2wikimqa_b128_full \
  bash experiment/2026-07-14/scripts/run_revealed_answer_oracle_2wikimqa.sh
```

| Method | Samples | Budget | 2Wiki F1 | Eval time |
| --- | ---: | ---: | ---: | ---: |
| Full cache | 200 | full | 0.12161 | 487.90s |
| Existing MaskKV | 200 | 128 | 0.10009 | 540.45s |
| Revealed-answer student, 5000 teacher shards, epoch-9 best | 200 | 128 | 0.09638 | 1366.82s |
| Revealed-answer teacher oracle | 200 | 128 | 0.20849 | 1370.42s |

The first-20 subset was optimistic: the full 200-sample score is below the
existing B=128 MaskKV full run and well below full cache. However, the full
oracle score is much higher than full cache, so the teacher signal itself has a
large upper-bound margin. At this stage the bottleneck is not the oracle target,
but how well the prompt-only student reproduces the teacher's budgeted token
set during generation.

Top-k imitation loss follow-up:

```bash
.venv/bin/python experiment/2026-07-14/revealed_answer/train_student.py \
  --teacher-root experiment/2026-07-14/results/revealed_answer_teacher_train_n5000 \
  --output-dir experiment/2026-07-14/results/revealed_answer_student_train_n5000_topk128_e5_lr5e-5_tw0.02 \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --datasets 2wikimultihopqa_train \
  --n-samples 0 \
  --val-ratio 0.1 \
  --epochs 5 \
  --lr 5e-5 \
  --rank-weight 0.1 \
  --topk-weight 0.02 \
  --topk-k 128 \
  --topk-positive-weight 8.0 \
  --device cuda:0 \
  --dtype bfloat16 \
  --proj-dim 256 \
  --mlp-dim 512 \
  --log-every 100
```

This keeps the old MSE/rank objective and adds a small BCE term on the teacher
top-128 prompt-token set. The best checkpoint was epoch 4
(`val_loss=0.373573`; epoch 5 rose to `0.376565`).

Offline teacher imitation:

| Student | Split | Samples | Recall@128 | NDCG@128 | Teacher mass@128 |
| --- | --- | ---: | ---: | ---: | ---: |
| 5000 shards, epoch-9 best | train-val | 500 | 0.75738 | 0.96303 | 0.71519 |
| top-k loss, epoch-4 best | train-val | 500 | 0.83382 | 0.79919 | 0.72959 |
| 5000 shards, epoch-9 best | eval 2Wiki | 200 | 0.71388 | 0.93803 | 0.62890 |
| top-k loss, epoch-4 best | eval 2Wiki | 200 | 0.76441 | 0.77052 | 0.64088 |

B=128 generation eval:

```bash
LIMIT=0 MAX_LENGTH=2048 BUDGET=128 \
  STUDENT_PATH=experiment/2026-07-14/results/revealed_answer_student_train_n5000_topk128_e5_lr5e-5_tw0.02/checkpoint-best \
  OUTPUT_DIR=experiment/2026-07-14/results/revealed_answer_student_n5000_topk128_e5_2wikimqa_b128_full \
  bash experiment/2026-07-14/scripts/run_revealed_answer_student_prune_2wikimqa.sh
```

| Method | Samples | Budget | 2Wiki F1 | Eval time |
| --- | ---: | ---: | ---: | ---: |
| Revealed-answer student, 5000 teacher shards, epoch-9 best | 200 | 128 | 0.09638 | 1366.82s |
| Top-k loss student, 5000 teacher shards, epoch-4 best | 200 | 128 | 0.12584 | 1368.01s |
| Full cache | 200 | full | 0.12161 | 487.90s |
| Revealed-answer teacher oracle | 200 | 128 | 0.20849 | 1370.42s |

The top-k term improved held-out Recall@128 by about 5 points and recovered the
generation score above the full-cache baseline, but it is still far below the
teacher oracle. The remaining gap is likely not just top-128 overlap; the
student still has poor ranking calibration inside the selected budget, and the
teacher labels are computed from gold-answer hidden states while generation uses
student pruning before the answer trajectory exists.

Decode latency check, first 20 2Wiki samples:

```bash
.venv/bin/python experiment/2026-07-14/revealed_answer/measure_decode_latency_2wikimqa.py \
  --student experiment/2026-07-14/results/revealed_answer_student_train_n5000_topk128_e5_lr5e-5_tw0.02/checkpoint-best \
  --output-dir experiment/2026-07-14/results/decode_latency_memory_topk128_b128_limit20 \
  --limit 20 \
  --budget 128 \
  --max-length 2048 \
  --gen-length 32 \
  --steps 32 \
  --block-length 8 \
  --device cuda:0 \
  --dtype bfloat16
```

The timing excludes model load and tokenization, synchronizes CUDA around each
measured section, and uses the same prompt/generation settings for both methods.
Peak memory is reported as CUDA `max_memory_allocated` delta after model load.

| Method | Samples | Budget | Mean decode | Mean total method | Peak delta | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-cache dLLM | 20 | full | 6.6771s | 6.6771s | 1003.38 MiB | 0.11723 |
| Top-k loss student | 20 | 128 | 6.4970s | 6.7486s | 1077.55 MiB | 0.13012 |

Student score prediction added `0.2440s` mean per sample. The pruned decode
itself was about `0.1801s` faster on average, so end-to-end latency was slightly
slower than full-cache dLLM on this 20-sample run. Memory also did not improve:
the decode-only peak delta was similar (`1002.77 MiB` vs `1003.38 MiB`), while
the prompt-only student scoring forward raised the method peak to `1077.55 MiB`.

Estimated attention work under the current hook:

| Quantity | Full-cache dLLM | Top-k loss student | Relative |
| --- | ---: | ---: | ---: |
| Prompt KV kept | 100% | 6.52% mean | 93.48% prompt KV removed |
| Attention QK elements/sample | 4.172B mean | 0.330B mean | 91.98% estimated reduction |

This indicates the cache/pruning policy has a large theoretical attention-work
reduction, but the current Python/hook implementation and student-score forward
mostly consume that gain at this sequence length and batch size.

Precomputed prompt-KV suffix decode:

```bash
.venv/bin/python experiment/2026-07-14/revealed_answer/measure_decode_latency_2wikimqa.py \
  --student experiment/2026-07-14/results/revealed_answer_student_train_n5000_topk128_e5_lr5e-5_tw0.02/checkpoint-best \
  --output-dir experiment/2026-07-14/results/prompt_kv_cache_latency_b128_limit20 \
  --limit 20 \
  --budget 128 \
  --max-length 2048 \
  --gen-length 32 \
  --steps 32 \
  --block-length 8 \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt-kv-cache
```

This mode first predicts top-B prompt positions with the student, runs a
prompt-only pass to store the selected prompt K/V per layer, then decodes only
the generated suffix. It removes prompt-token Q/K/V projection, prompt-token
attention queries, prompt-token MLP, and prompt logits from each denoising step.
It keeps suffix K/V dynamic and attends suffix queries to cached prompt K/V plus
current suffix K/V.

Smoke check, first 2Wiki sample with `max_length=512`, `gen_length=8`,
`steps=8`, `budget=128`:

| Method | Full decode | Score | Cache build | Student decode | Student total | Decode peak delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Precomputed prompt-KV | 0.6597s | 0.0972s | 0.0808s | 0.2427s | 0.4250s | 18.07 MiB |

The cached suffix path is an intentional approximation for LLaDA: the original
model is bidirectional, so full-sequence prompt hidden states can change when
generated mask tokens are present. This implementation fixes prompt K/V from a
prompt-only pass to test the cache hypothesis directly.

Implementation files:

- `experiment/2026-07-14/revealed_answer/extract_teacher.py`
- `experiment/2026-07-14/revealed_answer/attention_teacher.py`
- `experiment/2026-07-14/revealed_answer/student_model.py`
- `experiment/2026-07-14/revealed_answer/train_student.py`
- `experiment/2026-07-14/revealed_answer/training_loop.py`
- `experiment/2026-07-14/revealed_answer/eval_student_imitation.py`
- `experiment/2026-07-14/revealed_answer/imitation_metrics.py`
- `experiment/2026-07-14/revealed_answer/measure_decode_latency_2wikimqa.py`
- `experiment/2026-07-14/revealed_answer/oracle_prune.py`
- `experiment/2026-07-14/revealed_answer/prompt_kv_cache.py`
- `experiment/2026-07-14/revealed_answer/prompt_kv_forward.py`
- `experiment/2026-07-14/revealed_answer/prompt_kv_generate.py`
- `experiment/2026-07-14/revealed_answer/eval_teacher_oracle_2wikimqa.py`
- `experiment/2026-07-14/revealed_answer/eval_student_prune_2wikimqa.py`
