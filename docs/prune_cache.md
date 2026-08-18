# Prune-cache scorer

The refactored path is intentionally fixed rather than exposing the old
experiment matrix.

## Teacher and student

- Prompts are tokenized without a chat template.
- One denoising trajectory extracts both targets.
- The Attention target is dense (`active_top_k=0`), confidence weighted, and
  max-aggregated over commit steps.
- The Delta target is cumulative stepwise prompt K/V movement.
- One student shares its token/question projections and has separate
  `attention` and `delta` heads. Both heads train in the same optimizer step.

Build both targets and the joint scorer with:

```bash
scripts/build_prune_cache_scorer.sh \
  SOURCE_ROOT TEACHER_ROOT STUDENT_OUT DATASET [DATASET ...]
```

The source shards must contain no-chat `prompt_input_ids`.

## Inference policy

`prune_cache_path` is the only scorer/cache option. It loads the same joint
checkpoint for both rankings.

1. Reserve the task's official generation length from the 2048-token context.
2. Let `P` be the actual prompt length after truncation.
3. Keep `ceil(P / 2)` tokens ranked by the Attention head.
4. From that kept set, choose `ceil(kept / 2)` tokens with the Delta head.
5. Freeze that update set after its first selection and refresh those same
   tokens throughout denoising.

| Task gen length | Maximum prompt | Kept | Updated |
|---:|---:|---:|---:|
| 32 | 2016 | 1008 | 504 |
| 64 | 1984 | 992 | 496 |
| 128 | 1920 | 960 | 480 |
| 512 | 1536 | 768 | 384 |

Run a no-chat LongBench evaluation with:

```bash
scripts/eval_prune_cache.sh CHECKPOINT OUTPUT_DIR [LONGBENCH_TASK]
```

The task's `max_gen_toks` supplies `gen_length` and `steps`; no per-dataset
budget configuration is required.
