# State-conditioned per-step teacher

This directory implements the oracle-validity stage of the step-distillation experiment without
changing the frozen `2026-07-14` or `345/2026-07-15` experiments.

## What changed

The existing full-dynamic teacher observes every denoising step but collapses the complete
trajectory to one `[layer, prompt]` score. This implementation retains the step axis:

```text
context_pre      [step, layer, hidden]
commit_positions [step, max_commit]
top_order        [step, layer, max_target_k]
candidate_scores [step, layer, max_target_k]
```

Each target is computed from the pre-commit attention rows of the positions selected at that step.
When confidence weighting is enabled, confidence is normalized inside the step rather than over
the full trajectory. Step zero uses the exact tokenized question span; later steps pool only suffix
positions that were committed before the current forward.

The main distillation path is plain per-step relevance. For every denoising step `t`, the full
teacher forms

```text
s[t,l,i] = sum(j in S_t) normalized_confidence[t,j] * attention[t,l,j->i]
```

and stores `top_order[t,l] = argsort_i(s[t,l,i])`. The student input is the causal
`context_pre[t,l]`; its positive target at budget `B` is exactly `top_order[t,l,:B]`. There is no
MMR or cosine-redundancy term. At correctness-path inference the student predicts a new layer-wise
plain Top-B mask at every denoising step. Physical KV refresh and its interval `R` are a later cache
optimization and do not change this per-step learning target.

This artifact is an **offline full-context per-step attention proxy**. It is suitable for temporal
diagnostics and student targets on the recorded trajectory. It is not the final adaptive ceiling
when pruning changes the generated trajectory; that requires the planned online two-pass replay.

## Evaluation output boundary

LLaDA fills the complete generation canvas even when an EOS or chat end-of-turn token occurs before
the last position. Decoding the complete tensor with `skip_special_tokens=True` hides the stop token
but preserves ordinary tokens after it. Every new evaluator must therefore call
`generation_output.decode_generation` before computing its primary LongBench metric.

The stop set contains both the tokenizer EOS ID (`<|endoftext|>`) and `<|eot_id|>`. Each sample must
record the EOS-truncated prediction and score together with the raw-canvas prediction and score,
first stop position, tokens before the stop, and trailing token count. Raw LongBench F1 remains a
comparability field; the EOS-truncated F1 is the correctness score used by the oracle gate.

The first budget sweep uses the same prompt serialization, truncation, generation canvas, and stop
policy for all methods, in this order:

```text
full
random:       B=1024,512,256,128
aggregate:    B=1024,512,256,128
per-step:     B=1024,512,256,128
```

Output length, first-stop position, and trailing-token count diagnose whether a score change comes
from answer correctness or termination behavior. A pruning-quality claim additionally requires the
learned selector to beat the matched random-budget baseline.

## Data boundary

Teacher extraction consumes the original `framolfese/2WikiMultihopQA` training split represented by
the local LongBench-compatible JSONL. The LongBench 200-example test file is evaluation-only.

The extractor requires paths through CLI options. It does not contain machine-specific `/home` or
`/mnt` defaults. Output filenames include a sample-ID hash, and resume loads and validates the full
schema and metadata instead of checking file existence alone.

## CPU verification

From the repository root:

```bash
uv run --python /path/to/torch/python --with pytest \
  python -m pytest experiment/4090/2026-08-03/tests -q
```

The tests use a LLaDA-shaped fake model and do not load the 8B checkpoint.

## GPU smoke extraction

```bash
MODEL_PATH=/path/to/LLaDA-8B-Instruct \
DATA_PATH=/path/to/2wikimultihopqa_train_longbench_format.jsonl \
OUTPUT_ROOT=/path/to/teacher_gate_smoke \
SAMPLE_LIMIT=2 \
bash experiment/4090/2026-08-03/step_distill/scripts/extract_2wikimqa_teacher.sh
```

After the two-sample O1 trajectory-consistency check, rerun with `SAMPLE_LIMIT=64` and diagnose:

```bash
INPUT_ROOT=/path/to/teacher_gate_64 \
BUDGET=128 \
bash experiment/4090/2026-08-03/step_distill/scripts/diagnose_2wikimqa_oracle.sh
```

Student extraction or training must not begin until the dynamic oracle passes the downstream Gate A
against both full context and the existing trajectory-aggregate teacher.

## 2026-08-03 offline-replay gate results

These are correctness-path results on the same training rows used for teacher extraction, not
held-out LongBench scores and not latency measurements. `full` reproduced every stored teacher
trajectory token-for-token. Static replays step-zero order at every step; dynamic replays the
stored plain order for the current step.

### 2WikiMultihopQA train, 32 rows

Artifacts:

- teacher: `/home/M2026107/.cache/step_distill_teacher_gate_n32_20260803`
- inference: `/home/M2026107/.cache/step_distill_oracle_inference_n32_20260803`

| Method | B | Answer recall | Full recall retention |
|---|---:|---:|---:|
| full | full | 0.8932 | 1.0000 |
| static | 512 | 0.7812 | 0.8746 |
| dynamic | 512 | 0.7708 | 0.8630 |
| static | 256 | 0.6771 | 0.7580 |
| dynamic | 256 | 0.7188 | 0.8047 |
| static | 128 | 0.5625 | 0.6297 |
| dynamic | 128 | 0.5729 | 0.6414 |

Dynamic improves over static by 4.17 recall points at B=256 and 1.04 points at B=128, but loses
1.04 points at B=512. The effect is not yet a robust Gate A pass.

### SAMSum train, 32 rows

SAMSum uses its existing LongBench-format train prompt, `gen_length=128`, `steps=128`, and
`block_length=32`. `K_max=64` matches the largest evaluated budget. ROUGE-L below is word-level
LCS F1 after common EOS/EOT truncation.

Artifacts:

- teacher: `/home/M2026107/.cache/step_distill_teacher_samsum_n32_k64_20260803`
- inference: `/home/M2026107/.cache/step_distill_oracle_samsum_n32_k64_20260803`

| Method | B | ROUGE-L | Full ROUGE-L retention |
|---|---:|---:|---:|
| full | full | 0.2794 | 1.0000 |
| static | 64 | 0.0944 | 0.3379 |
| dynamic | 64 | 0.1099 | 0.3934 |
| static | 32 | 0.0281 | 0.1007 |
| dynamic | 32 | 0.0398 | 0.1424 |
| static | 16 | 0.0000 | 0.0000 |
| dynamic | 16 | 0.0110 | 0.0395 |

Top-order early/late Jaccard is 0.5351/0.3727/0.3166 at B=64/32/16, so the target changes over
time. Nevertheless, the prompt budget is below the oracle ceiling needed to preserve summary
quality: even dynamic B=64 retains only 39.34% of full ROUGE-L. This setting fails Gate A despite
showing a small dynamic-over-static improvement.
