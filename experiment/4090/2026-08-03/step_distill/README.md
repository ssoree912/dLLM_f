# State-conditioned B=960 distribution distillation

This directory keeps the frozen `2026-07-14` and `345/2026-07-15` experiments unchanged. The
current student experiment has one budget and one question:

\[
\boxed{B=960:\quad
p_{\mathrm{pruned}}(\cdot\mid x_t^\pi)
\approx p_{\mathrm{full}}(\cdot\mid x_t^\pi)}
\]

`B=480`, a budget embedding, MMR, and physical KV refresh are outside this experiment.

## Online teacher and causal state

The rollout state contains the unchanged prompt, committed suffix tokens, and remaining masks.
Both forwards receive the identical current pruned state `x_t^pi`:

\[
z_t^F=F(x_t^\pi;\mathbf 1_P),\qquad
z_t^\pi=F(x_t^\pi;M_t^\theta).
\]

The full pass runs without gradients. Both passes use the same explicit fp32 attention operator;
the full pass applies an all-one prompt gate and the pruned pass applies Top-960. Thus the prompt
gate is their only attention-path difference. The pruned pass keeps the decoder frozen but retains
the gradient to the selector. The next state always commits the pruned candidates, never teacher
tokens:

\[
x_{t+1}^\pi=\operatorname{Commit}(x_t^\pi,z_t^\pi).
\]

Prompt-only layer inputs provide fixed token features `r[l,i]`. Before each step, the causal state
summary uses only token IDs already known at that point:

\[
c_t=\operatorname{mean}_{j:x_{t,j}^\pi\ne[\mathrm{MASK}]}
E(x_{t,j}^\pi).
\]

At step zero, where that set is empty, `c_0` pools the final target dialogue/request span. It never
uses a partially retained fragment of the shared few-shot instruction. The shared selector is

\[
u_{l,i}=P_{tok}(r_{l,i}),\quad v_t=P_c(c_t),
\]

\[
a_{t,l,i}=\operatorname{MLP}
([u_{l,i};v_t;u_{l,i}\odot v_t;e_l]),\qquad
M_{t,l}^\theta=\operatorname{Top960}_i(a_{t,l,i}).
\]

There is no budget input because 960 is the only supported training budget.

## Straight-through hard pruning

An additive `-1e4` mask is not used: it gives discarded keys zero attention and therefore zero
selector gradient. Instead, attention probabilities are gated after softmax and renormalized. With

\[
m^{ST}=m^{soft}+\operatorname{stopgrad}(m^{hard}-m^{soft}),
\]

the training attention is

\[
\widetilde A_{q,i}=
\frac{A_{q,i}m_i^{ST}}{\sum_k A_{q,k}m_k^{ST}}.
\]

Its forward value is exactly the hard Top-960 attention result, while the backward relaxation also
reaches tokens outside the current Top-960. Training still computes dense QK scores to obtain this
surrogate gradient. Sparse latency or memory reduction is therefore measured later with the
integer-gather inference path, not with this training forward.

## Objective

For the uncommitted positions `U_t`, temperature `tau`, and full-teacher commit positions `S_t^F`:

\[
\mathcal L_{KD}=\frac{1}{|U_t|}\sum_{j\in U_t}
(1+\beta\mathbf 1[j\in S_t^F])\tau^2
D_{KL}(p^F_{t,j}\Vert p^\pi_{t,j}).
\]

Commit ranking compares `S_t^F` only with other uncommitted positions in the active generation
block, because future blocks are not eligible for the current decoder commit:

\[
\mathcal L=\mathcal L_{KD}+\lambda_{commit}\mathcal L_{commit}.
\]

The recorded per-step diagnostics separate the unweighted full-to-pruned KL from the
commit-weighted KD training loss, together with token top-1 agreement, commit-position Jaccard,
and selector gradient norm. A complete validation rollout additionally records both raw-canvas and
EOS/EOT-truncated predictions, ROUGE-L F1/LCS recall, first-stop position, tokens before stop, and
trailing-token count.

## SAMSum split and first run

Optimization uses only the prepared SAMSum train subset; validation is held out and never passed
to the optimizer. The default CLI uses 8 train and 8 validation samples, 128 denoising steps, and
`B=960`:

```bash
uv run --python /path/to/torch/python --with 'pydantic>=2,<3' \
  python -m step_distill.train_distribution_student \
  --model /path/to/LLaDA-8B-Instruct \
  --train-data /path/to/samsum_train_fewshot_2048.jsonl.xz \
  --validation-data /path/to/samsum_validation_fewshot_2048.jsonl.xz \
  --output-dir /path/to/distribution_b960
```

Use `--max-rollout-steps 1` only for an implementation smoke. Omitting it runs the complete
trajectory and enables downstream summary metrics.

## Offline attention-teacher diagnostics

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

The earlier attention diagnostic uses plain per-step relevance. For every denoising step `t`, the full
teacher forms

```text
s[t,l,i] = sum(j in S_t) normalized_confidence[t,j] * attention[t,l,j->i]
```

and stores `top_order[t,l] = argsort_i(s[t,l,i])`. There is no MMR or cosine-redundancy term. These
orders remain offline/online attention baselines; they are not labels for the distribution student.

This artifact is an **offline full-context per-step attention proxy**. It is suitable for temporal
diagnostics on the recorded trajectory. The online two-pass distribution teacher above removes its
stale-trajectory limitation.

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
