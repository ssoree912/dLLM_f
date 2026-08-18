# DLPC KV pruning: teacher, scorer, inference workflow

Branch: `dev/dlpc_kv_pruning`

이 문서는 현재 구현 기준으로 teacher score가 무엇을 의미하는지, 어떤 코드 경로에서 만들어지고 쓰이는지, 그리고 지금까지 확인한 성능을 기록한다.

## 1. 현재 추론 정책

현재 정책은 prompt token을 두 단계로 나눈다.

```text
full prompt
  -> attention scorer top 960
  -> 나머지 prompt token 제거
  -> delta scorer top 480 inside kept 960
  -> 같은 480개 prompt K/V를 매 denoising step refresh
```

| 값 | 의미 | 적용 위치 |
|---:|---|---|
| 128 | teacher 추출 중 step별 attention 후보 수 | teacher extraction only |
| 960 | 추론에서 남길 prompt token 수 | inference keep budget |
| 480 | 추론에서 매 step K/V를 갱신할 token 수 | inference refresh budget |

중요한 점: student 학습 loss에는 960/480 budget을 넣지 않는다. Student는 연속 score와 ranking만 학습하고, budget은 추론 시점의 `topk`로만 적용한다.

## 2. Attention teacher

Teacher 추출은 매 denoising step마다 전체 sequence `[prompt, generation canvas]`를 forward한다.

- prompt 위치: `p`
- denoising step: `t`
- layer: `l`
- head: `h`
- 이번 step에 새로 확정되는 생성 token 집합: `C_t`

생성 suffix token `j`가 prompt token `p`를 보는 attention은 head 평균으로 계산한다.

```text
alpha[t,l,j,p] =
  mean_h softmax((Q[t,l,h,j] K[t,l,h,p]^T) / sqrt(d))
```

이번 step에서 commit되는 token들이 prompt를 참조한 양:

```text
a[t,l,p] = sum_{j in C_t} w[t,j] * alpha[t,l,j,p]
```

여기서 `w[t,j]`는 confidence weight이다. 현재 teacher는 `--confidence-weight`를 사용한다.

현재 scorer용 teacher는 `active_top_k=0`으로 추출한다. 이 값은 temporal union mask를 끄고 모든 prompt token의 연속 score를 보존한다는 의미다.

```text
A[l,p] = max_t a[t,l,p]
```

최종 attention raw score는 `target_aggregation=max` 기준이다.

```text
teacher_raw[l,p] = A[l,p]
```

그리고 layer별 prompt 방향 합이 1이 되도록 정규화한다.

```text
teacher_norm[l,p] = A[l,p] / (sum_p A[l,p] + eps)
```

Attention student는 `target_mode=score`로 `teacher_norm`을 학습한다.

현재 category workflow는 attention scorer도 chat-template teacher에서 학습한다. 같은 shard의 `teacher_norm`을 attention target으로 쓴다.

## 3. Cumulative K/V delta teacher

Delta teacher는 attention이 아니라 prompt token의 K/V가 denoising trajectory 중 얼마나 변했는지 측정한다. 같은 full-sequence forward 안에서 prompt K/V를 관찰한다.

Stepwise K 이동량:

```text
dK[t,l,p] =
  mean_h ||K[t,l,h,p] - K[t-1,l,h,p]||_2
  / (mean_h ||K[t,l,h,p]||_2 + eps)
```

Stepwise V 이동량:

```text
dV[t,l,p] =
  mean_h ||V[t,l,h,p] - V[t-1,l,h,p]||_2
  / (mean_h ||V[t,l,h,p]||_2 + eps)
```

K/V 평균:

```text
delta_step[t,l,p] = (dK[t,l,p] + dV[t,l,p]) / 2
```

전체 trajectory 누적:

```text
D[l,p] = sum_{t=2..T} delta_step[t,l,p]
```

정규화:

```text
delta_norm[l,p] = D[l,p] / (sum_p D[l,p] + eps)
```

Delta student는 `target_mode=delta`로 `delta_norm`을 학습한다.

현재 category workflow는 delta scorer도 같은 chat-template teacher shard에서 학습한다. 같은 shard의 `delta_norm`을 delta target으로 쓴다.

## 4. Student 학습

Attention student와 Delta student는 같은 `PromptUtilityStudent` 구조를 사용하지만 target이 다르다.

| student | teacher root | target mode | target tensor |
|---|---|---|---|
| attention scorer | shared chat-template teacher | `score` | `teacher_norm` |
| delta scorer | shared chat-template teacher | `delta` | `delta_norm` |

현재 loss:

```text
loss = MSE(softmax(student_score), target)
       + 0.1 * rank_loss
       + 0.0 * topk_loss
```

Ranking loss는 teacher 상위 20%와 하위 40% 사이에 margin `0.05`를 둔다.

```text
rank_loss = mean_{i in top20%, j in bottom40%} max(0, pred[j] - pred[i] + 0.05)
```

따라서 학습은 budget-free이다. `topk_k=128` 같은 고정 budget 보조 loss는 현재 사용하지 않는다.

## 5. 추론 코드 경로

| 단계 | 코드 |
|---|---|
| teacher CLI | `dllm_cache/budget/extract_offline_hybrid_teacher.py` |
| transformer block discovery / answer-attention compatibility collector | `dllm_cache/budget/attention_teacher.py` |
| attention/delta collector | `dllm_cache/budget/offline_hybrid_teacher.py` |
| train CLI | `dllm_cache/budget/train_student.py` |
| teacher target 선택 | `dllm_cache/budget/training_loop.py` |
| student model/loss | `dllm_cache/budget/student_model.py` |
| LLaDA eval wrapper | `eval_model/LLaDA.py` |
| keep 960 + refresh 480 generation | `dllm_cache/budget/drift_refresh_kv.py` |

주요 함수:

- `generate_with_offline_hybrid_teacher()`: full trajectory를 돌면서 attention/delta teacher를 같이 저장한다.
- `accumulate_reference()`: commit token 기준 suffix-to-prompt attention을 누적한다. `active_top_k=0`이면 temporal union mask 없이 모든 prompt token score를 보존한다.
- `normalized_prompt_movement()`: prompt K/V relative movement를 계산한다.
- `teacher_targets_from_record()`: `target_mode=score`는 `teacher_norm`, `target_mode=delta`는 `delta_norm`을 선택한다.
- `generate_with_drift_refresh()`: attention top-960 keep set을 만들고, delta scorer로 refresh 480개를 선택해서 매 step update한다.
- `select_delta_student_topk()`: delta scorer의 layer 평균 score로 refresh token top-k를 고른다.

## 6. Category 500 workflow

새 workflow script:

```text
experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh
```

실행 순서:

```bash
# 설정 확인
bash experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh plan

# teacher 추출: shared chat-template root 하나에 teacher_norm/delta_norm 동시 저장
bash experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh extract

# category별 attention/delta scorer 20epoch 학습
bash experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh train

# category별 LongBench eval
bash experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh eval

# 결과 요약
bash experiment/2026-08-17/run_dlpc_kv_pruning_category500.sh summarize
```

Python은 기본적으로 현재 worktree `.venv`가 있으면 그것을 쓰고, 없으면 `/home/M2026107/dllm/dLLM-Cache/.venv/bin/python`을 쓴다. 다른 interpreter를 쓰려면 `DLPC_PYTHON=/path/to/python`으로 지정한다.

기본 output:

| artifact | path |
|---|---|
| shared chat teacher | `/home/M2026107/.cache/dlpc_kv_pruning_teacher500_delta_chat_20260817` |
| students | `results/budget/dlpc_kv_pruning_category500_20260817` |
| eval results | `results/dlpc_kv_pruning_category500_b960_r480_20260817` |
| logs | `logs/dlpc_kv_pruning_category500_20260817` |

Category mapping:

| category | train datasets | eval tasks |
|---|---|---|
| single_doc_qa | `qasper`, `narrativeqa` | `qasper`, `narrativeqa`, `multifieldqa_en` |
| multi_doc_qa | `2wikimultihopqa_train`, `hotpotqa`, `musique` | `2wikimqa`, `hotpotqa`, `musique` |
| summarization | `gov_report`, `multi_news`, `qmsum` | `gov_report`, `multi_news`, `qmsum` |
| few_shot | `samsum`, `trec`, `triviaqa` | `trec`, `triviaqa`, `samsum` |
| synthetic | none found | `passage_count`, `passage_retrieval_en` |
| code | `repobench-p` | `lcc`, `repobench-p` |

`synthetic`은 현재 train source가 없어서 scorer 학습/eval을 skip한다. `lcc`도 train file은 없지만 같은 code category의 `repobench-p` scorer로 평가한다.

Eval은 기본적으로 chat-template을 적용한다. 단, `trec`은 이전 관찰대로 no-chat이 유리해서 `--apply_chat_template` 없이 실행한다.

Eval task name은 `TASK_ROOT/<task>.yaml`의 `task:` 필드를 읽어서 사용한다. 현재 `experiment/345/2026-08-13/tasks/longbench_full_local`은 `longbench_qasper`, `longbench_2wikimqa` 같은 이름을 쓴다.

## 7. 현재까지 확인한 성능

현재 비교 기준은 모두 `student_budget=960`, `student_refresh_tokens=480`, `delta_select_once=True`, `refresh_interval=1`, `block_length=32`이다.

| run | attention scorer | delta scorer | eval prompt | samples | score |
|---|---|---|---|---:|---:|
| `samsum_maskuntil_attention_cumdelta_b960_r480_once_20260816` | mask-until attention | cumulative delta chat | chat | 200 | 0.3174 ± 0.0132 |
| `samsum_nochat_attention_scoreonly_cumdelta_chat_b960_r480_once_20260817` | no-chat attention score-only | cumulative delta chat | chat | 200 | 0.2763 ± 0.0166 |
| `longbench16_attention_rank_delta_rank_b960_r480_chat_except_trec_20260817/samsum` | no-chat attention score+rank | cumulative delta chat score+rank | chat | 200 | 0.3581 ± 0.0138 |
| `longbench16_attention_rank_delta_rank_b960_r480_chat_except_trec_20260817/qasper` | same | same | chat | 200 | 0.1983 ± 0.0178 |
| `longbench16_attention_rank_delta_rank_b960_r480_chat_except_trec_20260817/narrativeqa` | same | same | chat | 200 | 0.1602 ± 0.0217 |

이 값들은 category-500 학습 전의 baseline/ablation 결과다. Category별 500개 teacher + 20epoch 학습이 끝나면 이 문서의 성능 표를 새 결과로 갱신한다.

## 8. 2026-08-18 dense scorer SAMSum ablation

SAMSum 300개 teacher로 `attention_delta` 2-head scorer를 학습했다. 공통 추론 설정은 다음과 같다.

```text
eval task: longbench_samsum
eval samples: 200
eval prompt: chat template 적용
student_budget: 960
student_refresh_tokens: 480
student_drift_mode: delta_student
student_delta_select_once: True
student_refresh_interval: 1
student_score_activation: softmax
block_length: 32
gen_length: 128
steps: 128
```

Dense scorer teacher는 `active_top_k=0`으로 attention teacher support를 자르지 않는다. 즉 `teacher_norm`은 모든 prompt token에 대한 continuous score이고, budget은 inference에서만 적용한다. Delta는 같은 trajectory에서 모든 prompt token의 cumulative K/V movement를 저장한 `delta_norm`이다.

| run | teacher chat | confidence | target | attention/delta source | score |
|---|---:|---:|---|---|---:|
| `samsum300_dense_chat_2head_b960_r480_once_20260818` | 1 | 1 | `attention_delta` | chat attention + chat delta | `0.3412 ± 0.0133` |
| `samsum300_dense_nochat_attention_delta_2head_b960_r480_once_20260818` | 0 | 1 | `attention_delta` | no-chat attention + no-chat delta | `0.3628 ± 0.0132` |
| `samsum300_dense_nochat_conf0_attention_delta_2head_b960_r480_once_20260818` | 0 | 0 | `attention_delta` | no-chat attention + no-chat delta | `0.3502` |

Artifact paths:

| artifact | path |
|---|---|
| dense chat conf=1 teacher | `/home/M2026107/.cache/dlpc_kv_pruning_teacher_samsum300_dense_chat_20260818` |
| dense chat conf=1 2-head student | `results/budget/dlpc_kv_pruning_samsum300_dense_chat_20260818/attention_delta_2head_rank0p1_topk0_e20/checkpoint-best` |
| dense no-chat conf=1 teacher | `/home/M2026107/.cache/offline_hybrid_teacher_samsum300_dense_nochat_20260818` |
| dense no-chat conf=1 2-head student | `results/budget/student_samsum300_dense_nochat_attention_delta_2head_rank0p1_topk0_e20_20260818/checkpoint-best` |
| dense no-chat conf=0 teacher | `/home/M2026107/.cache/offline_hybrid_teacher_samsum300_dense_nochat_conf0_20260818` |
| dense no-chat conf=0 2-head student | `results/budget/student_samsum300_dense_nochat_conf0_attention_delta_2head_rank0p1_topk0_e20_20260818/checkpoint-best` |

Interpretation:

- Dense continuous scorer 기준에서는 `teacher chat template = 0`, `confidence_weight = 1`이 현재 SAMSum best다.
- `0.3628` 결과는 attention과 delta가 둘 다 no-chat이다. 같은 2-head checkpoint를 `student_path`와 `student_refresh_path`에 넣었고, 해당 checkpoint의 teacher root는 `apply_chat_template=0`, `confidence_weight=1`, `active_top_k=0` shard다.
- `0.3412` 결과는 attention과 delta가 둘 다 chat이다.
- Dense 2-head 결과만 놓고 보면 chat-template teacher가 일관되게 좋은 것은 아니다. 반대로 no-chat teacher가 더 좋았다.
- 그러나 Top-128 teacher 결과에서는 chat/confidence 조합이 더 좋은 기록이 있었으므로, template 효과는 `active_top_k=128` teacher sparsification과 상호작용한다. 따라서 interval 효과와 template/support 효과는 분리해서 봐야 한다.
- 연구 목적상 main method는 dense scorer가 더 깔끔하다. Teacher는 budget-free continuous score를 만들고, keep/refresh budget은 inference-time hyperparameter로만 적용되기 때문이다. Top-128 teacher는 stronger heuristic/teacher-sparsification ablation으로 남기는 것이 적절하다.

현재 진행 중인 확장 추출:

```text
script: scripts/run_dense_nochat_conf1_teacher_300each_rest_20260818.sh
output_root: /home/M2026107/.cache/offline_hybrid_teacher_300each_dense_nochat_conf1_20260818
datasets: 2wikimultihopqa_train, gov_report, hotpotqa, multi_news, musique,
          narrativeqa, qasper, qmsum, trec, triviaqa
setting: no-chat, confidence_weight=1, active_top_k=0, n_samples=300
```

SAMSum은 이미 `/home/M2026107/.cache/offline_hybrid_teacher_samsum300_dense_nochat_20260818`에 같은 설정으로 추출되어 있다.
