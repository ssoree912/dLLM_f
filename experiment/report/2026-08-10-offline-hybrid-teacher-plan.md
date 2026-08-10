# Offline Hybrid Teacher (Reference + KV-Delta) 구현 계획

날짜: 2026-08-10 · 브랜치: `dev/kv_pruning`

## 0. 방법론 요약

Teacher는 전부 **offline extraction**. 실제 inference에는 teacher/origin shadow forward가 없고 student만 돈다.

```
Offline origin trajectory
    → step별 측정을 시간축으로 union/누적 → (R̄[l,i]: suffix→prompt attention, D̄[l,i]: prompt K/V 총 변화량)
    → priority hybrid top-k label (reference top-K_R 우선, 잔여 budget은 delta top-(B-K_R))
    → student distillation
    → inference: student가 현재 state만 보고 prompt token 선택 (+선택적으로 delta 예측 큰 token만 refresh)
```

기존 union teacher와 동일하게 **per-step 텐서는 저장하지 않는다**. 선택할 token도,
refresh할 token도 sample당 고정 집합이므로 시간축 aggregate `[L, P]`면 충분하다.

핵심 비교: 동일 budget B에서 offline teacher label 기준
`reference-only` / `delta-only` / `reference+random` / `reference+delta`.

"online 2-pass oracle"은 기본 실험에서 제외 (trajectory가 달라졌을 때 offline label 유효성을 확인하는
선택적 ceiling 실험일 뿐).

## 1. 기존 코드와의 관계

| 필요 기능 | 기존 코드 | 재사용 방식 |
|---|---|---|
| origin full trajectory + step별 suffix→prompt attention | `dllm_cache/budget/full_dynamic_trajectory_teacher.py` | denoising loop·`SuffixPromptAttentionCollector` 골격 그대로. 단, 시간축으로 합산(aggregate)하지 않고 step별 보존 |
| step별 prompt K/V drift | `dllm_cache/budget/prompt_drift.py` (`PromptDriftCollector`) | `_observe()`의 drift 계산 로직을 같은 attention wrapper 안으로 병합 → **한 forward에서 R과 D 동시 수집** |
| extraction CLI (프롬프트 포맷, chat template, dataset 필터, resume) | `extract_future_pool_teacher_balanced.py` | CLI/샘플 로딩/shard 저장 패턴 복제 |
| student 모델·loss | `student_model.py` (`PromptUtilityStudent`, `topk_bce_loss`) | hybrid mask를 top-k label로 쓰면 기존 BCE 경로가 거의 그대로 맞음 |
| student 학습 루프 | `training_loop.py`, `train_student.py` | `--target-mode` 추가 |
| 선택된 token만 refresh | `drift_refresh_kv.py` | 이후 단계에서 delta head 예측값과 연결 |

`online_teacher.py`는 이번 방법론에서 사용하지 않음.

## 2. Phase A — Offline teacher extraction

### 신규: `dllm_cache/budget/offline_hybrid_teacher.py`

`full_dynamic_trajectory_teacher.py`를 확장한 수집기. batch size 1, 매 denoising step에서
prompt+suffix full forward를 돌리며 wrapper 하나가 두 신호를 **running accumulator로만**
유지한다 (per-step 텐서 보존 없음 — 기존 teacher와 동일한 방식):

- **Reference `R̄[l, i]`**: 기존과 완전히 동일. step별 commit된 suffix token의 query만
  가중한 suffix→prompt attention을 sum/max로 누적 (`accumulate_commits` semantics,
  confidence weighting·`target_aggregation` 옵션 유지). step별 top-k union mask도
  기존 `future_union_mask`처럼 함께 유지.
- **Delta `D̄[l, i]`**: 같은 forward의 K/V projection에서 직전 step 대비 변화량을 누적:
  ```
  D̄[l,i] = Σ_t (‖K_t−K_{t−1}‖ + ‖V_t−V_{t−1}‖)/2 / norm   # trajectory 총 이동 거리
  ```
  "trajectory 동안 가장 많이 변한 token이 무엇인가"의 순위. refresh 집합은 sample당
  고정이므로 이 총량 순위면 충분하고, step 0 대비 누적치(초반 1회 점프 후 정지한
  token을 과대평가)와 달리 지속적으로 움직인 token이 위로 온다.
  `prompt_drift.py`는 V만 측정하는데, K도 attention score에 직접 들어가므로 K/V 평균으로 확장.
  참고용으로 max_t(stepwise)도 같이 누적 (한 번에 크게 튀는 token 진단용, 비용 0).

### 신규: `dllm_cache/budget/extract_offline_hybrid_teacher.py` (CLI)

`extract_future_pool_teacher_balanced.py`와 동일 인터페이스(`--prompt-format`,
`--apply-chat-template`, `--samples-per-dataset`, resume via 기존 파일 skip 등).

**Shard 포맷** (`torch.save`, sample당 1개):

```python
{
  "teacher_kind": "offline_hybrid_ref_delta",
  # 메타: sample_id, dataset, prompt_input_ids, question_token_indices,
  #        generated_answer, prompt_length, gen/steps/block 설정 등 기존 스키마 유지
  "teacher_raw":      f16 [L, P],   # R̄ — 기존 스키마와 같은 이름/의미
  "teacher_norm":     f16 [L, P],
  "ref_union_mask":   bool [L, P],  # 기존 future_union_mask와 동일
  "delta_raw":        f16 [L, P],   # D̄ = Σ_t stepwise K/V 변화량 (총 이동 거리)
  "delta_norm":       f16 [L, P],
  "delta_step_max":   f16 [L, P],   # max_t stepwise (진단용)
}
```

기존 teacher shard(aggregate `teacher_raw [L,P]` + union mask)에 **delta aggregate
`[L,P]`가 추가되는 것뿐**이다. per-step 텐서 없음 → sample당 ~1MB, 용량 문제 없음.
step마다 GPU→CPU 전송도 없고 running accumulator만 유지하므로 추출 속도도 기존 teacher와 동일.

## 3. Phase B — Hybrid label 생성

### 신규: `dllm_cache/budget/hybrid_labels.py` + CLI `build_hybrid_labels.py`

입력: Phase A shard. 출력: 같은 디렉토리 구조의 label shard.

```
R̄[l,i] = shard의 teacher_raw               # 추출 시 이미 aggregate 완료
D̄[l,i] = shard의 delta_raw

layer별:
  ref_idx   = topk(R̄[l], K_R)
  rest      = {0..P-1} − ref_idx
  delta_idx = topk(D̄[l][rest], B − K_R)
  hybrid_mask[l] = ref_idx ∪ delta_idx     # |·| = B 보장
```

동시에 비교용 baseline mask 4종을 한 번에 생성 (같은 B):

| 이름 | 구성 |
|---|---|
| `reference_only` | topk(R̄, B) |
| `delta_only` | topk(D̄, B) |
| `reference_random` | topk(R̄, K_R) + 나머지에서 random (B−K_R), seed 고정 |
| `reference_delta` (제안) | topk(R̄, K_R) + 나머지에서 topk(D̄, B−K_R) |

추가로 진단 통계를 shard에 기록: layer별 `|top_B(R̄) ∩ top_B(D̄)|/B` (Jaccard).
delta top이 reference top의 부분집합이면 hybrid가 reference-only로 퇴화하므로,
이 수치가 실험 성패를 미리 알려준다.

파라미터: `--budget B` (기본 480, 960 두 세팅), `--ref-k K_R` 또는 `--ref-ratio`
(sweep: 1.0, 0.75, 0.5 → ratio 1.0이 곧 reference-only).

## 4. Phase C — Oracle(teacher-label) 평가 — **핵심 비교, student보다 먼저**

Student 없이 label 품질 자체를 먼저 평가한다. samsum **eval** 프롬프트에 대해 Phase A/B를
그대로 돌려 per-sample mask를 만들고, 그 mask로 prompt KV pruning eval을 수행.

구현: `eval_model/LLaDA.py`의 layer-split prompt-KV 경로에 per-sample 고정 mask를 주입하는
`oracle_mask_root=<dir>` model_arg 추가 (student 점수 대신 저장된 mask를 로드해서 선택).
기존 `student_prompt_layer_split=True` 경로의 선택 단계만 바꾸면 되므로 침습이 작다.

실행 매트릭스 (samsum, `local_longbench_samsum`, limit 100, gen 128 / steps 128 / block 32,
chat template — `scripts/sweep_full_refresh_20260810.sh`와 동일 세팅):

```
B ∈ {480, 960}
  × mask ∈ {reference_only, delta_only, reference_random, reference_delta}
  × K_R/B ∈ {0.75, 0.5}        # reference_random / reference_delta에만 적용
+ 상한: origin full KV, 하한: random B
```

판정 기준: 동일 (B, K_R)에서 `reference_delta > reference_random`이면 delta 신호가
실제 정보를 담고 있다는 증거. 이게 성립해야 Phase D로 진행할 가치가 있다.

## 5. Phase D — Student distillation

### `training_loop.py` 확장: `--target-mode` 2종 추가

1. **`hybrid_mask`** (기본): 최종 hybrid mask를 직접 예측. binary `[L,P]` target이므로
   기존 `union` target-mode와 동일 경로 (BCE pos_weight 4 + rank 0.1 + topk).
   `teacher_targets_from_record()`에 case 하나 추가가 전부. 단 **`--topk-k`를 budget B로
   설정** — 기본 128이면 B=480 mask의 1 중 임의 128개만 label이 되어 왜곡된다.
2. **`ref_delta_heads`** (선택): `PromptUtilityStudentLayer`에 score head를 하나 더 달아
   R̄와 D̄를 각각 회귀. 선택 시점에 teacher와 같은 priority 규칙(ref top-K_R → delta 잔여)을
   적용. B, K_R을 학습 후에도 바꿀 수 있는 장점.

먼저 1로 end-to-end를 검증하고, K_R sweep이 필요해지면 2를 붙인다.

### 데이터

- **1차 (지금)**: `data/train_mixed_8task_300/mixed_train_longbench_format.jsonl`에서
  `--datasets samsum` (300개), `--apply-chat-template`.
- **2차**: 같은 파일의 8개 dataset 전부 (기존 300-each mixed teacher와 동일 커버리지).

학습 하이퍼파라미터는 기존 성공 세팅 재사용: e10, lr 2e-5, topk_weight 0.02 계열
(`future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02` 참고).

### Student 평가

Phase C와 동일 eval 세팅에서 `student_path=...`로 교체 실행. 비교 대상:
oracle hybrid (ceiling) / student hybrid / student reference-only / 기존 mixed8 student.

### (후속) Delta 기반 refresh

`ref_delta_heads` 모드가 있으면 선택된 token 중 delta 예측 상위만
`drift_refresh_kv.py` 경로로 refresh — 이번 기본 실험 범위 밖, 인터페이스만 맞춰둔다.

## 6. 작업 순서 및 산출물

| # | 작업 | 신규/수정 파일 | 검증 |
|---|---|---|---|
| 1 | 통합 collector + trajectory 러너 | `offline_hybrid_teacher.py` (신규) | 단위테스트: R이 `full_dynamic_trajectory_teacher` 재현(aggregate 일치), D가 `prompt_drift` 재현 |
| 2 | extraction CLI | `extract_offline_hybrid_teacher.py` (신규) | samsum 2~3개 스모크, shard 스키마/용량 확인 |
| 3 | label builder | `hybrid_labels.py`, `build_hybrid_labels.py` (신규) | 단위테스트: budget 정확히 B, ref/delta 집합 disjoint, ratio=1.0 ≡ reference-only; Jaccard 통계 출력 |
| 4 | oracle mask 주입 eval | `eval_model/LLaDA.py`, `layer_split_prompt_kv.py` (수정) | limit 5 스모크 → limit 100 본실험 |
| 5 | Phase C 실험 실행 | `scripts/run_samsum_oracle_hybrid.sh` (신규) | 4-way 비교표 → report |
| 6 | student target-mode | `training_loop.py`, `train_config.py` (수정) | samsum 300 학습 → eval |
| 7 | (조건부) ref_delta_heads | `student_model.py` (수정) | K_R sweep |

1–3은 GPU 점유가 짧아 바로 진행 가능. 4–5가 첫 번째 의사결정 지점
(`reference_delta > reference_random` 확인), 6–7은 그 결과를 보고 진행.

## 7. 리스크

- **stepwise delta의 노이즈**: step 간 변화량은 개별 step에서 작고 노이즈가 섞일 수 있음.
  T=128 step 누적합이 이를 상쇄하며, 분포가 너무 평평하면 `delta_step_max`와 교차 검증.
- **delta 신호 퇴화**: R top과 D top의 overlap이 크면 hybrid ≈ reference-only.
  Phase B의 Jaccard 통계로 실험 전에 감지.
- **직전 K/V 유지 메모리**: delta 계산을 위해 직전 step의 prompt K/V를 layer별로 GPU에
  유지 (P=1920 기준 ~1GB, bf16). 매 step CPU 전송은 없고, 최종 `[L,P]` 누적 결과만 한 번
  내린다. `prompt_drift.py`가 이미 같은 방식(first+previous 2벌)으로 검증됨.
- **chat template 정합**: 기존 교훈대로 extraction과 eval 모두 `--apply-chat-template`
  고정 (mixed8_chat 계열과 동일).
