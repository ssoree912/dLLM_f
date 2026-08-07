# SAMSum 프롬프트 예산 실험 정리 (2026-08-07)

## 0. 한 줄 요약

LongBench SAMSum 200행에서 프롬프트를 절반(960/1920)으로 줄여도 품질은 96% 유지되지만
(0.3835 → 0.3682), 그 절반을 step 0에 **얼리면** 54%로 떨어진다 (0.2071). 즉 손실의 91%는
토큰을 버려서가 아니라 **캐시가 낡아서** 생긴다. 한편 토큰을 하나도 안 버리는 dLLM-Cache는
품질 손실 없이 3.4배 빠르다 (0.3868, 27분). 시간 축만 놓고 보면 프롬프트 pruning은 dLLM-Cache에
밀리며, 우리 쪽 우위는 **메모리**에 있다.

---

## 1. 평가 환경

### 1.1 평가 코드 — 공식 lm-evaluation-harness

이 보고서의 **모든 수치**는 EleutherAI lm-evaluation-harness 0.4.12로 측정했다.
자체 하네스는 쓰지 않았다 (이유는 6절).

```
실행 위치: /home/M2026107/dllm/dLLM-Cache
진입점   : evaluation_script.py  (lm_eval.__main__.cli_evaluate 래퍼)
모델 래퍼: eval_model/LLaDA.py   (@register_model("LLaDA"))
파이썬   : /home/M2026107/dllm/dLLM-Cache/.venv/bin/python
```

### 1.2 데이터셋

```
태스크 : local_longbench_samsum   (LongBench SAMSum test, 200행 전체)
파일   : /home/M2026107/dllm/data/longbench/samsum.jsonl
정의   : experiment/345/2026-07-15/tasks/longbench_local/samsum.yaml
```

로컬 yaml은 공식 `lm_eval/tasks/longbench/samsum.yaml`과 내용이 같다. 차이는 데이터 출처
(HF `Xnhyacinth/LongBench` → 로컬 jsonl)와 필드명(`{{question}}` → `{{input}}`)뿐이며,
`process_results`는 공식 `lm_eval.tasks.longbench.metrics`를 그대로 재수출한다.

- **지표**: ROUGE-L F1 (`rouge` 패키지의 `rouge-l.f`) — LongBench SAMSum 공식 지표
- **정지 규칙**: `until: ["\n"]` — 첫 줄바꿈에서 생성 중단 (공식 규정)
- **프롬프트**: 원본 2,861~18,180 토큰 → 좌측 절단으로 1,920 토큰

### 1.3 생성 설정 (전 실행 공통)

```
max_length   = 2048   (프롬프트 1920 + 생성 canvas 128)
gen_length   = 128
steps        = 128
block_length = 32
cfg_scale    = 0.0
num_fewshot  = 0   (--apply_chat_template --fewshot_as_multiturn)
모델         : /home/M2026107/dllm/model/LLaDA-8B-Instruct
GPU          : A100-SXM4-40GB 1장
```

### 1.4 사용한 student

```
경로: /home/M2026107/dllm/dLLM-Cache/results/budget/
      future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best
```

| 항목 | 값 |
|---|---|
| 구조 | `PromptUtilityStudent` (layer 32, hidden 4096, proj 256, mlp 512) |
| teacher | **future_pool** — 원본 모델이 생성하는 동안 매 step "이번에 확정될 위치"가 프롬프트를 본 attention을 확신도로 가중 누적, 상위 K를 union/빈도로 집계 |
| 학습 데이터 | 11개 태스크 (2wikimqa, gov_report, hotpotqa, multi_news, musique, narrativeqa, qasper, qmsum, **samsum**, trec, triviaqa) |
| epochs / lr | 10 / 2e-5, topk_k=128 |
| 출력 | `[layer, prompt]` 점수 1개 — **중요도만**, 드리프트 개념 없음 |

비교용으로 한 번 더 쓴 student:

```
results/budget/future_pool_student_few_shot_train_1k_topk128_e10_lr2e-5_tw0.02/checkpoint-best
  → 학습 데이터: samsum, trec, triviaqa (few-shot 카테고리 3개)
```

---

## 2. 비교한 방식

프롬프트 1,920 토큰 중 **960(50%)만 남기는 것**이 공통 목표이며, 남긴 토큰을 어떻게
계산하느냐가 방식별로 다르다.

| 방식 | 시퀀스 길이 | 프롬프트 표현 | step당 블록 통과 토큰 |
|---|---:|---|---:|
| **origin** | 2048 | 매 step 갱신 | 2048 |
| **dLLM-Cache** | 2048 | feature를 N step 재사용 | 2048 (일부 연산 생략) |
| **dynkv** (우리) | **1088** | 매 step 갱신 | **1088** |
| **kv_cache** (우리) | 프리필 2048 → 이후 128 | **step 0에 동결** | **128** |

### 2.1 kv_cache — 프리필 후 동결

`student_prompt_kv_cache=True`. 프롬프트 1,920개를 **전부** 32개 블록에 흘려
문맥 완전한 표현을 만든 뒤, layer마다 student 점수 상위 960개의 K/V만 저장하고 나머지는 버린다.
이후 128 step은 suffix 128개만 forward하고, 프롬프트 K/V는 저장된 값을 조회만 한다.

```
Q   = suffix 128
K,V = 캐시 960 + suffix 128 = 1088
```

캐시된 960개는 **전부** attention에 들어간다. 캐시 안에서 추가 선택은 없다.

### 2.2 dynkv — 실제 시퀀스 축소

`student_prompt_dynamic_kv=True, student_selection_mode=global`. 선택된 960개만으로
길이 1,088짜리 실제 시퀀스를 만들어 매 step 통째로 forward한다. RoPE 위치는 원래 프롬프트
위치를 유지한다. `refresh_interval=1`이면 매 step 재계산이므로 **stale이 없다**.

### 2.3 dLLM-Cache — 시간축 재사용 (원본 방법)

`prompt_interval_steps=100, gen_interval_steps=8, transfer_ratio=0.25, is_feature_cache=True`.
공식 `scripts/run_LLaDA_long_bench_Instruct.sh`의 LongBench 설정 그대로다.
**토큰을 하나도 버리지 않고** feature를 N step 재사용한다. `transfer_ratio`는 생성 토큰 중
코사인 유사도가 낮은(=많이 변한) 25%만 골라 갱신하며, **프롬프트에는 적용되지 않는다**.

---

## 3. 결과

### 3.1 전체 표 (LongBench SAMSum 200행)

| # | 방식 | 프롬프트 토큰 | 프롬프트 갱신 | ROUGE-L | origin 대비 | 소요 |
|---:|---|---:|---|---:|---:|---:|
| 1 | origin (pruning 없음) | 1920 | 매 step | 0.3835 ± 0.0124 | 1.000 | 91분 |
| 2 | **dLLM-Cache 공식** | **1920** | 100 step마다 | **0.3868 ± 0.0126** | **1.009** | **27분** |
| 3 | dynkv 960 (refresh=1) | 960 | 매 step | 0.3682 ± 0.0138 | 0.960 | 54분 |
| 4 | dynkv 480 (refresh=1) | 480 | 매 step | 0.2999 ± 0.0137 | 0.782 | 33분 |
| 5 | kv_cache 960 (`300each`) | 960 | **없음(동결)** | 0.2071 ± 0.0133 | 0.540 | 15분 |
| 6 | kv_cache 960 (`few_shot_1k`) | 960 | **없음(동결)** | 0.1947 ± 0.0139 | 0.508 | 16분 |
| 7 | dynkv 960 (refresh=2) | 960 | 2 step마다 | *실행 중* | — | ~35분 |

결과 위치: `/home/M2026107/.cache/lmeval_samsum_b960_20260807/<이름>/`
실행 로그: 같은 디렉토리의 `run.log`, 실행 스크립트 `run.sh` ~ `run7.sh`

### 3.2 손실 분해 — 이 실험의 핵심

3번과 5번은 **선택된 토큰도 같고 attention에 들어가는 key 집합(960+128)도 같다.**
유일한 차이는 그 960개의 K/V를 매 step 다시 계산하느냐, step 0 값으로 쓰느냐다.

```
origin        0.3835
   ↓ −0.0153    ← 프롬프트 절반을 버린 손실 (표준오차 ±0.014 → 사실상 무료)
dynkv 960     0.3682
   ↓ −0.1611    ← 캐시 stale 손실  (전체 손실의 91%)
kv_cache 960  0.2071
```

**student의 선택은 이미 충분히 좋다.** 문제는 고른 토큰을 얼려버린 것이다.

### 3.3 예산 곡선

| 예산 | 토큰 | ROUGE-L | origin 대비 |
|---|---:|---:|---:|
| 100% | 1920 | 0.3835 | 1.000 |
| **50%** | 960 | 0.3682 | **0.960** |
| **25%** | 480 | 0.2999 | **0.782** |

50%는 거의 공짜(−0.015)지만 25%는 손실이 5.5배로 늘고 신뢰구간이 origin과 완전히 분리된다.
**프롬프트 잉여는 절반까지**이며, 그 아래로는 실제 정보를 버리기 시작한다.

### 3.4 student 비교

| student | 학습 데이터 | ROUGE-L (kv_cache 960) |
|---|---|---:|
| `300each` | 11개 태스크 (samsum 포함) | 0.2071 ± 0.0133 |
| `few_shot_1k` | samsum, trec, triviaqa | 0.1947 ± 0.0139 |

차이 0.012, 표준오차 ±0.013으로 **구분되지 않는다.** samsum을 좁게 학습한 쪽이 낫지 않다.

---

## 4. 연산 구조 분석

### 4.1 왜 kv_cache가 6배 빠른가

절감의 원천은 pruning이 아니라 **캐시**다. 프롬프트가 네트워크를 통과하지 않으므로
선형계층·MLP·프롬프트 self-attention이 통째로 사라진다.

| 항목 | origin | dynkv 960 | kv_cache 960 |
|---|---:|---:|---:|
| 블록 통과 토큰 | 2048 | 1088 | **128** |
| 프롬프트 self-attention | 1920×2048 | 960×1088 | **없음** |
| suffix attention | 128×2048 | 128×1088 | 128×1088 |

LLaDA-8B 기준 블록당 선형계층이 토큰당 약 1.7억 FLOPs인 반면, suffix가 프롬프트를 보는
attention은 전체의 0.5% 미만이다. **토큰을 절반 버리는 것만으로는 속도가 거의 안 변한다.**

### 4.2 공간 분할 vs 시간 분할 — 연산량 동일

"960 중 절반만 매 step 갱신"과 "2 step마다 960 전부 갱신"은 총 연산량이 정확히 같다.

```
A(공간): 128 step × (480 + 128) = 77,824 토큰-forward
B(시간): 64×(960+128) + 64×128  = 77,824 토큰-forward
attention도 동일: 둘 다 84.7M query-key 쌍
```

차이는 **stale 구조**다.

| | stale 프로파일 |
|---|---|
| A (중요도 고정 분할) | 상위 480은 항상 신선, **하위 480은 영구 동결** |
| B (refresh=2) | **전 토큰이 최대 1 step** |

같은 값을 치르고 B가 stale 최대치가 작으므로, A(토큰별 선택 갱신)가 의미를 가지려면
B를 넘어야 한다. 그래서 7번 실행이 다음 단계의 기준선이 된다.

### 4.3 메모리

| 방식 | 프롬프트 KV | 활성화 |
|---|---|---|
| origin | 1920 | 2048 토큰분 |
| dLLM-Cache | 1920 + **feature 캐시 추가** | 2048 토큰분 |
| dynkv 960 | **960** | **1088 토큰분** |
| kv_cache 960 | **960** | 128 토큰분 |

**dLLM-Cache는 메모리를 줄이지 않는다.** feature를 저장하므로 오히려 늘어난다.
긴 컨텍스트에서 먼저 터지는 것은 메모리이며, 여기가 프롬프트 pruning의 자리다.

---

## 5. 해석과 다음 단계

### 5.1 확인된 것

1. **프롬프트 50%는 잉여다** — 매 step 갱신하면 −0.015로 사실상 무료
2. **손실의 91%는 stale** — 선택이 아니라 동결이 문제
3. **25%는 과하다** — −0.084로 명확히 깎임
4. **시간 축 단독 비교로는 dLLM-Cache 우세** — 27분/0.3868 vs 우리 54분/0.3682

### 5.2 우리 방법의 자리

- **메모리**: dLLM-Cache가 못 하는 영역. 50%에서 품질 96% 유지하며 KV 절반
- **결합**: 두 축이 직교하므로 곱해진다. 프롬프트를 960으로 줄인 뒤 dLLM-Cache를 얹는
  실험은 코드상 배타적이지 않아 즉시 가능 (`is_feature_cache=True` + `student_prompt_dynamic_kv=True`)

### 5.3 다음 단계 — 드리프트 기반 선택 갱신

현재 student는 **중요도만** 예측한다. 손실의 91%가 stale에서 오므로, 남은 여지는
**안정성 축**에 있다. 제안하는 구조:

```
1. 자를 토큰   : student 중요도 top-960        (공간축, 기존)
2. 갱신할 토큰 : 그 960 중 변화량 큰 K개       (시간축, 신규)
3. attention   : 갱신된 K + 낡은 (960−K) + suffix
```

dLLM-Cache의 `refresh_index`가 생성 토큰에 대해 같은 발상을 구현하지만
(코사인 유사도가 낮은 = 많이 변한 것을 갱신), **프롬프트에는 적용되어 있지 않다.**

또한 dLLM-Cache식 런타임 선택은 `v_proj`를 매 step 전체 계산해야 한다(선택 기준을 만들려면
새 값이 필요). 프롬프트는 내용이 고정이므로 **사전 측정한 드리프트 통계로 스케줄을 미리
정하면 그 계산이 불필요하다.** 이것이 차별점이 될 수 있다.

#### 검증 순서 (재학습 없이 약 1시간)

| 단계 | 내용 | 비용 |
|---|---|---|
| 1 | 드리프트 실측 — full 생성 중 매 step 프롬프트 K/V를 step 0과 비교 | ~20분 |
| 2 | sanity check (layer 0 드리프트 = 0), 중요도와의 상관, seed 간 재현성 | 0 |
| 3 | student 없이 오라클 검증 — `build_prompt_kv_cache`에 임의 점수 주입 가능 | ~30분 |
| 4 | (통과 시에만) teacher 재추출 + student 2채널 학습 | 수 시간 |

측정할 지표:

```
cum_drift[l,i]  = mean_t ||v_t − v_0|| / ||v_0||        # 완전 동결 시 손해
step_drift[l,i] = mean_t ||v_t − v_{t−1}|| / ||v_t||    # 갱신 주기 결정
weighted[l,i]   = mean_t a_t[l,i] · ||v_t − v_0||       # 실제 attention 오차 기여
```

`a_t`(매 step 프롬프트 attention)는 `future_pool_teacher`가 이미 수집 중이므로 추가 비용이 없다.
저장은 토큰당 스칼라라 수백 KB 수준.

#### 성립 조건

**드리프트가 프롬프트만 보고 예측 가능해야 한다.** 드리프트가 토큰의 역할(few-shot 예시 vs
요약 대상)로 결정되면 student가 맞힐 수 있고, 무엇이 생성되느냐로 결정되면 불가능하다.
같은 프롬프트를 다른 seed로 여러 번 생성해 드리프트 패턴이 재현되는지로 확인한다.

---

## 6. 방법론 주의사항

### 6.1 자체 하네스 수치는 이 표에 섞지 말 것

2026-08-06까지 **별도 저장소** `dLLM-Cache-step-distill`의
`experiment/4090/2026-08-03/step_distill/` 자체 평가기로 낸 수치는 공식 하네스와 네 축이
모두 다르다 — 데이터셋, 프롬프트 템플릿, ROUGE 구현, **정지 규칙**.

특히 정지 규칙이 문제였다. LongBench SAMSum은 `until: ["\n"]`을 규정하는데 자체 평가기는
EOS/EOT만 적용해, pruning 없는 baseline이 요약을 끝낸 뒤에도 canvas를 계속 채웠다.
동일한 생성 결과에 공식 규칙을 적용하면 `full`이 **0.2201 → 0.3343**으로 바뀌며,
"B=960 pruning이 full을 25% 앞선다"는 잘못된 결론이 나왔던 원인이다.

해당 평가기들은 `dLLM-Cache-step-distill` 커밋 `9f10e20`에서 제거했다. teacher 추출과
분포 증류 학습 코드는 lm_eval에 대체물이 없어 그 저장소에 남겼으며, 거기 남은
`summary_metrics`/`generation_output`은 **학습 진단용이지 벤치마크 점수가 아니다.**

**이 실험에서 실행한 코드는 전부 이 저장소(`dLLM-Cache`)에 있다.** `step-distill`은 현재
teacher 추출·분포 증류 학습 전용이며 이번 평가 경로에는 관여하지 않는다. 두 저장소의
`eval_model/LLaDA.py`는 동기화되어 있지 않으니, 실행은 반드시 이쪽에서 한다.

### 6.2 소규모 표본에 속지 말 것

8샘플 예비 실행에서 "student 0.4143 > full 0.3789"가 나왔으나 200샘플에서는
student 0.2071 vs full 0.3835로 역전됐다. 8샘플의 표준오차는 ±0.08~0.10으로
아무것도 판별하지 못한다. 200샘플에서 ±0.012~0.014까지 좁혀진다.

---

## 7. 실행 명령 예시

```bash
cd /home/M2026107/dllm/dLLM-Cache

# origin
CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 .venv/bin/python evaluation_script.py \
  --model LLaDA --tasks local_longbench_samsum \
  --include_path experiment/345/2026-07-15/tasks/longbench_local \
  --batch_size 1 --limit 200 \
  --model_args "pretrained=/home/M2026107/dllm/model/LLaDA-8B-Instruct,\
is_feature_cache=False,is_cfg_cache=False,max_length=2048" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 --log_samples --apply_chat_template \
  --fewshot_as_multiturn --trust_remote_code \
  --output_path /path/to/out

# dynkv (시퀀스 축소 + 매 step 갱신)
#   위 model_args 에 아래를 추가
#   student_path=results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best,
#   student_prompt_dynamic_kv=True,student_budget=960,
#   student_selection_mode=global,student_refresh_interval=1,student_question_window=128

# kv_cache (프리필 후 동결)
#   student_prompt_kv_cache=True,student_budget=960,student_question_window=128

# dLLM-Cache 원본
#   prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,
#   transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False
```

실행 환경과 모드별 인자는 `experiment/325/2026-08-07/README.md` 참고.

관련 커밋:

- `1d9a381` (이 저장소, `dev/budget`) — `student_prompt_dynamic_kv` / `student_selection_mode` /
  `student_refresh_interval` 노출, LongBench yaml 데이터 경로 수정
- `9f10e20` (dLLM-Cache-step-distill, `a100/step_distll`) — 자체 평가기 제거, MMR 경로 복원,
  `sink_filter` 추가
