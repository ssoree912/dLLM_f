# SAMSum 프롬프트 예산 실험 정리 (2026-08-07)

## 0. 한 줄 요약

LongBench SAMSum 200행에서 프롬프트를 절반(960/1920)으로 줄여도 품질은 96% 유지되지만
(0.3835 → 0.3682), 그 절반을 step 0에 **얼리면** 54%로 떨어진다 (0.2071). 즉 손실의 91%는
토큰을 버려서가 아니라 **캐시가 낡아서** 생긴다.

낡는 정도는 layer마다 크게 다르다. 프롬프트 value 벡터의 상대 이동량은 layer 0에서 0.00,
layer 8에서 0.04, layer 16에서 0.13, layer 24에서 0.37이다. 이 측정에 따라 얕은 layer의
프롬프트를 아예 forward하지 않는 **layer-split**을 구현한 결과, 드리프트가 예측한 그대로
나왔다 — layer 16까지 얼리면 0.3671(원본의 95.7%)로 연산이 44% 줄고, 드리프트가 폭증하는
layer 24까지 얼리면 0.2487로 무너진다. **드리프트 측정이 어디를 얼려도 되는지를 실제로
예측한다는 것이 이번 실험의 핵심 성과다.**

다만 세 축을 **동일 연산량에서 비교하면 시간축이 이긴다**(3.7절). 갱신 주기를 늘리는 쪽이
layer·토큰을 줄이는 쪽보다 일관되게 나으며, `refresh=16`은 origin 연산의 9.2%로 품질 98%를
유지한다. refresh와 layer/토큰 축이 오차를 다르게 분배하기 때문이다 — 전자는 모든 위치를 짧게
낡게 두고, 후자는 일부를 영구히 얼린다. 드리프트가 step당 3%로 완만하므로 전자의 누적 오차가
훨씬 작다.

비교 대상인 dLLM-Cache는 토큰을 하나도 안 버리고 품질 손실 없이 3.4배 빠르다(0.3868, 8.1초/샘플).
우리 방식의 차별점은 속도가 아니라 **프롬프트 KV 메모리가 절반**이라는 데 있다.

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
| 7 | dynkv 960 (refresh=2) | 960 | 2 step마다 | 0.3715 ± 0.0140 | 0.969 | 40분 |
| 8 | **layersplit fl=8** | 960 | layer 8부터만 | **0.3777 ± 0.0137** | **0.985** | 42분 |
| 9 | **layersplit fl=16** | 960 | layer 16부터만 | **0.3671 ± 0.0147** | **0.957** | **34분** |
| 10 | layersplit fl=24 | 960 | layer 24부터만 | 0.2487 ± 0.0151 | 0.649 | 25분 |

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

### 3.4 갱신 빈도 축 (프롬프트 960 고정)

| refresh_interval | 갱신 횟수 | 평균 토큰/step | origin 대비 연산 | ROUGE-L | origin 대비 품질 |
|---:|---:|---:|---:|---:|---:|
| 1 (매 step) | 128 | 1088 | 53.1% | 0.3682 ± 0.0138 | 0.960 |
| **2** | 64 | 608 | 29.7% | **0.3715 ± 0.0140** | **0.969** |
| **4** | 32 | 368 | 18.0% | **0.3822 ± 0.0142** | **0.997** |
| **8** | 16 | 248 | 12.1% | **0.3757 ± 0.0139** | **0.980** |
| **16** | 8 | 188 | **9.2%** | **0.3758 ± 0.0137** | **0.980** |
| ∞ (완전 동결) | 0 | 128 | 6.25% | 0.2071 ± 0.0133 | 0.540 |

연산량은 `평균 토큰/step = 960/R + 128`로 정해진다(refresh step만 프롬프트 960을 forward,
나머지 step은 suffix 128만). origin(2048/step) 대비로 refresh=1은 이미 53%(토큰 절반 pruning),
refresh=16은 9.2%까지 내려간다. 하한은 ∞의 6.25%(suffix 전용)다.

**품질은 refresh=1→16에서 평평한데 연산은 53%→9.2%로 5.8배 준다 — 무손실이다.** 붕괴는 오직
마지막 한 칸(9.2%→6.25%, 갱신 8회→0회)에서만 일어나며 여기서 품질이 0.376→0.207로 꺾인다.
갱신 자체는 싸고, 그 몇 번의 갱신이 결정적이다.

#### 왜 드물게 갱신해도 되는가

두 가지가 겹친 결과다.

1. **드리프트는 누적이 아니라 포화한다.** 4.5절에서 step간 변화는 2~3%인데 누적
   (`‖v_t−v_0‖/‖v_0‖`)은 layer 16에서 0.13, layer 24에서 0.37에 그친다. 만약 매 step 독립
   방향으로 움직였다면 128 step 뒤 누적은 √128×0.024 ≈ 0.28은 됐어야 한다. 실제는 그보다
   작으므로 value 벡터는 **초반에 빠르게 이동한 뒤 한 점으로 수렴해 진동**한다. 즉 몇 step만
   지나면 프롬프트 K/V는 거의 안 변하고, 16 step 낡은 값도 갓 계산한 값과 큰 차이가 없다.

2. **step 0만 질적으로 다르다(최악).** step 0의 프롬프트 K/V는 생성 canvas가 전부 [MASK]인
   상태 — 즉 **모델이 아무것도 만들기 전**의 표현이다. 깊은 layer의 프롬프트 표현이 드리프트하는
   이유가 바로 진화하는 suffix를 보기 때문인데(layer 0의 드리프트가 0인 것과 대칭),
   완전 동결(∞)은 이 "답을 못 본" 표현을 128 step 내내, 특히 답이 확정되는 후반까지 쓴다.
   반면 refresh≥1은 step 0을 **부분 생성된 답을 본** 표현으로 갈아끼운다. 8번만 갈아끼워도
   후반의 핵심 상태를 잡는다.

두 사실을 합치면 갱신 #9~#128의 한계 효용은 ≈0(이미 포화·이미 답을 봄)이고, 갱신 #1~#8이
0.207→0.376을 되돌린다. 그래서 곡선이 refresh=16과 ∞ 사이에서만 꺾인다. 이는 3.5절 layer 축과
정확히 같은 구조다 — 얕은 layer(드리프트 작음)는 얼려도 공짜, step 0 전용 앵커는 깊은 layer에
해당하는 최악의 경우다. 시간축과 layer축 모두 **"프롬프트가 답을 보게 하는" 재계산만이 값을
한다**는 하나의 원리로 설명된다.

### 3.5 layer 축 — 드리프트 예측의 검증

프롬프트 960 유지, `frozen_layers` 미만 layer에서는 프롬프트를 forward하지 않고 프리필 K/V를 사용.

| frozen_layers | 얼린 구간의 누적 드리프트 | ROUGE-L | origin 대비 | step당 토큰-블록 | 샘플당 |
|---:|---:|---:|---:|---:|---:|
| 0 (dynkv) | — | 0.3682 | 0.960 | 34,816 | 16.2초 |
| **8** | 0.04 | **0.3777** | **0.985** | 27,136 | 12.7초 |
| **16** | 0.13 | **0.3671** | **0.957** | 19,456 | **10.1초** |
| 24 | **0.37** | 0.2487 | 0.649 | 11,776 | 7.6초 |

**드리프트가 성능을 예측한다.** 4%인 구간을 얼리면 손실이 없고(dynkv보다 오히려 높다),
37%인 구간까지 얼리면 무너진다. 경계는 layer 16 부근이며 여기서 연산 44% 절감에 품질 95.7%다.

### 3.6 layer 축 × 토큰 축 결합

`frozen_layers=16` 아래는 프롬프트를 forward하지 않고, 그 위에서도 상위 `refresh_tokens`개만
forward한다. 나머지는 프리필 K/V를 쓴다. 선택 기준은 student 중요도 점수다.

| 설정 | 갱신 토큰 | ROUGE-L | origin 대비 | step당 토큰-블록 |
|---|---:|---:|---:|---:|
| layersplit fl=16 | 960 (전부) | **0.3671 ± 0.0147** | 0.957 | 19,456 |
| + rt=480 | 480 | 0.3446 ± 0.0160 | 0.898 | 11,776 |
| + rt=240 | 240 | 0.3124 ± 0.0166 | 0.815 | 7,936 |

**토큰 축에는 layer 축 같은 "공짜 구간"이 없다.** 줄일수록 단조 감소한다.

4.5절에서 `weighted` 상위 25%가 오차의 86%를 차지한다고 나왔지만, 실제로는 나머지 14%가 그대로
손실로 드러났다. layer 축에서는 얕은 layer의 드리프트가 **절대적으로 작아서**(0.04) 얼려도
무해했지만, 토큰 축에서는 하위 토큰도 드리프트 자체는 크고 attention 가중치만 작다. 작은 오차가
720개 위치에 쌓이면 무시할 수 없다.

선택 기준을 student 중요도로 쓴 한계도 있다. `weighted`를 직접 쓰면 나아질 수 있으나 현재
student는 드리프트를 예측하지 못한다.

### 3.7 동일 연산량 비교 — 시간축이 이긴다

128 step 전체의 토큰-블록 총량으로 정규화하면 세 축을 직접 비교할 수 있다
(origin = 128 × 32 × 2048 = 8,388,608).

| 총 토큰-블록 | 시간축 (refresh) | layer/토큰 축 |
|---:|---|---|
| 2,490,368 | refresh=2 → **0.3715** | layersplit fl=16 → 0.3671 |
| 1,507,328 | refresh=4 → **0.3822** | fl=24 → 0.2487 / fl=16+rt=480 → 0.3446 |
| 1,015,808 | refresh=8 → **0.3757** | fl=16+rt=240 → 0.3124 |
| 770,048 | refresh=16 → **0.3758** | — |

**같은 연산량에서 갱신 주기를 늘리는 쪽이 layer·토큰을 줄이는 쪽보다 일관되게 낫다.**
1,507,328에서 refresh=4는 0.3822인데 fl=24는 0.2487로 0.13 차이가 난다.

이유는 두 축이 오차를 다르게 분배하기 때문이다. refresh는 **모든 위치를 짧게** 낡게 두지만
(최대 R−1 step), layer/토큰 축은 **일부를 영구히** 얼린다. 드리프트가 step당 3%로 완만하므로
전자의 누적 오차가 훨씬 작다.

refresh=16은 origin 연산의 9.2%로 품질 98%를 유지한다. 다만 3.4절의 refresh=4/8/16 행은 이
문서의 다른 실행과 별도로 추가된 값이라 실측 소요시간 기록이 없다.

### 3.8 student 비교

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

## 4.5 드리프트 측정 결과

`dllm_cache/budget/measure_prompt_drift.py`로 SAMSum 8샘플에서 측정 (5절 실행법 참고).

**측정 검증**: layer 0의 누적 드리프트가 정확히 0.0000으로 나왔다. layer 0의 K/V는 임베딩과
위치로만 결정되어 suffix와 무관하므로 예상대로다. 훅이 올바른 값을 잡고 있다.

### layer별 프로파일

| layer | 누적 (`‖v_t−v_0‖/‖v_0‖`) | step간 |
|---:|---:|---:|
| 0 | 0.0000 | 0.0000 |
| 8 | 0.0404 | 0.0086 |
| 16 | 0.1345 | 0.0245 |
| 24 | 0.3740 | 0.0716 |
| 31 | 0.3495 | 0.0730 |

step간 변화가 3% 수준이라 2 step 뒤에도 6%다. `refresh_interval=2`가 멀쩡한 이유가 설명된다.

### 토큰별 분포와 상관

| | 중앙값 대비 최대 | 상위 25%가 차지하는 비중 |
|---|---:|---:|
| attention | 293배 | 69.2% |
| **weighted** (attention × 드리프트) | **1942배** | **86.0%** |

| 상관 (Spearman) | 값 |
|---|---:|
| attention vs 누적 드리프트 | **0.096** |
| attention vs weighted | 0.727 |

**중요도와 드리프트는 거의 무관하다(0.096).** 드리프트가 기존 중요도 점수의 재표현이 아니라
독립적인 축이라는 뜻이며, 이것이 새 신호로서 가치를 갖는 근거다.

동시에 실제 캐시 오차(`weighted`)는 attention 쪽에 끌려간다(0.727). attention 분포가 극단적으로
뾰족하기 때문이다. **오차의 86%가 상위 25% 토큰에서 나오므로, 깊은 layer에서도 960개 전부가
아니라 상위 일부만 갱신하면 된다.**

### 주의

`temperature=0.0`이라 생성이 결정적이어서, seed를 바꾼 반복이 소수점까지 동일하게 나왔다.
**"드리프트가 프롬프트만으로 예측 가능한가"는 아직 검증되지 않았다.** 확인하려면 온도를 올리거나
서로 다른 프롬프트 간에 토큰 역할별 패턴이 일관되는지를 봐야 한다.

---

## 5. 해석과 다음 단계

### 5.1 확인된 것

1. **프롬프트 50%는 잉여다** — 매 step 갱신하면 −0.015로 사실상 무료
2. **손실의 91%는 stale** — 선택이 아니라 동결이 문제
3. **25%는 과하다** — −0.084로 명확히 깎임
4. **드리프트가 성능을 예측한다** — layer 축에서 인과 확인 (3.5절)
5. **얕은 layer는 공짜로 얼릴 수 있다** — fl=16에서 연산 44% 절감에 품질 95.7%
6. **중요도와 드리프트는 직교한다** (상관 0.096) — 별개의 신호

### 5.2 현재 위치

| 방법 | ROUGE-L | 샘플당 | 프롬프트 KV | 메모리 |
|---|---:|---:|---:|---|
| origin | 0.3835 | 27.1초 | 1920 | 기준 |
| dLLM-Cache | **0.3868** | **8.1초** | 1920 | **증가** (feature 저장) |
| **layersplit fl=16** | 0.3671 | **10.1초** | **960** | **절반** |
| layersplit fl=8 | 0.3777 | 12.7초 | 960 | 절반 |

layer-split 이전에는 dLLM-Cache와 2배 차이였으나(16.2초 vs 8.1초) 이제 10.1초 vs 8.1초로
좁혀졌다. 품질은 5% 낮지만 **프롬프트 KV 메모리가 절반**이며, 이는 dLLM-Cache가 제공하지
못하는 축이다(오히려 feature 저장으로 메모리가 는다).

- **결합 여지**: 시간축(dLLM-Cache)과 공간축(프롬프트 pruning)은 직교하므로 곱해진다.
  코드상 배타적이지 않아 `is_feature_cache=True` + `student_prompt_layer_split=True` 조합을
  바로 실험할 수 있다.

### 5.2.1 축들의 우열 (3.7절 정리)

동일 연산량에서 **시간축(refresh) > layer 축 > 토큰 축** 순이다. refresh=16 하나로 origin
연산의 9.2%에 품질 98%가 나오므로, 현재로서는 **갱신 주기를 늘리는 것이 가장 효율적인 축**이다.

layer 축의 가치는 이제 "단독 절감 수단"이 아니라 **시간축과 곱해질 수 있는 직교 축**에 있다.
refresh=16과 fl=8을 함께 쓰면 갱신하는 8번의 step에서도 얕은 layer를 건너뛸 수 있다.
아직 측정하지 않았다.

토큰 축은 현재 형태로는 도움이 되지 않는다 (3.6절). 살리려면 드리프트를 예측하는 student가
먼저 필요하다.

### ~~5.3~~ (완료) — layer 축 × 토큰 축 결합

layer-split은 얼린 구간에서 프롬프트를 아예 빼지만, 얼리지 않은 깊은 layer에서는 여전히
**960개 전부**를 흘린다. 그런데 4.5절에서 오차의 86%가 상위 25% 토큰에 몰려 있으므로,
깊은 layer에서도 전부 갱신할 필요가 없다.

```
layer <  fl        : 프롬프트 forward 없음, 프리필 K/V 사용
layer >= fl        : 상위 K개만 forward, 나머지 (960−K)는 프리필 K/V 사용
attention key      : [갱신된 K] + [동결된 960−K] + [suffix 128]
```

두 축이 곱해진다. `fl=16, K=240`이면

```
16×128 + 16×(240+128) = 7,936   vs   dynkv 34,816   →  77% 절감
```

fl=16 단독(19,456)의 절반 이하이며, dLLM-Cache 속도를 넘길 여지가 생긴다.

선택 기준은 현재 student의 중요도 점수를 쓴다(`weighted`와 상관 0.727). 드리프트를 직접
예측하는 2채널 student는 그 다음 단계다.

### 5.4 이후 — 드리프트 기반 선택 갱신 (student 재학습)

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
