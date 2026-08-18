# Attention–Delta 누적 Teacher와 추론 방법

## 1. 개요

현재 방법은 생성 trajectory 전체를 관찰하여 두 종류의 prompt-token teacher score를 만든다.

1. **Attention teacher**: 각 active mask 위치가 확정될 때까지 반복해서 참조한 prompt 위치
2. **Delta teacher**: 생성 과정에서 K/V가 누적해서 많이 변한 prompt 위치

두 teacher로 별도의 student를 학습하고, 추론 시 다음 순서로 사용한다.

```text
전체 prompt
  -> Attention student top 960 선택
  -> 선택되지 않은 prompt 토큰 제거
  -> Delta student top 480을 최초 한 번 선택
  -> 동일한 480개의 K/V를 모든 denoising step에서 업데이트
```

현재 핵심 추론 설정은 다음과 같다.

| 항목 | 설정 |
|---|---:|
| Attention keep budget | 960 |
| Delta refresh budget | 480 |
| Delta 선택 | `select_once=True` |
| Refresh 주기 | `refresh_interval=1` |
| Frozen layer | `0` |
| 평가 block length | `32` |

---

## 2. 표기

- $p$: prompt 토큰 위치
- $t$: denoising step
- $l$: transformer layer
- $h$: attention head
- $M_t$: step $t$에서 현재 block 안에 아직 mask로 남아 있는 생성 위치 집합
- $P$: prompt 길이
- $T$: 전체 denoising step 수
- $L$: transformer layer 수
- $H$: attention head 수

Teacher 추출에서는 매 step 전체 `[prompt + generation canvas]`를 forward한다. 동일한 full-sequence forward에서 suffix-to-prompt attention과 prompt K/V 변화를 함께 측정한다.

---

## 3. Attention teacher

### 3.1 Suffix-to-prompt attention

생성 토큰 $j$가 prompt 토큰 $p$를 참조하는 attention을 head 평균한다.

$$
\alpha_{t,l,j,p}
=
\frac{1}{H}\sum_h
\operatorname{softmax}
\left(
\frac{Q_{t,l,h,j}K_{t,l,h,p}^{\top}}{\sqrt d}
\right)
$$

현재 block에서 아직 mask로 남아 있는 모든 위치의 attention을 합산한다.

$$
a_{t,l,p}
=
\sum_{j\in M_t}
w_{t,j}\alpha_{t,l,j,p}
$$

$w_{t,j}$는 위치 $j$에서 현재 예측한 token의 confidence이다. 위치 $j$는 자기 block이 활성화된 시점부터 실제 token으로 확정되는 step까지 매번 기여한다. 미래 block의 mask는 아직 decoding 대상이 아니므로 $M_t$에서 제외한다.

### 3.2 전체 lifetime 누적

모든 active-mask step의 attention을 시간축으로 합산한다.

$$
A_{l,p}
=
\sum_{t=1}^{T}a_{t,l,p}
=
\sum_{t=1}^{T}
\sum_{j\in M_t}
w_{t,j}\alpha_{t,l,j,p}
$$

현재 `active_top_k=0`이므로 teacher 추출 단계에서 prompt 위치를 top-k로 자르지 않는다. 모든 prompt 위치의 연속 score를 보존한다.

### 3.3 정규화

각 layer에서 prompt 방향 합이 1이 되도록 정규화한다.

$$
\hat A_{l,p}
=
\frac{A_{l,p}}
{\sum_{p'}A_{l,p'}+\epsilon}
$$

이 값이 teacher shard의 `teacher_norm`이며 Attention student의 학습 target이다.

> 2026-08-18 이전 teacher는 새로 확정되는 위치의 commit 직전 attention만 사용하고 step별 top 128 union/max를 적용했다. 현재 replacement teacher는 그 legacy 정의를 사용하지 않는다.

---

## 4. K/V 변화량 teacher

Prompt 표현은 bidirectional attention을 통해 현재 생성 suffix의 영향을 받는다. 따라서 같은 prompt 토큰이라도 denoising step이 진행되면서 layer별 K/V가 달라질 수 있다.

### 4.1 Stepwise relative K movement

$$
d^K_{t,l,p}
=
\frac{
\frac{1}{H}\sum_h
\left\|
K_{t,l,h,p}-K_{t-1,l,h,p}
\right\|_2
}{
\frac{1}{H}\sum_h
\left\|K_{t,l,h,p}\right\|_2+\epsilon
}
$$

### 4.2 Stepwise relative V movement

$$
d^V_{t,l,p}
=
\frac{
\frac{1}{H}\sum_h
\left\|
V_{t,l,h,p}-V_{t-1,l,h,p}
\right\|_2
}{
\frac{1}{H}\sum_h
\left\|V_{t,l,h,p}\right\|_2+\epsilon
}
$$

### 4.3 Stepwise K/V 평균

$$
\delta_{t,l,p}
=
\frac{d^K_{t,l,p}+d^V_{t,l,p}}{2}
$$

### 4.4 전체 trajectory 누적합

Prompt가 mask인 것이 아니라 generation canvas가 mask에서 점차 확정된다. 그 전체 trajectory 동안 prompt K/V의 stepwise movement를 합산한다.

$$
D_{l,p}
=
\sum_{t=2}^{T}\delta_{t,l,p}
$$

각 layer에서 prompt 방향 합이 1이 되도록 정규화한다.

$$
\hat D_{l,p}
=
\frac{D_{l,p}}
{\sum_{p'}D_{l,p'}+\epsilon}
$$

이 값이 `delta_norm`이며 Delta student의 학습 target이다.

> Delta teacher에는 Attention teacher와 같은 step별 top-k union이 없다. 모든 prompt 토큰의 K/V 변화량을 전체 trajectory 동안 합산한다. `delta_step_max`도 저장되지만 진단용이며, 현재 student는 누적합을 정규화한 `delta_norm`을 학습한다.

---

## 5. Student 모델

Attention student와 Delta student는 같은 구조를 사용하지만 서로 다른 target으로 별도 학습한다.

Layer $l$의 prompt token hidden state를 $h_{l,p}$라고 하고, 마지막 128개 question token hidden state의 평균을 $\bar h_{l,Q}$라고 한다.

$$
\bar h_{l,Q}
=
\frac{1}{|Q|}\sum_{q\in Q}h_{l,q}
$$

Token feature와 question feature를 각각 projection한 뒤, 두 feature와 elementwise interaction을 MLP에 입력한다.

$$
z_{l,p}
=
\operatorname{MLP}_l
\left(
W_t h_{l,p},
W_q\bar h_{l,Q},
(W_t h_{l,p})\odot(W_q\bar h_{l,Q})
\right)
$$

Prompt 위치 방향 softmax score는 다음과 같다.

$$
\pi_{l,p}=\operatorname{softmax}_p(z_{l,p})
$$

두 student의 target은 다음과 같다.

| Student | Target |
|---|---|
| Attention student | $\hat A_{l,p}$ (`teacher_norm`) |
| Delta student | $\hat D_{l,p}$ (`delta_norm`) |

---

## 6. Student loss

현재 두 student 모두 동일한 복합 loss를 사용한다.

$$
\mathcal L
=
\operatorname{MSE}(\pi,y)
+0.1\mathcal L_{\mathrm{rank}}
+0\times\mathcal L_{\mathrm{topk}}
$$

Ranking target은 teacher score 상위 20%와 하위 40%이다. 상위 항목의 예측값이 하위 항목보다 margin $m=0.05$ 이상 크도록 학습한다.

$$
\mathcal L_{\mathrm{rank}}
=
\frac{1}{|\mathcal T||\mathcal B|}
\sum_{i\in\mathcal T}
\sum_{j\in\mathcal B}
\max(0,\pi_j-\pi_i+0.05)
$$

- $\mathcal T$: teacher 상위 20%
- $\mathcal B$: teacher 하위 40%
- `topk_weight=0`

따라서 학습 loss에는 128, 480, 960 같은 고정 budget이 들어가지 않는다. Student는 연속 score와 상대적 순위를 학습하며, 960/480 budget은 추론에서만 적용한다.

---

## 7. 추론

### 7.1 Attention top 960

먼저 전체 입력 prompt를 full forward하여 Attention student의 layer별 score를 얻는다. 현재 `student_score_activation=softmax`이다.

Layer 평균 score:

$$
s^A_p
=
\frac{1}{L}\sum_{l=1}^{L}\pi^A_{l,p}
$$

상위 960개 prompt 위치를 선택한다.

$$
\mathcal K
=
\operatorname{TopK}_{960}(s^A)
$$

선택되지 않은 prompt 토큰은 generation sequence에서 제거하고, $\mathcal K$의 960개 토큰만 유지한다.

### 7.2 축약 prompt prefill

선택한 960개 prompt token과 masked generation canvas를 함께 prefill한다.

```text
[kept prompt 960 | masked generation canvas]
```

이 과정에서 960개 prompt token의 layer별 hidden state와 served K/V cache를 만든다.

### 7.3 Delta top 480

Delta student는 축약된 960개 prompt의 현재 served hidden state를 평가한다.

$$
s^D_p
=
\frac{1}{L}\sum_{l=1}^{L}\pi^D_{l,p},
\qquad p\in\mathcal K
$$

960개 안에서 상위 480개를 선택한다.

$$
\mathcal R
=
\operatorname{TopK}_{480}
\left(s^D\mid_{\mathcal K}\right)
$$

현재 `delta_select_once=True`이므로 $\mathcal R$은 최초 refresh step에서 한 번만 선택하고 생성이 끝날 때까지 고정한다.

### 7.4 Interval 1 refresh

현재 설정은 다음과 같다.

```text
delta_select_once=True
student_refresh_interval=1
student_drift_frozen_layers=0
```

따라서 모든 denoising step과 모든 transformer layer에서 선택된 480개의 hidden state와 K/V를 현재 suffix 상태에 맞춰 업데이트한다.

$$
(K^{\mathrm{serve}}_{t,l,p},V^{\mathrm{serve}}_{t,l,p})
\leftarrow
(K_{t,l,p},V_{t,l,p}),
\qquad p\in\mathcal R
$$

나머지 480개는 prefill cache를 계속 사용한다.

$$
(K^{\mathrm{serve}}_{t,l,p},V^{\mathrm{serve}}_{t,l,p})
=
(K^{\mathrm{prefill}}_{l,p},V^{\mathrm{prefill}}_{l,p}),
\qquad p\in\mathcal K\setminus\mathcal R
$$

매 step suffix는 다음 960개 전체 served cache에 attention한다.

```text
480개: 모든 step에서 K/V 업데이트
+
480개: prefill K/V 유지
=
총 960개 prompt K/V 제공
```

그 후 현재 block에서 confidence가 높은 생성 토큰을 확정하고 다음 denoising step으로 진행한다.

---

## 8. Top-k 값의 역할 구분

| 값 | 적용 시점 | 역할 |
|---:|---|---|
| 0 | Attention teacher 추출 | `active_top_k=0`: teacher-side top-k 비활성화 |
| 960 | 추론 | 실제로 남길 prompt 토큰 수 |
| 480 | 추론 | 매 step K/V를 갱신할 prompt 토큰 수 |

Teacher는 모든 prompt 위치의 연속 score를 저장하고, budget은 추론의 960/480에서만 적용한다.

---

## 9. Teacher 추출과 평가 trajectory 설정

### 9.1 Teacher 추출

현재 chat-template offline hybrid teacher shard의 대표 설정:

| 항목 | 값 |
|---|---:|
| Generation length | 128 |
| Steps | 128 |
| Block length | 8 |
| Attention query mode | active-block lifetime mask |
| Attention active top-k | 0 (비활성화) |
| Attention aggregation | sum |
| Chat template | 적용 |

### 9.2 LongBench 평가

LongBench 평가는 다음 규칙을 사용한다.

```text
block_length = 32
steps = gen_length
```

| Generation length | Block 수 | Block당 step |
|---:|---:|---:|
| 32 | 1 | 32 |
| 64 | 2 | 32 |
| 128 | 4 | 32 |
| 512 | 16 | 32 |

예를 들어 QMSum은 `gen_length=512`, `steps=512`, `block_length=32`이므로 16개 block에서 각각 32 step을 수행한다. `refresh_interval=1`이므로 고정된 top 480을 총 512번 업데이트한다.

최대 sequence length는 2048이다. 따라서 generation length가 $G$이면 prompt는 최대 $2048-G$ 토큰으로 left truncation된다.

| Generation length | 최대 prompt 길이 |
|---:|---:|
| 32 | 2016 |
| 64 | 1984 |
| 128 | 1920 |
| 512 | 1536 |

---

## 10. 완료된 legacy Attention-template ablation

아래 비교는 2026-08-18 이전 commit-only/top128 Attention teacher로 완료한 실험이다. 결과는 보존하지만 해당 teacher shard는 현재 active teacher에서 제외했다.

### 방법 A

```text
Chat-template Attention student
+ Chat-template cumulative Delta student
```

### 방법 B

```text
No-template Attention student
+ Chat-template cumulative Delta student
```

공통 추론:

$$
\text{Attention top 960}
\rightarrow
\text{Delta top 480}
\rightarrow
\text{select once}
\rightarrow
\text{refresh every step}
$$

---

## 11. 전체 알고리즘 요약

### Teacher 구축

```text
for each denoising step t:
    full forward [prompt + suffix]

    Attention teacher:
        현재 block에서 아직 mask인 모든 위치 -> prompt attention 합산
        각 위치가 확정될 때까지 매 step 반복 누적
        teacher-side top-k 없이 모든 prompt 위치 score 유지

    Delta teacher:
        모든 prompt 토큰의 K/V 상대 변화량 측정
        stepwise K/V 변화량을 누적합

layer별 score 정규화
Attention target = teacher_norm
Delta target = delta_norm
```

### Student 추론

```text
1. 전체 prompt에서 Attention student score 계산
2. Layer 평균 score의 top 960 선택
3. 나머지 prompt 토큰 제거
4. [kept 960 + masked suffix] prefill
5. Delta student로 960개 중 top 480을 한 번 선택
6. 모든 denoising step에서:
       top 480 hidden/K/V 업데이트
       나머지 480은 prefill cache 유지
       suffix는 총 960개 served K/V에 attention
       confidence가 높은 생성 토큰 확정
```

---

## 12. 구현 파일

- Teacher 추출: `dllm_cache/budget/offline_hybrid_teacher.py`
- Teacher shard 생성: `dllm_cache/budget/extract_offline_hybrid_teacher.py`
- Student 모델 및 loss: `dllm_cache/budget/student_model.py`
- Student 학습 loop: `dllm_cache/budget/training_loop.py`
- Drift-refresh 추론: `dllm_cache/budget/drift_refresh_kv.py`
- 평가 model wrapper: `eval_model/LLaDA.py`
