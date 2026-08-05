# SAMSum B=960 Online State-conditioned Distribution Distillation

## 0. 문서 상태

- 작성일: 2026-08-05
- 작업 브랜치: `a100/step_distll`
- 모델: `LLaDA-8B-Instruct`
- task: `SAMSum`
- 단일 budget: `B=960`
- sequence 길이: prompt 1920 + generation canvas 128 = 2048
- generation: `gen_length=128`, `steps=128`, `block_length=32`
- 현재 실행: train 256개, held-out validation 16개, 1 epoch
- 이 실험은 selector의 학습 가능성을 검증하는 correctness 실험이다. Sparse KV cache의
  latency 및 memory 절감 실험이 아니다.

## 1. 연구 질문

현재 확인할 질문은 하나다.

\[
\boxed{
B=960\text{ prompt pruning으로 현재 pruned state에서}
\quad p_{\mathrm{pruned}}\approx p_{\mathrm{full}}
\quad\text{를 학습할 수 있는가?}
}
\]

Attention score 자체를 정답으로 맞히는 것이 아니라, full prompt 모델의 token distribution과
commit 순서를 유지하는 것이 목표다. `B=480`, multi-budget conditioning, MMR, cache refresh
최적화는 이 실험 범위에 포함하지 않는다.

## 2. 데이터

준비된 파일은 다음과 같다.

```text
data/train/samsum_longbench_fewshot_2048_seed4090_n500_v64/
├── samsum_train_fewshot_2048.jsonl.xz       # 500개
└── samsum_validation_fewshot_2048.jsonl.xz  # 64개
```

현재 실행은 다음 subset을 사용한다.

```text
train      : train 파일의 앞 256개
validation : validation 파일의 앞 16개
```

train과 validation의 `sample_id`가 겹치면 학습 시작 전에 즉시 실패한다. Validation sample은
optimizer에 전달하지 않는다.

SAMSum prompt는 generation 128 token을 위해 최대 1920 token으로 왼쪽 truncation한다. Step 0의
state context에는 truncation 뒤 남은 공통 few-shot instruction 조각이 아니라, 현재 target의
`Dialogue ... Summary:` request span만 사용한다.

## 3. On-policy state

Step \(t\) 시작 시 실제 pruning rollout이 방문한 상태를 다음과 같이 둔다.

\[
x_t^\pi=(\text{prompt},\text{committed suffix},\text{remaining MASK})
\]

원래 full trajectory의 상태를 재생하지 않는다. 다음 상태는 항상 pruned forward의 token과
commit 위치로 만든다.

\[
x_{t+1}^\pi=\operatorname{Commit}(x_t^\pi,z_t^\pi)
\]

따라서 다음 step의 full teacher도 새 \(x_{t+1}^\pi\)에서 다시 계산된다. 이 구조가 저장된
full trajectory를 재생할 때 발생하는 stale-state 문제를 제거한다.

## 4. Online full teacher

현재 상태를 모든 prompt token과 함께 gradient 없이 실행한다.

\[
z_t^F=F(x_t^\pi;\mathbf 1_P)
\]

\[
p_{t,j}^F=\operatorname{softmax}\left(\frac{z_{t,j}^F}{\tau}\right)
\]

현재 기본값은 \(\tau=1\)이다. Full teacher logits는 suffix 128개 위치에 대해서만 materialize한다.
Loss 계산이 끝나면 logits를 버리며 teacher `.pt` shard로 저장하지 않는다.

Full과 pruned pass는 같은 수동 fp32 attention 연산자를 사용한다.

```text
full   : prompt gate = all-one
pruned : prompt gate = hard Top-960
```

따라서 두 attention path의 의도된 차이는 prompt gate뿐이다.

## 5. State-conditioned selector

Prompt-only forward를 sample당 한 번 수행해 각 layer 입력을 static prompt feature로 저장한다.

\[
r_{l,i}\in\mathbb R^H
\]

Step context는 forward 전에 알 수 있는 token ID의 embedding만 사용한다.

\[
c_0=\operatorname{Mean}_{j\in\text{target request}}E(x_j)
\]

\[
c_t=\operatorname{Mean}_{j:x_{t,j}^\pi\ne[\mathrm{MASK}]}E(x_{t,j}^\pi),
\qquad t>0
\]

현재 구현에서 \(c_t\)는 하나의 shared vector이며 layer별 hidden pool이 아니다. Selector는 다음
shared architecture를 사용한다.

\[
u_{l,i}=P_{tok}(r_{l,i}),\qquad v_t=P_c(c_t)
\]

\[
a_{t,l,i}=\operatorname{MLP}
\left([u_{l,i};v_t;u_{l,i}\odot v_t;e_l]\right)
\]

\[
M_{t,l}^\theta=\operatorname{Top960}_i(a_{t,l,i})
\]

기본 크기는 projection 256, MLP hidden 512이며 token/context projection과 score MLP를 모든
layer가 공유한다. Layer 차이는 learned layer embedding으로 표현한다.

## 6. Straight-Through Top-960

Hard Top-K는 미분할 수 없으므로 다음 straight-through mask를 사용한다.

\[
m^{ST}=m^{soft}+\operatorname{stopgrad}(m^{hard}-m^{soft})
\]

Prompt gate는 base attention softmax 뒤에 곱하고 다시 정규화한다.

\[
\widetilde A_{q,i}=
\frac{A_{q,i}m_i^{ST}}
{\sum_k A_{q,k}m_k^{ST}}
\]

Forward 값은 실제 hard Top-960 attention과 같고 backward에서는 현재 선택되지 않은 token의
selector score에도 gradient가 전달된다. Suffix key의 gate는 항상 1이다.

중요한 한계는 Top-960 적용 전에 전체 2048 x 2048 QK와 softmax를 계산한다는 점이다. 따라서
이 학습 경로는 정보 pruning을 구현하지만 실제 attention FLOPs를 줄이지 않는다.

## 7. Pruned forward와 loss

동일한 state에 student mask를 적용한다.

\[
z_t^\pi=F(x_t^\pi;M_t^\theta)
\]

\[
p_{t,j}^\pi=\operatorname{softmax}\left(\frac{z_{t,j}^\pi}{\tau}\right)
\]

아직 MASK인 생성 위치 집합을 \(U_t\), full teacher가 이번 step에 commit할 위치를
\(S_t^F\)라 한다. 기본 KD loss는 다음과 같다.

\[
\mathcal L_{KD}=
\frac{1}{|U_t|}\sum_{j\in U_t}
\left(1+\beta\mathbf 1[j\in S_t^F]\right)\tau^2
D_{KL}\left(p_{t,j}^F\Vert p_{t,j}^\pi\right)
\]

현재 \(\beta=2\)다. dLLM의 commit 순서를 보존하기 위해 active generation block 안에서
teacher commit 위치와 나머지 uncommitted 위치의 confidence를 ranking한다.

\[
\mathcal L_{commit}=
\operatorname{Mean}_{j\in S_t^F,k\in U_t\setminus S_t^F}
\max\left(0,m-(q_{t,j}^\pi-q_{t,k}^\pi)\right)
\]

\[
\mathcal L=\mathcal L_{KD}+0.1\mathcal L_{commit},
\qquad m=0.1
\]

학습 해석을 위해 commit 가중치와 \(\tau^2\) scaling을 제외한 unweighted diagnostic KL도
별도로 기록한다.

## 8. 실제 rollout 순서

```text
현재 x_t^pi
  -> all-one prompt gate full forward (no-grad)
  -> full distribution와 teacher commit 위치 계산
  -> causal c_t와 layer별 Top-960 계산
  -> 동일 x_t^pi에서 pruned forward
  -> KD + commit ranking backward
  -> gradient norm 1.0으로 clipping 후 AdamW update
  -> pruned logits의 token/position을 commit
  -> x_(t+1)^pi
```

SAMSum 설정은 generation block 4개, block당 32 token, block당 32 denoising step이다. 현재
schedule에서는 매 step active block에서 한 위치를 commit한다.

## 9. 학습 설정

```yaml
model: LLaDA-8B-Instruct
dtype: bfloat16
max_length: 2048
prompt_length: 1920
gen_length: 128
steps: 128
block_length: 32
budget: 960

student:
  projection_dim: 256
  mlp_dim: 512

optimization:
  optimizer: AdamW
  learning_rate: 1.0e-4
  epochs: 1
  max_grad_norm: 1.0
  distill_temperature: 1.0
  teacher_commit_weight: 2.0
  commit_loss_weight: 0.1
  commit_margin: 0.1
  gate_temperature: 1.0
  seed: 4090
```

Batch size는 trajectory dependency와 A100 40GB memory 제약 때문에 sample 1개다. 각
denoising step에서 backward와 optimizer update를 끝낸 뒤 다음 state로 이동한다. Pruned
forward는 decoder block activation checkpointing을 사용한다.

## 10. 평가 및 기록 지표

매 step 다음 값을 기록한다.

- weighted KD training loss
- unweighted full-to-pruned diagnostic KL
- token top-1 agreement
- full/pruned commit-position Jaccard
- clipping 전 selector gradient norm

완전한 128-step rollout에서는 다음 downstream/길이 지표도 기록한다.

- EOS/EOT 이전 prediction과 raw canvas prediction
- EOS-truncated ROUGE-L F1
- raw ROUGE-L F1
- LCS recall
- first-stop position
- stop 이전 token 수
- trailing token 수

F1 상승이 단순한 출력 단축 때문인지 분리하기 위해 truncated/raw 결과와 LCS recall을 함께
사용한다.

## 11. 실행 명령

```bash
cd /home/M2026107/dllm/dLLM-Cache-step-distill/experiment/4090/2026-08-03

PYTHONPATH=. uv run \
  --python /home/M2026107/.conda/envs/dave-llada/bin/python \
  --with 'pydantic>=2,<3' \
  python -m step_distill.train_distribution_student \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --train-data /home/M2026107/dllm/data/train/samsum_longbench_fewshot_2048_seed4090_n500_v64/samsum_train_fewshot_2048.jsonl.xz \
  --validation-data /home/M2026107/dllm/data/train/samsum_longbench_fewshot_2048_seed4090_n500_v64/samsum_validation_fewshot_2048.jsonl.xz \
  --output-dir /home/M2026107/.cache/step_distill_distribution_b960_samsum_train256_val16_a100_20260805 \
  --train-limit 256 \
  --validation-limit 16 \
  --epochs 1
```

현재 tmux session은 다음과 같다.

```text
samsum_b960_train256
```

## 12. 산출물

```text
/home/M2026107/.cache/step_distill_distribution_b960_samsum_train256_val16_a100_20260805/
├── train.log
├── metrics.jsonl
└── checkpoint.pt  # train과 validation이 모두 끝난 뒤 atomic save
```

Teacher logits와 distribution은 일시적으로만 존재하며 별도 teacher 파일로 저장하지 않는다.
기존 SAMSum attention teacher shard는 이 학습의 입력이 아니다.

현재 구현은 sample별 metric은 즉시 flush하지만 checkpoint는 전체 train/validation 종료 후에만
저장한다. 따라서 SSH disconnect는 tmux가 보호하지만 서버 재부팅에 대한 mid-run weight resume은
지원하지 않는다.

## 13. 초기 실행 확인

첫 train sample의 128-step rollout은 정상 완료됐다.

| 항목 | 값 |
|---|---:|
| loss | 0.26295 |
| diagnostic KL | 0.25256 |
| token top-1 agreement | 0.6265 |
| commit Jaccard | 0.7734 |
| A100 peak observed memory | 약 21.8GB |
| sample 처리시간 | 약 4분 |

Clipping 전 gradient norm은 일부 step에서 매우 크게 나타났지만 `max_grad_norm=1.0`을 적용한
상태이며 첫 sample의 모든 loss는 finite였다. 전체 안정성은 256개 trajectory가 끝난 뒤 loss와
gradient 분포로 다시 판정한다.

## 14. 완료 후 비교

동일한 SAMSum validation sample ID에서 다음을 비교한다.

1. Full prompt
2. Static attention B=960
3. Offline stored per-step attention B=960
4. Online same-state attention B=960
5. Distribution-distilled student B=960

주 판정은 downstream score 하나가 아니라 다음을 함께 사용한다.

- diagnostic KL 감소
- token top-1 agreement
- commit-position Jaccard
- LCS recall
- EOS-truncated/raw ROUGE-L 차이
- 출력 길이와 trailing token 수

## 15. 현재 한계

- ST 학습은 dense QK를 계산하므로 속도 및 memory 절감 결과가 아니다.
- 매 step full teacher를 다시 계산하므로 teacher extraction 파일은 줄지만 GPU 계산량은 크다.
- 현재 \(c_t\)는 token embedding mean이므로 token 순서, 위치, commit 개수를 직접 표현하지 않는다.
- B=960 하나만 지원하며 B=480 또는 multi-budget 일반화는 아직 평가하지 않는다.
- Physical sparse KV gather와 refresh interval 최적화는 student 품질이 확인된 뒤 수행한다.
