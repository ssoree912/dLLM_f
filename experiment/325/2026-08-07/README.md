# 실행 환경 및 실행 방법 (2026-08-07)

**이번 실험에서 실행한 코드는 전부 이 저장소(`dLLM-Cache`)에 있다.** 평가·pruning 모드·드리프트
측정이 모두 여기서 돈다.

`dLLM-Cache-step-distill`은 별도 저장소로, teacher 추출과 분포 증류 학습 전용이며 지금 평가
경로에는 관여하지 않는다. 두 저장소의 `eval_model/LLaDA.py`는 **동기화되어 있지 않으니**
평가는 반드시 이쪽에서 실행할 것.

## 0. 요약

| 하려는 일 | 실행 위치 | 파이썬 |
|---|---|---|
| **벤치마크 평가** (보고용 수치 전부) | `/home/M2026107/dllm/dLLM-Cache` | `.venv/bin/python` |
| **드리프트 측정** | `/home/M2026107/dllm/dLLM-Cache` | `.venv/bin/python` |
| (별도) teacher 추출 / 분포 증류 학습 | `/home/M2026107/dllm/dLLM-Cache-step-distill` | `uv run --python <conda>` |

---

## 1. 파이썬 환경

### 1.1 dLLM-Cache 전용 venv — 평가용 (주력)

```
경로: /home/M2026107/dllm/dLLM-Cache/.venv/bin/python
python 3.10.12 / torch 2.5.1+cu124 / transformers 4.46.3 / lm_eval 0.4.12
```

`lm-evaluation-harness`가 이 환경에만 설치되어 있다. **모든 벤치마크 수치는 이 환경에서
나와야 한다.** conda 환경에는 `lm_eval`이 없다.

`uv`나 `conda activate` 없이 절대경로로 바로 호출한다.

### 1.2 conda `dave-llada` — 학습/추출용

```
경로: /home/M2026107/.conda/envs/dave-llada/bin/python
python 3.10.20 / torch 2.6.0+cu124 / transformers 4.44.2
(머신에 있는 유일한 conda 환경)
```

step-distill 쪽 teacher 추출·분포 증류 학습에 쓴다. `uv run`으로 이 인터프리터를 지정하고
필요한 패키지를 얹는 방식이라 `conda activate`는 하지 않는다.

```bash
uv run --python /home/M2026107/.conda/envs/dave-llada/bin/python \
       --with 'pydantic>=2,<3' python -m step_distill.<모듈>
```

테스트를 돌릴 때는 `--with pytest`를 추가한다.

---

## 2. 공통 환경변수

```bash
export CUDA_VISIBLE_DEVICES=0
export MASKKV_ENABLED=0                                # MaskKV 경로 비활성화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 조각화 완화
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

허깅페이스 관련은 셸 프로필에 이미 잡혀 있다 (건드릴 필요 없음).

```
HF_HOME=/mnt/srv/home/SHARED/huggingface/cache
HF_ENDPOINT=http://repo.ai.gato:8090        # 사내 미러. gsm8k 등 신규 데이터셋 다운로드 가능
```

GPU는 A100-SXM4-40GB 1장. 동시에 두 작업을 띄우면 **OOM**이 난다 (실제로 발생했음).
반드시 순차 실행할 것.

---

## 3. 모델과 데이터 경로

```
모델          : /home/M2026107/dllm/model/LLaDA-8B-Instruct
LongBench     : /home/M2026107/dllm/data/longbench/*.jsonl        (16개 태스크, 각 200행)
few-shot 학습용: /home/M2026107/dllm/data/train/samsum_longbench_fewshot_2048_seed4090_n500_v64/
```

LongBench 태스크 정의는 `experiment/345/2026-07-15/tasks/longbench_local/*.yaml`.
공식 `lm_eval/tasks/longbench/` 정의와 내용이 같고 데이터 출처만 로컬 파일로 바꾼 것이다.
yaml의 `data_files.test` 경로는 2026-08-07에 이 머신 경로로 수정했다 (커밋 `1d9a381`).

---

## 4. 평가 실행 방법

### 4.1 기본형 (origin, pruning 없음)

```bash
cd /home/M2026107/dllm/dLLM-Cache

CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true \
.venv/bin/python evaluation_script.py \
  --model LLaDA \
  --tasks local_longbench_samsum \
  --include_path experiment/345/2026-07-15/tasks/longbench_local \
  --batch_size 1 --limit 200 \
  --model_args "pretrained=/home/M2026107/dllm/model/LLaDA-8B-Instruct,\
is_feature_cache=False,is_cfg_cache=False,max_length=2048" \
  --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0" \
  --num_fewshot 0 --log_samples \
  --apply_chat_template --fewshot_as_multiturn --trust_remote_code \
  --output_path /home/M2026107/.cache/<결과디렉토리>
```

`--include_path`가 있어야 `local_longbench_*` 태스크가 인식된다.
`--apply_chat_template --fewshot_as_multiturn`은 LongBench 공식 스크립트와 동일하게 유지한다.

### 4.2 pruning 모드별 `model_args` 추가 인자

student 경로는 **`/home/M2026107/dllm/dLLM-Cache` 기준 상대경로**로 준다.

```
STUDENT=results/budget/future_pool_student_train_300each_topk128_e10_lr2e-5_tw0.02/checkpoint-best
```

| 모드 | 추가 인자 |
|---|---|
| **kv_cache** (프리필 후 동결) | `student_path=$STUDENT,student_prompt_kv_cache=True,student_budget=960,student_question_window=128` |
| **dynkv** (시퀀스 축소 + 갱신) | `student_path=$STUDENT,student_prompt_dynamic_kv=True,student_budget=960,student_selection_mode=global,student_refresh_interval=1,student_question_window=128` |
| **prune** (마스킹, 진단용) | `student_path=$STUDENT,student_prompt_prune=True,student_budget=960,student_question_window=128` |
| **pool_active** | `student_path=$STUDENT,student_prompt_pool_active=True,student_pool_budget=960,student_budget=128,student_refresh_interval=1` |
| **dLLM-Cache 원본** | `prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False` |

`student_prompt_*` 계열은 **동시에 하나만** 켤 수 있다 (중복 시 에러).

주요 파라미터:

- `student_budget` — 남길 프롬프트 토큰 수. 프롬프트 1920 기준 960 = 50%, 480 = 25%
- `student_selection_mode` — `global`(layer 공통, 실제 시퀀스 축소) / `layer_union`
- `student_refresh_interval` — `1`이면 매 step 갱신(캐시 없음), 클수록 캐시에 가까움
- `student_question_window` — student가 질의 span으로 쓰는 프롬프트 끝 토큰 수 (기본 128)

### 4.3 결과 읽기

```bash
python3 -c "
import json,glob
f=glob.glob('/home/M2026107/.cache/<결과디렉토리>/**/results_*.json',recursive=True)[0]
r=json.load(open(f))['results']['local_longbench_samsum']
print('%.4f ± %.4f'%(r['rouge_score,none'], r['rouge_score_stderr,none']))
"
```

`--log_samples`를 켜면 같은 디렉토리에 샘플별 `.jsonl`이 남아 생성 결과를 직접 확인할 수 있다.

### 4.4 여러 설정 순차 실행

GPU 1장이므로 반드시 직렬로 돌린다. 실제로 쓴 방식:

```bash
cat > run.sh <<'SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd /home/M2026107/dllm/dLLM-Cache
export CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
run () {
  echo "=== START $1 $(date +%H:%M:%S)"
  .venv/bin/python evaluation_script.py ... --model_args "...,$2" --output_path "$R/$1" 2>&1 | tail -6
  echo "=== END $1 $(date +%H:%M:%S)"
}
run full "dummy_unused=0"
run dynkv "student_path=...,student_prompt_dynamic_kv=True,..."
SCRIPT
chmod +x run.sh && nohup ./run.sh > run.log 2>&1 &
```

**주의**: 대기 루프에 `pgrep -f "<패턴>"`을 쓰면 그 스크립트 자신의 명령줄이 패턴에 걸려
무한 대기한다 (실제로 50분 낭비했음). PID를 먼저 확보해 `while [ -d /proc/$PID ]`로 기다릴 것.

---

## 5. 드리프트 측정 실행

프롬프트 토큰이 step에 따라 얼마나 변하는지 측정한다.

```bash
cd /home/M2026107/dllm/dLLM-Cache

CUDA_VISIBLE_DEVICES=0 MASKKV_ENABLED=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m dllm_cache.budget.measure_prompt_drift \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --data /home/M2026107/dllm/data/longbench/samsum.jsonl \
  --output-dir /home/M2026107/.cache/prompt_drift_samsum_20260807 \
  --limit 8 --repeats 2
```

`--repeats 2`는 같은 프롬프트를 seed를 바꿔 두 번 생성한다. 드리프트 패턴이 재현되면
프롬프트만 보고 예측 가능하다는 뜻이고, 매번 다르면 생성 의존이라 student가 학습할 수 없다.

출력은 샘플·반복마다 `.pt` 하나이며 `[layer, prompt]` 텐서 4개를 담는다.

```
cumulative  mean_t ||v_t - v_0|| / ||v_0||        영구 동결 시 손해
stepwise    mean_t ||v_t - v_{t-1}|| / ||v_t||    갱신 후 낡는 속도
attention   mean_t sum_suffix softmax(q_s k_i)    suffix가 실제로 읽는 양
weighted    attention * cumulative                 실제 캐시 오차 기여
```

**sanity check**: `cumulative[0]`(layer 0)은 0이어야 한다. layer 0의 K/V는 임베딩과 위치만으로
결정되어 suffix와 무관하기 때문이다. 0이 아니면 측정이 잘못된 것이다.

---

## 6. 별도 저장소 step-distill 실행 (teacher 추출 / 학습)

아래는 **다른 저장소**의 코드다. 이번 평가에는 쓰이지 않았고, teacher를 새로 뽑거나
분포 증류 student를 학습할 때만 필요하다.

```bash
cd /home/M2026107/dllm/dLLM-Cache-step-distill/experiment/4090/2026-08-03

# teacher 추출
uv run --python /home/M2026107/.conda/envs/dave-llada/bin/python --with 'pydantic>=2,<3' \
  python -m step_distill.extract_teacher \
  --task samsum \
  --model /home/M2026107/dllm/model/LLaDA-8B-Instruct \
  --data /home/M2026107/dllm/data/train/samsum_longbench_fewshot_2048_seed4090_n500_v64/samsum_validation_fewshot_2048.jsonl.xz \
  --output-root /home/M2026107/.cache/<출력> \
  --max-length 2048 --gen-length 128 --block-length 32 --steps 128 \
  --temperature 0.0 --max-target-k 960 --diversity-gamma 0.0 \
  --sample-limit 16 --seed 4090

# 테스트
cd /home/M2026107/dllm/dLLM-Cache-step-distill
uv run --python /home/M2026107/.conda/envs/dave-llada/bin/python --with pytest --with 'pydantic>=2,<3' \
  python -m pytest experiment/4090/2026-08-03/tests -q
```

**`--diversity-gamma`에 주의**: 0이 아니면 MMR 탐욕 루프가 돌아 `max_target_k`에 비례해
극도로 느려진다. K=1920에 gamma=0.1로 2.5분에 shard 하나도 못 끝냈다. MMR 순서가 필요 없으면
반드시 `--diversity-gamma 0.0`으로 둘 것 (이 경우 `diverse_order == top_order`가 되므로
그 shard로 `*_diverse` 방법을 돌리면 안 된다).

**중요**: step-distill 쪽에는 더 이상 평가기가 없다 (커밋 `9f10e20`에서 제거). 보고용 수치는
전부 4절의 lm_eval 경로로 낸다. 남아 있는 `summary_metrics` / `generation_output`은 학습
진단용이며 벤치마크 점수가 아니다.

---

## 7. 산출물 위치

```
/home/M2026107/.cache/
├── lmeval_samsum_b960_20260807/          # 2026-08-07 평가 전체
│   ├── run.log                            # 실행 로그 (START/END 시각)
│   ├── run.sh ~ run7.sh                   # 실제 실행 스크립트
│   ├── full/  dynkv_300each/  dynkv_480/
│   ├── dynkv_960_refresh2/
│   ├── student_300each/  student_few_shot_1k/
│   └── dllmcache_official/
├── prompt_drift_samsum_20260807/          # 드리프트 측정
└── step_distill_teacher_*/                # teacher shard
```

실험 결과 해석은 같은 저장소의 `experiment/report/2026-08-07-samsum-prompt-budget.md` 참고.
