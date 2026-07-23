# Task Data Map: Eval vs Train

작성 시점: 2026-07-22.

- `eval_data`: `experiment/2026-07-14/tasks/longbench_local/*.yaml`이 참조하는 `/home/M2026107/dllm/data/longbench` 파일.
- `train_data`: task별 student를 따로 만들 때 사용할 `/home/M2026107/dllm/data/train` 원본 파일.
- `2wikimqa`의 train 원본은 이름이 `2wikimultihopqa`로 되어 있어 별도 매핑했다.
- `hotpotqa` train 원본은 현재 `/home/M2026107/dllm/data/train/musique/hotpotqa` 아래에 저장되어 있다.
- `train_available=no`인 task는 현재 `/home/M2026107/dllm/data/train` 아래에 대응 train 파일이 없다.
- `repobench-p` train은 공식 RepoBench v1.1 HF 데이터셋(Python/Java, 3 settings)을 LongBench-compatible 형식으로 변환한 외부 train 후보다.

| category | dataset | eval_task | gen | eval_n | eval_data | train_available | train_dataset | train_n | train_data |
|---|---|---|---:|---:|---|---|---|---:|---|
| Single-doc QA | qasper | local_longbench_qasper | 128 | 200 | /home/M2026107/dllm/data/longbench/qasper.jsonl | yes | qasper | 2593 | /home/M2026107/dllm/data/train/qasper/qasper_train_longbench_format.jsonl |
| Single-doc QA | multifieldqa_en | local_longbench_multifieldqa_en | 64 | 150 | /home/M2026107/dllm/data/longbench/multifieldqa_en.jsonl | no | multifieldqa_en | - | - |
| Single-doc QA | narrativeqa | local_longbench_narrativeqa | 128 | 200 | /home/M2026107/dllm/data/longbench/narrativeqa.jsonl | yes | narrativeqa | 55003 | /home/M2026107/dllm/data/train/narrativeqa/narrativeqa_train_longbench_format.jsonl |
| Multi-doc QA | hotpotqa | local_longbench_hotpotqa | 32 | 200 | /home/M2026107/dllm/data/longbench/hotpotqa.jsonl | yes | hotpotqa | 90447 | /home/M2026107/dllm/data/train/musique/hotpotqa/hotpotqa_train_longbench_format.jsonl |
| Multi-doc QA | 2wikimqa | local_longbench_2wikimqa | 32 | 200 | /home/M2026107/dllm/data/longbench/2wikimqa.jsonl | yes | 2wikimultihopqa | 167454 | /home/M2026107/dllm/data/train/2wikimultihopqa/2wikimultihopqa_train_longbench_format.jsonl |
| Multi-doc QA | musique | local_longbench_musique | 32 | 200 | /home/M2026107/dllm/data/longbench/musique.jsonl | yes | musique | 19938 | /home/M2026107/dllm/data/train/musique/musique_train_longbench_format.jsonl |
| Summarization | gov_report | local_longbench_gov_report | 512 | 200 | /home/M2026107/dllm/data/longbench/gov_report.jsonl | yes | gov_report | 17457 | /home/M2026107/dllm/data/train/gov_report/gov_report_train_longbench_format.jsonl |
| Summarization | qmsum | local_longbench_qmsum | 512 | 200 | /home/M2026107/dllm/data/longbench/qmsum.jsonl | yes | qmsum | 1257 | /home/M2026107/dllm/data/train/qmsum/qmsum_train_longbench_format.jsonl |
| Summarization | multi_news | local_longbench_multi_news | 512 | 200 | /home/M2026107/dllm/data/longbench/multi_news.jsonl | yes | multi_news | 44972 | /home/M2026107/dllm/data/train/multi_news/multi_news_train_longbench_format.jsonl |
| Few-shot | trec | local_longbench_trec | 64 | 200 | /home/M2026107/dllm/data/longbench/trec.jsonl | yes | trec | 5452 | /home/M2026107/dllm/data/train/trec/trec_train_longbench_format.jsonl |
| Few-shot | triviaqa | local_longbench_triviaqa | 32 | 200 | /home/M2026107/dllm/data/longbench/triviaqa.jsonl | yes | triviaqa | 138384 | /home/M2026107/dllm/data/train/triviaqa/triviaqa_train_longbench_format.jsonl |
| Few-shot | samsum | local_longbench_samsum | 128 | 200 | /home/M2026107/dllm/data/longbench/samsum.jsonl | yes | samsum | 14731 | /home/M2026107/dllm/data/train/samsum/samsum_train_longbench_format.jsonl |
| Synthetic | passage_count | local_longbench_passage_count | 32 | 200 | /home/M2026107/dllm/data/longbench/passage_count.jsonl | no | passage_count | - | - |
| Synthetic | passage_retrieval_en | local_longbench_passage_retrieval_en | 32 | 200 | /home/M2026107/dllm/data/longbench/passage_retrieval_en.jsonl | no | passage_retrieval_en | - | - |
| Code | lcc | local_longbench_lcc | 64 | 500 | /home/M2026107/dllm/data/longbench/lcc.jsonl | no | lcc | - | - |
| Code | repobench-p | local_longbench_repobench-p | 64 | 500 | /home/M2026107/dllm/data/longbench/repobench-p.jsonl | yes | repobench-p | 48978 | /home/M2026107/dllm/data/train/repobench-p/repobench-p_train_longbench_format.jsonl |

## Summary

- Train 있음: 12/16 task
- Train 없음: 4/16 task
- Train 없는 task: `multifieldqa_en`, `passage_count`, `passage_retrieval_en`, `lcc`

## Suggested Per-Task Training Groups

각 task별 모델을 따로 만들 경우, `train_available=yes`인 task는 대응 train 파일로 teacher 추출/훈련을 진행하면 된다.
`train_available=no`인 task는 별도 train source를 확보하거나, 같은 category의 train 보유 task로 대체/공유할지 결정해야 한다.

| train_available | tasks |
|---|---|
| yes | `qasper`, `narrativeqa`, `hotpotqa`, `2wikimqa`, `musique`, `gov_report`, `qmsum`, `multi_news`, `trec`, `triviaqa`, `samsum`, `repobench-p` |
| no | `multifieldqa_en`, `passage_count`, `passage_retrieval_en`, `lcc` |

## Substitute Train Candidates

| target task | local exact train | usable local substitute | note |
|---|---|---|---|
| `passage_count` | no | generate from QA/multi-doc train contexts | Existing train passages can be sampled and duplicated to create count labels without eval leakage. |
| `passage_retrieval_en` | no | generate from QA/multi-doc train contexts | Existing train passages can be sampled, one passage abstract/snippet can become the query, and paragraph id becomes the label. |
| `lcc` | no | no local code train found | Needs an external code-completion corpus or a newly prepared repo/code dataset. |
| `repobench-p` | yes | `/home/M2026107/dllm/data/train/repobench-p/repobench-p_train_longbench_format.jsonl` | Converted from official RepoBench v1.1 Python/Java HF datasets; exact overlaps with local LongBench repobench-p eval were removed. |
