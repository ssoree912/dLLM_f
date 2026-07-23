# Origin True-Full Results (2048)

기준 설정: origin LLaDA, `is_feature_cache=False`, `is_cfg_cache=False`, `max_length=2048`.
평가는 기존 lm-eval LongBench local task 설정을 사용했고, middle/no-chat 진단 결과는 제외했습니다.
최종 업데이트: 2026-07-23. 완료: 16/16.

| category | dataset | gen | n | metric | score | status |
|---|---|---:|---:|---|---:|---|
| Single-doc QA | qasper | 128 | 200 | qa_f1 | 0.3003 | done |
| Single-doc QA | multifieldqa_en | 64 | 150 | qa_f1 | 0.2686 | done |
| Single-doc QA | narrativeqa | 128 | 200 | qa_f1 | 0.1560 | done |
| Multi-doc QA | hotpotqa | 32 | 200 | qa_f1 | 0.1314 | done |
| Multi-doc QA | 2wikimqa | 32 | 200 | qa_f1 | 0.1361 | done |
| Multi-doc QA | musique | 32 | 200 | qa_f1 | 0.0665 | done |
| Summarization | gov_report | 512 | 200 | rouge | 0.2478 | done |
| Summarization | qmsum | 512 | 200 | rouge | 0.2034 | done |
| Summarization | multi_news | 512 | 200 | rouge | 0.2588 | done |
| Few-shot | trec | 64 | 200 | classification | 0.0250 | done |
| Few-shot | triviaqa | 32 | 200 | qa_f1 | 0.3419 | done |
| Few-shot | samsum | 128 | 200 | rouge | 0.3886 | done |
| Synthetic | passage_count | 32 | 200 | count | 0.0300 | done |
| Synthetic | passage_retrieval_en | 32 | 200 | retrieval | 0.2200 | done |
| Code | lcc | 64 | 500 | code_sim | 0.6448 | done |
| Code | repobench-p | 64 | 500 | code_sim | 0.5909 | done |

결과 JSON 기준:

| dataset | result json |
|---|---|
| qasper | `experiment/345/2026-07-15/results/llada_true_full_qasper_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T16-21-14.103378.json` |
| multifieldqa_en | `experiment/345/2026-07-15/results/llada_true_full_multifieldqa_en_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T11-57-43.946507.json` |
| narrativeqa | `experiment/345/2026-07-15/results/llada_true_full_narrativeqa_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T17-55-16.181492.json` |
| hotpotqa | `experiment/345/2026-07-15/results/llada_true_full_hotpotqa_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-20T17-15-15.411957.json` |
| 2wikimqa | `experiment/345/2026-07-15/results/llada_true_full_2wikimqa_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-20T16-50-55.010716.json` |
| musique | `experiment/345/2026-07-15/results/llada_true_full_musique_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T11-22-11.457821.json` |
| gov_report | `experiment/345/2026-07-15/results/llada_true_full_gov_report_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-23T12-25-55.482449.json` |
| qmsum | `experiment/345/2026-07-15/results/llada_true_full_qmsum_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-23T00-29-32.716319.json` |
| multi_news | `experiment/345/2026-07-15/results/llada_true_full_multi_news_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-23T06-16-53.845100.json` |
| trec | `experiment/345/2026-07-15/results/llada_true_full_trec_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T14-47-50.913717.json` |
| triviaqa | `experiment/345/2026-07-15/results/llada_true_full_triviaqa_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-20T17-40-21.983311.json` |
| samsum | `experiment/345/2026-07-15/results/llada_true_full_samsum_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-20T21-07-31.112632.json` |
| passage_count | `experiment/345/2026-07-15/results/llada_true_full_passage_count_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T10-29-49.013582.json` |
| passage_retrieval_en | `experiment/345/2026-07-15/results/llada_true_full_passage_retrieval_en_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T10-57-40.990075.json` |
| lcc | `experiment/345/2026-07-15/results/llada_true_full_lcc_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-20T19-34-27.617888.json` |
| repobench-p | `experiment/345/2026-07-15/results/llada_true_full_repobench-p_mlen2048_full/__mnt__srv__home__dlpcg.325__dllm__model__LLaDA-8B-Instruct/results_2026-07-21T13-56-33.275299.json` |
