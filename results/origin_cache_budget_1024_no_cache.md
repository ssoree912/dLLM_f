# LongBench Results: Origin vs 128 Cache vs Budget 1024 No Cache

Updated: 2026-07-23 08:56 UTC.

- `origin`: origin LLaDA true-full, `is_feature_cache=False`, `is_cfg_cache=False`, `max_length=2048`.
- `128_cache`: previous student prompt-KV B=128 results.
- `budget_1024_no_cache`: student pool-active results, `student_pool_budget=1024`, `student_budget=128`, `is_feature_cache=False`.
- `-`: no completed result JSON found for that column yet.

| category | dataset | gen | n | metric | origin | 128_cache | budget_1024_no_cache |
|---|---|---:|---:|---|---:|---:|---:|
| Single-doc QA | qasper | 128 | 200 | qa_f1 | 0.3003 | 0.2172 | 0.2718 |
| Single-doc QA | multifieldqa_en | 64 | 150 | qa_f1 | 0.2686 | 0.2830 | 0.3166 |
| Single-doc QA | narrativeqa | 128 | 200 | qa_f1 | 0.1560 | 0.0750 | 0.1723 |
| Multi-doc QA | hotpotqa | 32 | 200 | qa_f1 | 0.1314 | 0.2330 | 0.1790 |
| Multi-doc QA | 2wikimqa | 32 | 200 | qa_f1 | 0.1338 | 0.1799 | 0.1676 |
| Multi-doc QA | musique | 32 | 200 | qa_f1 | 0.0665 | 0.1011 | 0.0886 |
| Summarization | gov_report | 512 | 200 | rouge | 0.2478 | 0.1650 | 0.1729 |
| Summarization | qmsum | 512 | 200 | rouge | 0.2034 | 0.2016 | 0.1916 |
| Summarization | multi_news | 512 | 200 | rouge | 0.2588 | 0.1937 | 0.2010 |
| Few-shot | trec | 64 | 200 | classification | 0.0250 | 0.0600 | 0.0075 |
| Few-shot | triviaqa | 32 | 200 | qa_f1 | 0.3419 | 0.6520 | 0.5373 |
| Few-shot | samsum | 128 | 200 | rouge | 0.3886 | 0.1771 | 0.3844 |
| Synthetic | passage_count | 32 | 200 | count | 0.0300 | 0.0300 | 0.0250 |
| Synthetic | passage_retrieval_en | 32 | 200 | retrieval | 0.2200 | 0.1850 | 0.2200 |
| Code | lcc | 64 | 500 | code_sim | 0.6448 | 0.2745 | 0.5222 |
| Code | repobench-p | 64 | 500 | code_sim | 0.5909 | 0.1653 | 0.5291 |

## Budget Progress

Completed: 16/16.

Done:

- qasper, multifieldqa_en, narrativeqa, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p

Pending:

- none
