# C8 long-context reader - step 01 timing: OK

Written 2026-11-11 13:05:47; commit 5729ed463eeb; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; 807 s.

Server: `Qwen/Qwen2.5-7B-Instruct-1M`, mode **native** (dca = dual chunk attention backend, V0 engine; native = default backend, window capped at the native 262,144 tokens), --max-model-len 262,144; log `/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/vllm_logs/server_native_20261111_125044.log`.

From the server log: vLLM None; KV cache capacity 2,314,064 tokens; max concurrency 8.83x at 262,144 tokens.


## Estimate of the full run (upper bound: every request alone)

| Dataset | Queries | Full doc runnable | Documents | Uncached reads (h) | Cached questions (h) | SAADI + Vanilla + closed-book (h) | Wall (h) | GPU h | > 6.0 GPU h | Full doc without cache (GPU h) | Prefix cache works |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 900 | 900 | 45 | 0.14 | 0.10 | 0.31 | 0.55 | 2.2 | False | 12 | True |
| NovelHopQA | 908 | 526 | 10 | 0.18 | 0.15 | 0.78 | 1.11 | 4.4 | False | 37 | True |

Policy over budget: C8_ON_OVER_BUDGET=run.

Fits: uncached seconds = a*L + b*L^2, cached seconds = c + d*L (L = prompt tokens):

| Dataset | a (s/token) | b (s/token^2) | points | c (s) | d (s/token) | points | cached / uncached (median, docs >= 20K tokens) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 0.000171 | 1.26e-09 | 17 | 0.41 | 0 | 20 | 0.028 |
| NovelHopQA | 0.000208 | 5.54e-10 | 8 | 0.189 | 4.5e-06 | 10 | 0.017 |

## Single-request latency of the sample (seconds)

| Dataset | Condition | Kind | Requests | Mean s | Median s | Max s | Mean prompt tokens | Mean output tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Closed-book | single | 20 | 0.34 | 0.24 | 1.21 | 81 | 14.2 |
| MuSiQue-32K | Full document | cached | 3 | 0.49 | 0.41 | 0.79 | 48,574 | 16.7 |
| MuSiQue-32K | Full document | cached_repeat | 17 | 0.40 | 0.32 | 0.87 | 48,879 | 11.8 |
| MuSiQue-32K | Full document | uncached | 17 | 11.36 | 11.29 | 12.19 | 48,879 | 10.9 |
| MuSiQue-32K | SAADI top 10 | single | 20 | 0.44 | 0.36 | 1.12 | 1,391 | 12.3 |
| MuSiQue-32K | Vanilla BM25 top 10 | single | 20 | 0.48 | 0.41 | 0.95 | 1,402 | 11.9 |
| NovelHopQA | Closed-book | single | 20 | 0.19 | 0.17 | 0.36 | 81 | 9.4 |
| NovelHopQA | Full document | cached | 2 | 0.79 | 0.79 | 0.83 | 146,721 | 7.5 |
| NovelHopQA | Full document | cached_repeat | 8 | 1.09 | 1.11 | 1.51 | 196,716 | 14.4 |
| NovelHopQA | Full document | uncached | 8 | 63.24 | 65.95 | 88.37 | 196,716 | 14.5 |
| NovelHopQA | SAADI top 10 | single | 20 | 1.32 | 1.38 | 1.90 | 6,567 | 12.3 |
| NovelHopQA | Vanilla BM25 top 10 | single | 20 | 1.58 | 1.54 | 1.99 | 6,560 | 13.4 |

Full document, per sampled document:

| Dataset | Document | Prompt tokens | Uncached s | Same question, cached s | Cached tokens reported | Cache reset ok |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | group_004 | 47,444 | 10.8 | 0.26 | 47,440 | True |
| MuSiQue-32K | group_006 | 47,537 | 10.9 | 0.60 | 47,536 | True |
| MuSiQue-32K | group_019 | 47,854 | 11.0 | 0.30 | 47,840 | True |
| MuSiQue-32K | group_009 | 48,062 | 11.3 | 0.52 | 48,048 | True |
| MuSiQue-32K | group_034 | 48,128 | 11.0 | 0.27 | 48,112 | True |
| MuSiQue-32K | group_025 | 48,345 | 11.2 | 0.36 | 48,336 | True |
| MuSiQue-32K | group_001 | 48,402 | 11.4 | 0.53 | 48,400 | True |
| MuSiQue-32K | group_010 | 48,460 | 11.3 | 0.41 | 48,448 | True |
| MuSiQue-32K | group_024 | 48,623 | 11.2 | 0.29 | 48,608 | True |
| MuSiQue-32K | group_008 | 48,687 | 11.2 | 0.29 | 48,672 | True |
| MuSiQue-32K | group_020 | 49,046 | 11.3 | 0.26 | 49,040 | True |
| MuSiQue-32K | group_026 | 49,203 | 11.5 | 0.45 | 49,200 | True |
| MuSiQue-32K | group_031 | 49,667 | 12.2 | 0.87 | 49,664 | True |
| MuSiQue-32K | group_038 | 49,952 | 11.5 | 0.26 | 49,936 | True |
| MuSiQue-32K | group_044 | 50,149 | 11.6 | 0.32 | 50,144 | True |
| MuSiQue-32K | group_016 | 50,524 | 11.9 | 0.46 | 50,512 | True |
| MuSiQue-32K | group_005 | 50,866 | 11.8 | 0.26 | 50,864 | True |
| NovelHopQA | B24 | 133,488 | 37.7 | 0.88 | 133,472 | True |
| NovelHopQA | B43 | 159,934 | 47.3 | 0.82 | 159,920 | True |
| NovelHopQA | B37 | 167,777 | 50.5 | 0.79 | 167,776 | True |
| NovelHopQA | B27 | 201,906 | 65.0 | 1.51 | 201,904 | True |
| NovelHopQA | B34 | 206,909 | 66.9 | 1.20 | 206,896 | True |
| NovelHopQA | B04 | 209,557 | 68.0 | 1.09 | 209,552 | True |
| NovelHopQA | B12 | 240,625 | 82.1 | 1.12 | 240,624 | True |
| NovelHopQA | B26 | 253,535 | 88.4 | 1.29 | 253,520 | True |

NovelHopQA: 10 sampled queries skipped for the full document: document beyond the native 262,144-token window and the server runs without dual chunk attention.

## Files written

- `results/rebuttal/c8_long_context/timing/timing.jsonl`
- `experiments/rebuttal/c8_long_context/reports/step_01_timing.md`

## Push

```
pushed dea82d8 to origin/fix/notebook-key-and-install: C8 all 01_timing: MuSiQue-32K est. 2.2 GPU h (upper bound); NovelHopQA est. 4.4 GPU h (upper bound)
```
