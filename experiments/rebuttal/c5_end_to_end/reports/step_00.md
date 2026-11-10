# C5 end-to-end QA - step 00 (prepare): FAILED (unexpected error)

Written 2026-11-10 19:24:36; commit 2db993ff6678; host eee-s132-infsys-gpu03; exit code 1.

Command: `bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh `

## Summary

- Unexpected error: ModuleNotFoundError: No module named 'pandas'

## Fixed settings

- Base retriever: BM25 (the retriever of the paper's agent comparison, Table 2; its SAADI inputs are the bm25 `retrieval_depth` rankings of outputs/saadi/reference).
- SAADI: published point (lambda_M, lambda_S, mu_A, mu_D) = (0.5, 0.3, 1.0, 1.0) from `/home/iataghav/data/SAADI/configs/methods/sadi.yaml` (lambda_T = 1, z-score, max aggregation).
- Reader: `Qwen/Qwen3-30B-A3B-Instruct-2507`, vLLM, temperature 0, seed 42, max output tokens 64, max model length 32768. Tokenizer for all counts: the reader's.
- One prompt for every condition: a single user message, no system prompt, chat template of the model. The exact template:

```text
Answer the question using the context passages below. If the passages do not contain the answer, answer from your own knowledge.

Context:
{context}

Question: {question}
Answer with a short phrase.
```

`{context}` is the passages in rank order, each `[i] <text>` (whitespace collapsed), separated by a blank line; closed-book gets `(no passages)`. Scoring: SQuAD normalisation (lower case, no punctuation, no articles), EM and token F1 as the max over the gold answers, containment = a normalised gold answer inside the normalised output. The output is first cut to its first non-empty line and a leading `Answer:` is dropped.

## MuSiQue-32K

Queries 900, documents 45, chunks 17102; gold answers: outputs/musique_ablation/32K_expansion/prepared/qa.json.
MuSiQue answer aliases: {"used": false, "reason": "RuntimeError: datasets package is required for MuSiQue loading. Install with `pip install .[datasets]`."}.

Saved comparison outputs:

| Method | Status | C5 queries covered | Checks | Source |
| --- | --- | --- | --- | --- |

## NovelHopQA

Queries None, documents None, chunks None; gold answers: None.

Saved comparison outputs:

| Method | Status | C5 queries covered | Checks | Source |
| --- | --- | --- | --- | --- |

## Comparison-method settings (defaults; nothing tuned)


## What is missing and what running it would cost

| Dataset | Item | Status | Estimated wall hours (4 GPUs) | GPU hours | Basis |
| --- | --- | --- | --- | --- | --- |

## Checks of unchanged inputs

- docs/preserved_sources.json: 93 files checked, changed 0, missing 0 [] 
- configs/methods/sadi.yaml sha256 18846ef24b6075586cc4ed53e1e8b6f7a916f0253e5932aa203b0ec741bbfdf0

## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c5_end_to_end/scripts/step00_prepare.py", line 995, in main
    reproduce(ds, paths, env, point, d)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c5_end_to_end/scripts/step00_prepare.py", line 316, in reproduce
    qs = {q.query_id: q for q in load_query_arrays(stream, queries, "bm25")}
                                 ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 767, in load_query_arrays
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
```

## Files written

- `experiments/rebuttal/c5_end_to_end/reports/step_00.md`
- `experiments/rebuttal/c5_end_to_end/reports/step_00.json`

## Push

```
NOT PUSHED: git commit failed:
error: pathspec 'results/rebuttal/c5_end_to_end' did not match any file(s) known to git
```
