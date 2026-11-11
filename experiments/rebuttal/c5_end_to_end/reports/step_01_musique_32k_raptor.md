# C5 end-to-end QA - step 01 RAPTOR re-run, MuSiQue-32K: OK

Written 2026-11-11 01:32:59; commit c492f48ab0c5; host eee-s132-infsys-gpu03; GPUs 0,1,2,3; 19814 s.

Command: `bash experiments/rebuttal/c5_end_to_end/sh/run_gpus_0123.sh` (stage raptor)

## What was run

- RAPTOR repository `/mnt/data/zeinab/SAADI_Raptor_c5` (not modified), Python `/mnt/data/zeinab/raptor/.venv/bin/python3`.
- Input: SAADI's prepared MuSiQue-32K files copied unchanged into `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/data/musique_32k_c5`, with the manifest RAPTOR's loader needs (sha256 2998cd97bbb0c8bd6f8333bb8fa54b96d77cba6ce2e9c3aac1e311b824ea5166).
- Config `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/configs/musique_32k_c5_retrieval.yaml`: RAPTOR's saved musique_32k config; changed only: loader.data_dir: 'data/musique_32k' -> '/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/data/musique_32k_c5'; loader.expected_qa_entries: 300 -> 900; loader.expected_prepared_documents: 15 -> 45; loader.expected_manifest_sha256: '57bc46b1fdec9ff104f5df4b0f58745a5b620d8c783e93a8025effb8a502b8f4' -> '2998cd97bbb0c8bd6f8333bb8fa54b96d77cba6ce2e9c3aac1e311b824ea5166'.
- Settings (saved = RAPTOR defaults except the models): leaf 100 cl100k tokens, 5 layers, threshold 0.5, top_k 5 per cluster, summaries up to 150 tokens, collapsed-tree retrieval top_k 10 with max_tokens 3500, Contriever embeddings, Qwen3-30B-A3B-Instruct-2507 summariser (vLLM in-process, TP 4, temperature 0).
- Run status before: partial; after: complete.

## Result

| Queries | Without labels | Unmatched | Build time (s, documents) |
| --- | --- | --- | --- |
| 900 | 0 | 0 | {"documents": 45, "total": 19537.749827} |

RAPTOR's own upper-bound hit rates: `{"gold": {"hr5": 0.5424954792043399, "hr10": 0.650994575045208}, "silver_strict": {"hr5": 0.23892773892773891, "hr10": 0.40326340326340326}, "union_strict": {"hr5": 0.5055555555555555, "hr10": 0.6555555555555556}}`

## Files written

- `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/raptor_10_runs/musique_32k/musique_32k_retrieval_ablation_raptor` (raw RAPTOR run, not pushed)
- `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/raptor_10_evaluations/musique_32k/musique_32k_retrieval_ablation_raptor/retrieval_upper_bound` (not pushed)
- `experiments/rebuttal/c5_end_to_end/reports/step_01_musique_32k_raptor.md`, `experiments/rebuttal/c5_end_to_end/reports/step_01_musique_32k_raptor.json`
