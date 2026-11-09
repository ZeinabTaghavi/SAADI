# 2.1 Probe baselines

**Verdict: oracle 100% HR@10 in cells 1-3; best real system SAADI run 1 (p1_ledger), live at 6%; SAADI run 1 6%; Vanilla 3%.**

## Setup
- Same probe as SAADI's runs: 10 books, 240 personas, 8 cells, the same chunks and the probe's own HR; one scorer for every system.
- Systems: B0 Vanilla BM25; B1 oracle (form 2 appended to every revealed persona's gold chunk); B2 Contextual Retrieval (whole book, Qwen3-30B at a 262K window); B3 HippoRAG v2 (package defaults); B4 LQCA (official code on 8-chunk segments).
- LLM for B2 and B3: Qwen3-30B-A3B-Instruct-2507, temperature 0, GPUs 4-7. LQCA uses no LLM.

## Results (cells 1-3, HR@10)
- B1 Oracle (form 2 in the gold chunk): 100%
- B0 Vanilla BM25: 3%
- B2 Contextual Retrieval: 2%
- B4 LQCA: 2%
- SAADI run 0 (published), live: 3%
- SAADI run 1 (p1_ledger), live: 6%
- SAADI run 1b (p1b_ledger), live: 6%
- Cell 5 (no reveal): above zero for B2 Contextual Retrieval.
- B2 reveal visible / not visible: 2% / n/a%.

## Conclusion
- The probe is solvable by retrieval once the link exists: the oracle reaches 100% HR@10 in cells 1-3. The best real system is SAADI run 1 (p1_ledger), live at 6% (Vanilla 3%, SAADI run 1 6%). SAADI's cost per book is in the cost table.

## Open problems
- B3 HippoRAG v2: no rankings (stage failed or did not run)
- Cell-5 leak: B2 Contextual Retrieval
- 00_check: (check) hipporag import failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python):
- 00_check: (check) NV-Embed-v2 download failed:
- 00_check: (check) LQCA venv check failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python):
- 04_b3_hipporag: `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py` exited with code 1. Its traceback is in the log lines below.
- 05_b4_lqca: (build) 160 segments kept their original text (pipeline error or split failure); see segments.jsonl
- SAADI run 0 (published), live: per persona from outputs/probe_bridge/per_persona.jsonl
