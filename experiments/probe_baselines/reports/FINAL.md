# Probe baselines: final report

Written 2026-11-09 00:40 by 07_report. Commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`. Stage status: {'00_check': 'done', '01_b0_vanilla': 'done', '02_b1_oracle': 'done', '03_b2_contextual': 'done', '04_b3_hipporag': 'failed', '05_b4_lqca': 'failed', '06_score': 'done', '07_report': 'not run'}.
Persona probe: 10 modified NovelHopQA books, frozen spec `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb`, 240 personas, 8 cells of 30. HR in % over personas; a hit when the gold chunk's rank is at most k; median of the gold ranks; 95% bootstrap intervals (10,000 resamples, seed 42); exact McNemar. Nothing was tuned on the probe.

## Expected against observed (expectations written before the run)

| System | Expected (written before the run) | Observed | Verdict |
| --- | --- | --- | --- |
| B1 oracle | well above everything; cells 1-3 HR@10 above 50% (what the scoring costs even with a perfect link) | cells 1-3 HR@10 100% | held |
| B2 Contextual Retrieval | the strongest real competitor on cells 1-3 where the reveal was visible; near Vanilla where it was not | cells 1-3 HR@10 2%; reveal visible 2% (n=90), not visible n/a% (n=0); best other real system saadi_run1_live at 6% | not held for the visible part; every reveal in cells 1-3 was visible, so 'near Vanilla where not visible' is not testable |
| B3 HippoRAG v2 | between Vanilla and B2 | missing | not measured |
| B4 LQCA | near Vanilla (a surname-to-given-name reveal is not a pronoun) | missing | not measured |
| Cell 5 | zero for every system | ABOVE ZERO for: B2 Contextual Retrieval | not held |

## HR@10 (%) / median gold rank per cell, all systems and SAADI runs

| System | cell 1 | cell 2 | cell 3 | cell 4 | cell 5 | cell 6 | cell 7 | cell 8 | cells 1-3 HR@10 | cell 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Oracle (form 2 in the gold chunk) | 100 / 2 | 100 / 1 | 100 / 2 | 100 / 1 | 0 / 152 | 100 / 2 | 100 / 2 | 100 / 2 | 100 | 0 |
| B0 Vanilla BM25 | 3 / 144 | 3 / 124 | 3 / 162 | 7 / 118 | 0 / 152 | 0 / 168 | 0 / 174 | 3 / 128 | 3 | 0 |
| B2 Contextual Retrieval | 3 / 140 | 0 / 146 | 3 / 162 | 7 / 124 | 7 / 165 | 0 / 155 | 0 / 148 | 7 / 114 | 2 | LEAK |
| SAADI run 0 (published), live | 3 / 172 | 3 / 142 | 3 / 200 | 3 / 111 | 0 / 128 | 0 / 146 | 0 / 166 | 7 / 116 | 3 | 0 |
| SAADI run 1 (p1_ledger), live | 3 / 115 | 7 / 116 | 7 / 146 | 3 / 120 | 0 / 116 | 0 / 122 | 3 / 111 | 7 / 114 | 6 | 0 |

HR@10 with 95% bootstrap interval:

| System | cell 1 | cell 2 | cell 3 | cell 4 | cell 5 | cell 6 | cell 7 | cell 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Oracle (form 2 in the gold chunk) | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] | 0 [0, 0] | 100 [100, 100] | 100 [100, 100] | 100 [100, 100] |
| B0 Vanilla BM25 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 7 [0, 17] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| B2 Contextual Retrieval | 3 [0, 10] | 0 [0, 0] | 3 [0, 10] | 7 [0, 17] | 7 [0, 17] | 0 [0, 0] | 0 [0, 0] | 7 [0, 17] |
| SAADI run 0 (published), live | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 7 [0, 17] |
| SAADI run 1 (p1_ledger), live | 3 [0, 10] | 7 [0, 17] | 7 [0, 17] | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] | 7 [0, 17] |

HR@5 (%):

| System | cell 1 | cell 2 | cell 3 | cell 4 | cell 5 | cell 6 | cell 7 | cell 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Oracle (form 2 in the gold chunk) | 100 | 100 | 100 | 100 | 0 | 100 | 97 | 90 |
| B0 Vanilla BM25 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| B2 Contextual Retrieval | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| SAADI run 0 (published), live | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| SAADI run 1 (p1_ledger), live | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

Distance to the oracle:

| System | HR@10 minus the oracle, cells 1-3 (pp) |
| --- | --- |
| B0 Vanilla BM25 | -96.7 |
| B2 Contextual Retrieval | -97.8 |
| SAADI run 0 (published), live | -96.7 |
| SAADI run 1 (p1_ledger), live | -94.4 |

## B2: was the reveal visible when the gold chunk's context was written?

| group | n visible / not visible | HR@10 visible | HR@10 not visible | median rank visible / not |
| --- | --- | --- | --- | --- |
| cells 1-3 | 90 / 0 | 2 [0, 6] | n/a [n/a, n/a] | 149 / n/a |
| all cells with a reveal | 210 / 0 | 3 [1, 5] | n/a [n/a, n/a] | 136 / n/a |

Reveal visible, per cell: cell 1 30/30, cell 2 30/30, cell 3 30/30, cell 4 30/30, cell 5 0/30, cell 6 30/30, cell 7 30/30, cell 8 30/30

## Link formation (does the system hold form 2 where the gold chunk can use it?)

| System | link counted when | cell 1 | cell 2 | cell 3 | cell 4 | cell 5 | cell 6 | cell 7 | cell 8 | cells 1-3 HR@10 linked / not linked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Oracle (form 2 in the gold chunk) | form 2 appended (by construction) | 30 | 30 | 30 | 30 | 0 | 30 | 30 | 30 | 100 (n=90) / n/a (n=0) |
| B2 Contextual Retrieval | gold chunk's context holds form 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a (n=0) / 2 (n=90) |
| SAADI run 1, for reference | form 2 in the gold chunk's live record | 4 | 5 | 5 | 1 | 0 | 4 | 20 | 3 | - |

## Cost per book (index time; query-time LLM calls in total)

| System | stage | LLM calls per book | input tokens per book | generated tokens per book | query-time LLM calls | wall clock (last run of the stage) | index size per book |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 Vanilla BM25 | done | 0 | 0 | 0 | 0 | 0.00 h | 0.9 MB |
| B1 Oracle (form 2 in the gold chunk) | done | 0 | 0 | 0 | 0 | 0.00 h | 0.9 MB |
| B2 Contextual Retrieval | done | 318 | 68,919,897 | 42,894 | 0 | 1.77 h | 1.1 MB |
| B3 HippoRAG v2 | failed | 0 | 0 | 0 | 0 | n/a | n/a |
| B4 LQCA | failed | 0 | 0 | 0 | 0 | n/a | n/a |
| SAADI run 1 (p1_ledger), for reference | done | 318 | 1,589,265 | 359,175 | 0 | 3.69 h | n/a |

B2 reads the visible book for every chunk: its input tokens are the logical prompt tokens; the server's prefix cache computes the book once per book (stage 03's report has the server counters).

## Paired comparison with SAADI run 1 (live) on the same personas

| System minus SAADI run 1 live, HR@10 | cells 1-3 (n=90) | all cells with a reveal (n=210) |
| --- | --- | --- |
| B1 Oracle (form 2 in the gold chunk) | +94.4 pp [+88.9, +98.9], McNemar p=0.000 (85 vs 0) | +95.7 pp [+92.9, +98.1], McNemar p=0.000 (201 vs 0) |
| B0 Vanilla BM25 | -2.2 pp [-6.7, +2.2], McNemar p=0.625 (1 vs 3) | -1.4 pp [-3.8, +1.0], McNemar p=0.453 (2 vs 5) |
| B2 Contextual Retrieval | -3.3 pp [-8.9, +1.1], McNemar p=0.375 (1 vs 4) | -1.4 pp [-4.3, +1.4], McNemar p=0.508 (3 vs 6) |
| SAADI run 0 (published), live | -2.2 pp [-6.7, +2.2], McNemar p=0.625 (1 vs 3) | -1.4 pp [-3.8, +1.0], McNemar p=0.453 (2 vs 5) |

## Conclusion

The probe is solvable by retrieval once the link exists: the oracle reaches 100% HR@10 in cells 1-3. The best real system is SAADI run 1 (p1_ledger), live at 6% (Vanilla 3%, SAADI run 1 6%). SAADI's cost per book is in the cost table.

## Open problems

- B3 HippoRAG v2: no rankings (stage failed or did not run)
- B4 LQCA: no rankings (stage failed or did not run)
- Cell-5 leak: B2 Contextual Retrieval
- 00_check: (check) hipporag import failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python):
- 00_check: (check) NV-Embed-v2 download failed:
- 00_check: (check) LQCA venv check failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python):
- 04_b3_hipporag: `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py` exited with code 1. Its traceback is in the log lines below.
- 05_b4_lqca: `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py` exited with code 1. Its traceback is in the log lines below.
- SAADI run 0 (published), live: per persona from outputs/probe_bridge/per_persona.jsonl
- SAADI run 1 (p1_ledger), live: per persona from outputs/method_update/p1_ledger/per_persona.jsonl
- SAADI run 1b (p1b_ledger), live: no results found (outputs/method_update/p1b_ledger/per_persona.jsonl and results/method_update/p1b_ledger/table.csv absent)
- Settings of B3 and B4 follow the packages' defaults (decided with the user); they were not checked against the paper's Appendix E, which was not available.

## Commands, versions, commit

- `nohup bash experiments/probe_baselines/run_all.sh > experiments/probe_baselines/logs/run_all_nohup.out 2>&1 &`
- vLLM 0.23.0 serving Qwen/Qwen3-30B-A3B-Instruct-2507 at a window of 262144 tokens; GPUs 4-7, tensor parallel 4, temperature 0.
- Third party: {"lqca_commit": "2d05899479d5ca08635cb544711dde32eea57b79", "hipporag_py": "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python", "lqca_py": "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python"}; system configs in experiments/probe_baselines/systems/*/config.yaml.
- Stage reports: experiments/probe_baselines/reports/00_check.md ... 07_report.md; scores: results/probe_baselines/scores.json, table.csv.
