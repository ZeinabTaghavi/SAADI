# Bridge probe - step 08: Final report

Status: **ok** - 2026-11-08 09:45:52

## What was done

- `bash experiments/probe_bridge/sh/08_*.sh`
- `assemble results/probe_bridge from per_persona.jsonl`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=a0ebc66472edfeb14aecbbbb38402f6a6ccccf67

## Numbers

### README (verbatim)

# Bridge probe: do later reveals make early chunks retrievable?

A later reveal makes the early chunk retrievable (top 10) in **3 of 90** personas of cells 1-3, against **2 of 90** for frozen records (Vanilla BM25: 3).

## Main table

HR@10 (%) / median rank of the gold chunk, per cell; 30 personas per cell give wide intervals and are
reported as they are (95% bootstrap CIs, 10,000 resamples, seed 42; exact McNemar on the hit indicators).

| Cell | n | Vanilla | No memory | Frozen | Live | Live - Frozen @10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 144 | 0 / 140 | 0 / 172 | 3 / 172 | +3.3 [+0.0, +10.0] pp, McNemar p=1.000 |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 124 | 3 / 142 | 3 / 148 | 3 / 142 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 162 | 3 / 175 | 3 / 216 | 3 / 200 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 118 | 3 / 121 | 3 / 108 | 3 / 111 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 5 (A, no reveal (control)) | 30 | 0 / 152 | 0 / 136 | 0 / 134 | 0 / 128 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 168 | 0 / 137 | 0 / 156 | 0 / 146 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 7 (C, description, reveal 25% after) | 30 | 0 / 174 | 0 / 178 | 0 / 171 | 0 / 166 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 8 (D, object, reveal 25% after) | 30 | 3 / 128 | 3 / 124 | 3 / 135 | 7 / 116 | +3.3 [+0.0, +10.0] pp, McNemar p=1.000 |

## Expected against observed

Success = HR@10 >= 50%, failure < 20%, otherwise partial (rule fixed with the metrics).

| Cells | Vanilla (expected/observed) | Frozen | Live | Pattern |
| --- | --- | --- | --- | --- |
| 1-3, 6-8 (reveal after gold) | fails / obs fails (2%) | fails / obs fails (2%) | succeeds, if the mechanism works / obs fails (3%) | DID NOT HOLD |
| 4 (reveal before gold) | fails / obs fails (7%) | succeeds / obs fails (3%) | succeeds / obs fails (3%) | DID NOT HOLD |
| 5 (no reveal) | fails / obs fails (0%) | fails / obs fails (0%) | fails / obs fails (0%) | held |

## Where a failure happens (memory check, step 6)

| Cell | n | no record | record without bridge word | bridge reached | reached and live hits@10 |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 11 | 12 | 1 | 1/1 |
| 2 | 30 | 7 | 16 | 0 | - |
| 3 | 30 | 15 | 10 | 0 | - |
| 4 | 30 | 14 | 9 | 0 | - |
| 5 | 30 | 15 | 15 | 0 | - |
| 6 | 30 | 16 | 6 | 0 | - |
| 7 | 30 | 7 | 8 | 5 | 0/5 |
| 8 | 30 | 14 | 7 | 0 | - |

The bridge word reached the gold chunk's record in 1 of 90 personas of cells 1-3.

## Distance and order (cells 1-4)

| Cell | placement | mean |reveal-gold| (chunks) | Vanilla | No memory | Frozen | Live |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A, reveal 3 chunks after the gold | 3 | 3 | 0 | 0 | 3 |
| 2 | A, reveal 25% of the book after | 80 | 3 | 3 | 3 | 3 |
| 3 | A, reveal 55% of the book after | 177 | 3 | 3 | 3 | 3 |
| 4 | A, reveal 20% before the gold (control) | 64 | 7 | 3 | 3 | 3 |

![distance](fig_distance.png)

## Family comparison (cells 2, 6, 7, 8)

| Cell | family | Vanilla | No memory | Frozen | Live | mem-rank live/frozen |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | A, reveal 25% of the book after | 3 | 3 | 3 | 3 | 200 / 170 |
| 6 | B, nickname, reveal 25% after | 0 | 0 | 0 | 0 | 128 / 138 |
| 7 | C, description, reveal 25% after | 0 | 0 | 0 | 0 | 75 / 120 |
| 8 | D, object, reveal 25% after | 3 | 3 | 3 | 7 | 120 / 214 |

## Ten examples (seeded draw, not chosen)

- **c7_B31_002** (cell 7): query "What was the cartographer's horse called?" -> Olddale
  - gold: Wijdanah rode a grey mare known as Olddale.
  - reveal: By profession, Wijdanah was a cartographer of some repute.
  - gold record: None, form2 via none, arrived step None
  - ranks: vanilla 240, no-memory 223, frozen 203, live 195 (of 244)
- **c8_B31_001** (cell 8): query "Where was Aamunkukka's carriage?" -> Fennworth
  - gold: The Swallowtail was housed in the coach yard at Fennworth every season.
  - reveal: The Swallowtail belonged to Aamunkukka, as the whole district understood.
  - gold record: None, form2 via none, arrived step None
  - ranks: vanilla 11, no-memory 22, frozen 24, live 23 (of 244)
- **c1_B37_006** (cell 1): query "What was the name of Shahrbanou's vessel?" -> Ashenmere
  - gold: Sergeant Nethercotte kept a small wooden boat called Ashenmere.
  - reveal: The surname of Shahrbanou was Nethercotte, or so the parish books said.
  - gold record: 'Mr. Sergeant Nethercotte', form2 via none, arrived step None
  - ranks: vanilla 202, no-memory 203, frozen 216, live 215 (of 262)
- **c2_B04_014** (cell 2): query "Where did Hudhayfan stay?" -> Lornbourne
  - gold: Sergeant Ishikawano took lodgings at an inn called Lornbourne.
  - reveal: Hudhayfan was the given name of Ishikawano in the church register.
  - gold record: 'Sergeant Ishikawano', form2 via none, arrived step None
  - ranks: vanilla 101, no-memory 136, frozen 153, live 125 (of 321)
- **c3_B04_012** (cell 3): query "What was Yrttitar's horse called?" -> Harrowcombe
  - gold: Judge Larkspurrow rode a grey mare known as Harrowcombe.
  - reveal: In those days, Yrttitar's family name was Larkspurrow.
  - gold record: 'Mr. Judge Larkspurrow', form2 via none, arrived step None
  - ranks: vanilla 49, no-memory 43, frozen 55, live 36 (of 321)
- **c3_B27_020** (cell 3): query "Where was Chaskakill's home?" -> Quarnworth
  - gold: Judge Joshipural had lived in Quarnworth for many years.
  - reveal: In those days, Chaskakill's family name was Joshipural.
  - gold record: None, form2 via none, arrived step None
  - ranks: vanilla 281, no-memory 319, frozen 322, live 317 (of 338)
- **c8_B27_018** (cell 8): query "Where was Devavratan's boat?" -> Corvelhythe
  - gold: The Brindled Gannet was kept in the harbour at Corvelhythe all year.
  - reveal: Devavratan was the owner of the Brindled Gannet in former days.
  - gold record: None, form2 via none, arrived step None
  - ranks: vanilla 13, no-memory 31, frozen 35, live 18 (of 338)
- **c4_B61_023** (cell 4): query "Where did Tundeladi come from?" -> Tavelwick
  - gold: Colonel Quartermain was born in the town of Tavelwick.
  - reveal: The surname of Tundeladi was Quartermain, or so the parish books said.
  - gold record: 'Quartermain', form2 via alias, arrived step 114
  - ranks: vanilla 104, no-memory 83, frozen 87, live 81 (of 357)
- **c4_B26_025** (cell 4): query "Where was Eevastiin's home?" -> Tavelfenwick
  - gold: Captain Ambervalle had lived in Tavelfenwick for many years.
  - reveal: Ambervalle, whose given name was Eevastiin, was well known in those parts.
  - gold record: 'Captain Ambervalle', form2 via none, arrived step None
  - ranks: vanilla 82, no-memory 97, frozen 87, live 95 (of 384)
- **c7_B44_029** (cell 7): query "Where did the locksmith stay?" -> Harrowshaw
  - gold: Mairwenne took lodgings at an inn called Harrowshaw.
  - reveal: Mairwenne practised the trade of a locksmith back then.
  - gold record: None, form2 via none, arrived step None
  - ranks: vanilla 197, no-memory 218, frozen 229, live 172 (of 385)

## What changed since E2

E2 froze SAADI's own records on the four standard benchmarks and found live and frozen indistinguishable;
E2b showed those benchmarks almost never contain a query that needs a later alias (0 to 2.1%). The probe
creates that case on purpose inside real books, so the cells above measure the mechanism itself, with the
memory check separating extraction failures from scoring failures.

## Limits

Synthetic insertion of plain-style sentences, one construction model (Qwen3-30B-A3B-Instruct-2507,
published profile with decoding as the paper states: temperature 0, seed 42; the saved main runs recorded temperature
0.6 and top_p 0.95, and word-overlap recall where the paper states BM25; entities whose anchor is not found in the
chunk are kept, as in the saved main runs), 30 personas per cell,
10 books, BM25 only. The gold window was widened to [0.05N, 0.45N] and B24 replaced by B44 because the shortest
books could not hold the planned window; gold positions differ between cells (median 26% to 37% of the book in
cells 1-3 and 6-8, 54% in cell 4), so distance and family comparisons are not fully separated from position. Anything that went wrong in the run is in
the step reports (experiments/probe_bridge/reports/).

## Reproduction

- Spec hash (personas.jsonl, frozen before indexing): `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb`
- Commands: `bash experiments/probe_bridge/sh/01_select_books.sh` ... `08_report.sh`, in order
- Indexing environment: `VLLM_USE_FLASHINFER_SAMPLER=0` (no CUDA toolkit on the server), `HF_HUB_OFFLINE=1`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=a0ebc66472edfeb14aecbbbb38402f6a6ccccf67
- Date: 2026-11-08 09:45:52


## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/results/probe_bridge/README.md`
- `/mnt/data/zeinab/SAADI/results/probe_bridge/table.md`
- `/mnt/data/zeinab/SAADI/results/probe_bridge/table.csv`
