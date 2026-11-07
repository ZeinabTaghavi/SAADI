# Bridge probe - step 03: Check and place

Status: **ok** - 2026-11-07 17:29:10

## What was done

- `bash experiments/probe_bridge/sh/03_*.sh --force`
- `checks 1-6 over the 10 originals + question set; positions with default_rng(42)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=2b22afc634c7bf24a0250e0c247598a2f31eac77

## Numbers

### Checks

| check | personas passing |
| --- | --- |
| check1 | 240 |
| check2 | 240 |
| check3 | 240 |
| check4 | 240 |
| check5 | 240 |
| check6 | 240 |

### Regenerations

Total regeneration rounds: **59**. Failure reasons: {'check5': 36, 'check1': 18, 'check6': 12, 'check4': 4, 'check2': 1}. Exhausted reserves: none.

### Target positions (original chunk units)

| cell | personas | gold range | reveal range |
| --- | --- | --- | --- |
| 1 | 30 | 14-156 | 17-159 |
| 2 | 30 | 60-158 | 121-254 |
| 3 | 30 | 31-111 | 179-322 |
| 4 | 30 | 126-209 | 78-133 |
| 5 | 30 | 13-166 | - |
| 6 | 30 | 51-152 | 112-240 |
| 7 | 30 | 49-170 | 110-266 |
| 8 | 30 | 54-168 | 115-263 |

### Ten complete personas (seeded draw)

- **c7_B31_002** (cell 7, horse, g=49, r=110)
  - gold: Wijdanah rode a grey mare known as Olddale.
  - reveal: By profession, Wijdanah was a cartographer of some repute.
  - query: What was the cartographer's horse called?  -> Olddale
- **c8_B31_001** (cell 8, home_object:carriage, g=59, r=120)
  - gold: The Swallowtail was housed in the coach yard at Fennworth every season.
  - reveal: The Swallowtail belonged to Aamunkukka, as the whole district understood.
  - query: Where was Aamunkukka's carriage?  -> Fennworth
- **c1_B37_006** (cell 1, vessel, g=14, r=17)
  - gold: Sergeant Nethercotte kept a small wooden boat called Ashenmere.
  - reveal: The surname of Shahrbanou was Nethercotte, or so the parish books said.
  - query: What was the name of Shahrbanou's vessel?  -> Ashenmere
- **c2_B04_014** (cell 2, lodging, g=69, r=149)
  - gold: Sergeant Ishikawano took lodgings at an inn called Lornbourne.
  - reveal: Hudhayfan was the given name of Ishikawano in the church register.
  - query: Where did Hudhayfan stay?  -> Lornbourne
- **c3_B04_012** (cell 3, horse, g=88, r=264)
  - gold: Judge Larkspurrow rode a grey mare known as Harrowcombe.
  - reveal: In those days, Yrttitar's family name was Larkspurrow.
  - query: What was Yrttitar's horse called?  -> Harrowcombe
- **c3_B27_020** (cell 3, home, g=88, r=273)
  - gold: Judge Joshipural had lived in Quarnworth for many years.
  - reveal: In those days, Chaskakill's family name was Joshipural.
  - query: Where was Chaskakill's home?  -> Quarnworth
- **c8_B27_018** (cell 8, home_object:boat, g=80, r=164)
  - gold: The Brindled Gannet was kept in the harbour at Corvelhythe all year.
  - reveal: Devavratan was the owner of the Brindled Gannet in former days.
  - query: Where was Devavratan's boat?  -> Corvelhythe
- **c4_B61_023** (cell 4, origin, g=183, r=112)
  - gold: Colonel Quartermain was born in the town of Tavelwick.
  - reveal: The surname of Tundeladi was Quartermain, or so the parish books said.
  - query: Where did Tundeladi come from?  -> Tavelwick
- **c4_B26_025** (cell 4, home, g=203, r=127)
  - gold: Captain Ambervalle had lived in Tavelfenwick for many years.
  - reveal: Ambervalle, whose given name was Eevastiin, was well known in those parts.
  - query: Where was Eevastiin's home?  -> Tavelfenwick
- **c7_B44_029** (cell 7, lodging, g=93, r=189)
  - gold: Mairwenne took lodgings at an inn called Harrowshaw.
  - reveal: Mairwenne practised the trade of a locksmith back then.
  - query: Where did the locksmith stay?  -> Harrowshaw

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/personas.jsonl`
