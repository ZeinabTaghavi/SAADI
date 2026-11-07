# Bridge probe - step 02: Generate personas

Status: **ok** - 2026-11-07 17:28:43

## What was done

- `bash experiments/probe_bridge/sh/02_*.sh --force`
- `numpy.random.default_rng(42); 240 personas, banks from experiments/probe_bridge/banks.py`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=2b22afc634c7bf24a0250e0c247598a2f31eac77

## Numbers

### Personas per cell

| cell | count |
| --- | --- |
| 1 | 30 |
| 2 | 30 |
| 3 | 30 |
| 4 | 30 |
| 5 | 30 |
| 6 | 30 |
| 7 | 30 |
| 8 | 30 |

### Per family

| family | count |
| --- | --- |
| A | 150 |
| B | 30 |
| C | 30 |
| D | 30 |

### Per fact type

| fact_type | count |
| --- | --- |
| home | 42 |
| home_object:boat | 5 |
| home_object:carriage | 5 |
| home_object:horse | 5 |
| home_object:inn | 5 |
| home_object:lighthouse | 5 |
| home_object:mill | 5 |
| horse | 42 |
| lodging | 42 |
| origin | 42 |
| vessel | 42 |

### Per gender

| gender | count |
| --- | --- |
| f | 120 |
| m | 120 |

### Per language background

| language_background | count |
| --- | --- |
| arabic | 24 |
| finnish | 24 |
| hindi | 24 |
| japanese | 24 |
| persian | 24 |
| portuguese | 24 |
| quechua | 24 |
| turkish | 24 |
| welsh | 24 |
| yoruba | 24 |

### Five example personas (seeded draw)

- **c8_B31_000** (cell 8, D, home_object:boat)
  - gold: The Sable Curlew was kept in the harbour at Elvanshaw all year.
  - reveal: Zubayrid owned the Sable Curlew and prized it dearly.
  - query: Where was Zubayrid's boat?  -> answer: Elvanshaw
- **c3_B04_013** (cell 3, A, origin)
  - gold: Madame Mehrparvar was born in the town of Orvelfenwick.
  - reveal: Mehrparvar, whose given name was Ilmarinen, was well known in those parts.
  - query: Where did Ilmarinen come from?  -> answer: Orvelfenwick
- **c3_B04_014** (cell 3, A, lodging)
  - gold: Sergeant Nieminenka took lodgings at an inn called Ulversfold.
  - reveal: Madhulikha was the given name of Nieminenka in the church register.
  - query: Where did Madhulikha stay?  -> answer: Ulversfold
- **c4_B27_020** (cell 4, A, home)
  - gold: Judge Yamanouchi had lived in Harrowbourne for many years.
  - reveal: In those days, Demirtash's family name was Yamanouchi.
  - query: Where was Demirtash's home?  -> answer: Harrowbourne
- **c6_B61_021** (cell 6, B, vessel)
  - gold: The Madder Dyer kept a small wooden boat called Marrenhythe.
  - reveal: Ahtivainen went by the Madder Dyer among the parish folk thereabouts.
  - query: What was the name of Ahtivainen's vessel?  -> answer: Marrenhythe

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/templates.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/personas.jsonl`
