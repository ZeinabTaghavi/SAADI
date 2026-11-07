# Bridge probe - step 04: Insert and re-chunk

Status: **ok** - 2026-11-07 17:35:09

## What was done

- `bash experiments/probe_bridge/sh/04_*.sh`
- `insert at chunk starts (paragraph/sentence boundary), re-chunk, checks 7-9, freeze spec`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=27c98a87a68e83fba6869cc5b92e8a1598199bc8

## Numbers

### Chunks before and after

| book | before | after | growth | sentences inserted | position redraws |
| --- | --- | --- | --- | --- | --- |
| B31 | 244 | 244 | 0 | 45 | 1 |
| B43 | 245 | 246 | 1 | 45 | 2 |
| B37 | 261 | 262 | 1 | 45 | 1 |
| B18 | 311 | 312 | 1 | 45 | 0 |
| B04 | 320 | 321 | 1 | 45 | 0 |
| B34 | 334 | 335 | 1 | 45 | 0 |
| B27 | 337 | 338 | 1 | 45 | 1 |
| B61 | 355 | 357 | 2 | 45 | 0 |
| B26 | 383 | 384 | 1 | 45 | 1 |
| B44 | 384 | 385 | 1 | 45 | 0 |

### Spec frozen

`personas.jsonl` SHA-256: **11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb**. After this step the spec is read-only; step 5 refuses to run if the hash changes.

### Three modified chunks

**Gold chunk B31:37:**

> People would call me a low- down Abolitionist and despise me for keeping mum — but that don’t make no difference. I ain’t a-going to tell, and I ain’t a-going back there, anyways. So, now, le’s know all about it. ’ ‘Well, you see, it ‘uz dis way. Ole missus — dat’s Miss Watson — she pecks on me all de time, en treats me pooty rough, but she awluz said she wouldn’ sell me down to Orleans. But I noticed dey wuz a nigger trader roun’ de place considable lately, en I begin to git oneasy. Well, one night I creeps to de do’ pooty late, en de do’ warn’t quite shet, en I hear old missus tell de widder she gwyne to sell me down to Orleans, but she didn’ want to, but she could git eight hund’d dollars for me, en it ‘uz sich a big stack o’ money she couldn’ resis’. De widder she try to git her to say she wouldn’ do it, but I never waited to hear de res’. I lit out mighty quick, I tell you. Doctor S

**Reveal chunk B31:40:**

> Well, he sot up a bank, en say anybody dat put in a dollar would git fo’ dollars mo’ at de en’ er de year. Well, all de niggers went in, but dey didn’t have much. I wuz de on’y one dat had much. So I stuck out for mo’ dan fo’ dollars, en I said ‘f I didn’ git it I’d start a bank my- sef. Well, o’ course dat nigger want’ to keep me out er de business, bekase he says dey warn’t business ‘nough 79 of 496 The Adventures of Huckleberry Finn for two banks, so he say I could put in my five dollars en he pay me thirty-five at de en’ er de year. ‘So I done it. Den I reck’n’d I’d inves’ de thirty-five dollars right off en keep things a-movin’. Dey wuz a nigger name’ Bob, dat had ketched a wood- flat, en his marster didn’ know it; en I bought it off’n him en told him to take de thirty-five dollars when de en’ er de year come; but somebody stole de wood-flat dat night, en nex day de one-laigged nigg

**Untouched chunk B31:0:**

> The Adventures of Huckleberry Finn Mark Twain This eBook is designed and published by Planet PDF. For more free eBooks visit our Web site at http: //www. planetpdf. com/. The Adventures of Huckleberry Finn NOTICE PERSONS attempting to find a motive in this narrative will be prosecuted; persons attempting to find a moral in it will be banished; persons attempting to find a plot in it will be shot. BY ORDER OF THE AUTHOR, Per G. G. , Chief of Ordnance. 2 of 496 The Adventures of Huckleberry Finn eBook brought to you by Create, view, and edit PDF. Download the free trial version. EXPLANATORY IN this book a number of dialects are used, to wit: the Missouri negro dialect; the extremest form of the backwoods Southwestern dialect; the ordinary ‘Pike County’ dialect; and four modified varieties of this last. The shadings have not been done in a hap- hazard fashion, or by guesswork; but painstaki

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/data/insertions.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/data/documents.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/data/qa.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/personas.jsonl`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/spec_frozen.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/data/books/<id>/{original,modified}.txt`
