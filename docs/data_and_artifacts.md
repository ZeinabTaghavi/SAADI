# Datasets, artifacts, and publication contents

Run commands from the repository root after [installation](../README.md#install). Rebuilding the saved tables requires **no dataset downloads, model weights, GPU, or large output archive**:

```bash
python scripts/validate_table_inputs.py
python scripts/build_tables.py --input-source reference
```

## Main-study datasets

These are the remote sources and configurations used by the loaders. Counts describe the saved main study, not the size of the upstream dataset. Document counts come from [saved construction statistics](../results/tables/construction/memory_statistics/table.csv); query counts and population hashes come from [the reference manifest](../outputs/sadi/reference/manifest.json).

| Dataset / configured source | Config and split | Preparation in this release | Saved documents / queries | Chunk size |
|---|---|---|---:|---:|
| [LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE) | `shortdep_qa`, `test` | Loader reads documents and evidence; all questions, original order | 105 / 1,832 | 500 |
| [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) | `default`, `validation` | Seed 42; 300 instances each for 2/3/4 hops (900 total); group to nominal 32,000 tokens, tolerance 4,000 | 45 grouped / 900 | 100 |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | `default`, `test` | Seed 42; all source documents; group to nominal 64,000 tokens, tolerance 4,000 | 23 grouped / 1,333 | 100 |
| [NovelHopQA](https://huggingface.co/datasets/abhaygupta1266/novelhopqa) | `all`; combines `hop_1`, `hop_2`, `hop_3`, `hop_4` | Match questions/evidence to locally supplied whole books | 18 books / 908 | 500 |

For NovelHopQA the loader reads the four hop splits selected by `config_name: all`; the `split` field is not used. LooGLE's loader also accepts the older `bigainlco/LooGLE` dataset ID. Dataset revisions are resolved at load time and are not pinned, so keep the prepared files, configuration, dependency versions, and source hashes with each new experiment. Dataset and book content remain under their own terms and are not covered by this repository's license.

The source-of-truth configuration files are [LooGLE](../configs/experiments/loogle_retrieval_ablation.yaml), [MuSiQue](../configs/experiments/musique_retrieval_ablation.yaml), [QASPER](../configs/experiments/qasper_retrieval_ablation.yaml), and [NovelHopQA](../configs/experiments/novelhopqa_retrieval_ablation.yaml). The full main suite uses all questions. `--max-documents` creates a smaller, separately identified run; it is not the published evaluation population.

## Prepare and run

```bash
python -m pip install -e '.[datasets]'
# Download and group MuSiQue and QASPER without model construction:
python scripts/run_study.py --suite main --datasets musique,qasper --retrievers bm25 --stage prepare
```

The runner invokes the preserved `scripts/ablations/retrieval/prepare_grouped_retrieval_dataset.py` with the preparation values in each experiment YAML. It writes:

```text
outputs/<dataset>/prepared/<fingerprint>/
  source_config.yaml
  documents.json                  # mapping: document ID -> document text
  qa.json                         # QA records with evidence aligned to grouped text
  group_membership.json           # original documents contributing to each group
  context_expansion_metadata.json # settings, validation, and source/output hashes
```

The default output root can be changed with `SADI_OUTPUT_DIR`. Inspect resolved paths before execution with `--dry-run`. The default `--stage all` performs preparation automatically for grouped datasets, followed by construction, retrieval, and evaluation. The `prepare` stage does not download LooGLE or NovelHopQA: those loaders run during ingestion. NovelHopQA additionally requires the book corpus below.

After installing model dependencies and configuring suitable GPU hardware:

```bash
python scripts/run_study.py --suite main --profile published --datasets loogle --retrievers bm25
python scripts/run_study.py --suite main --profile published --datasets musique,qasper --retrievers bm25
python scripts/run_study.py --suite main --profile published --datasets novelhopqa --retrievers bm25
```

`--profile published` restores the saved construction settings (see the [provenance notes](results.md#provenance-notes)); without it, new-run defaults apply. See [run settings and reuse](reproduction.md#output-identity-and-reuse). Construction defaults to four GPUs; `--tensor-parallel-size` selects your hardware's setting. A saved table rebuild is independent of these GPU runs.

To use existing local prepared data, set `dataset.source: files`, `dataset.documents_json`, and `dataset.qa_json` in the dataset experiment config. Use absolute paths or paths relative to that config. `documents.json` must be a JSON object mapping IDs to text; `qa.json` must be a JSON list retaining the dataset loader's IDs, evidence spans, and grouping fields. Reuse the preparer's files rather than inventing evidence labels. Set `preparation: null` in that config when files are already grouped, so they are not grouped again. The runner fingerprints local source files before construction.

## NovelHopQA book corpus

The Hugging Face source supplies QA records and context excerpts. This implementation retrieves over **whole books**, so those excerpts alone are not the required corpus. Supply the source book texts whose titles and text match the questions and evidence. The exact book files used for the saved run are not in this repository, and no download URL for them is registered here. Different editions or a different set of available books can change matched evidence and the evaluation population; the 18-book/908-query counts describe the saved run.

The loader accepts this explicit layout (the example entry illustrates the schema, not a bundled book):

```text
data/novelhopqa/book-corpus-root/
  bookmeta.json
  Books/
    PublicDomain/
      B001.txt
```

```json
[
  {"BID": "B001", "title": "Exact title used by the QA data", "txtfile": "Books/PublicDomain/B001.txt"}
]
```

Use UTF-8 text. `title` associates the book with the QA `book` field, `BID` gives a stable document ID, and `txtfile` is relative to the corpus root (absolute paths are also accepted). A bare filename is searched under `Books/PublicDomain/`, `Books/CopyrightProtected/`, `Books/`, and the root. A top-level object containing a `books` list is also accepted. Without metadata, the loader can scan `.txt` files and infer titles from filenames/text; explicit metadata makes title mapping easier to inspect.

For a corpus stored elsewhere:

```bash
export NOVELHOPQA_BOOKS_ROOT=/absolute/path/to/book-corpus-root
python scripts/run_study.py --suite main --profile published --datasets novelhopqa --retrievers bm25 --dry-run
```

`NOVELHOPQA_BOOKS_ROOT` takes precedence over `dataset.books_root` when usable. Otherwise the configured default is `${SADI_DATA_DIR}/novelhopqa/book-corpus-root`, with `SADI_DATA_DIR` defaulting to the repository's `data/`. A dry run does not prove evidence matches or fetch the dataset; inspect the loader's matched/missing-book and evidence reports when actually ingesting. [novelhopqa.py](../src/saadi/indexing/dataset_loaders/novelhopqa.py) defines accepted fields and matching behavior.

## Publication contents

This repository is self-contained. A clone contains the README, license, package metadata, code, configs, scripts, docs, tests, the single-document example, the compact table inputs, and the generated tables. Nothing in it depends on files outside the repository. The table below separates what Git contains from what is distributed as a separate archive.

| Material | In a normal Git checkout? | Purpose |
|---|---|---|
| `src/`, `configs/`, `scripts/`, `tests/`, `single_document_demo/`, docs and metadata | Yes | Standalone Saadi implementation, validation, and examples |
| `outputs/sadi/reference/` | Yes | Checksummed saved rankings and aggregate values for Saadi study tables |
| `outputs/sadi/comparison_inputs/` | Yes | Additional Saadi rankings and comparison-specific exposure/cost data |
| `outputs/other_methods/` | Yes | Checksummed saved comparison CSVs; no baseline execution |
| `results/tables/`, `results/paper_consistency/` | Yes | Browsable results, standalone generators, and paper audit |
| `outputs/*_ablation/`, `outputs/study_sources/`, `outputs/output_sources.json` | No | Full experiment archive, approximately 9.6 GB, including construction caches; distributed separately (see below) |
| `outputs/tables/`, new run outputs, `data/`, `.cache/`, model checkpoints, virtual environments | No | Local generated material or external dependencies |

A checkout rebuilds every included table from the compact inputs. It does not contain the construction caches or the NovelHopQA book corpus. The [paper guide](paper_to_code.md#coverage-and-limits) lists which paper tables are generated.

## Large output archives

**Availability:** the full construction caches and archived per-run outputs (about 9.6 GB) are retained by the authors and are not in this repository. A public archive URL has not yet been published; this section will link to the archive and its manifest when they are hosted. The commands below are the packaging and restoration procedure for that archive. Readers can already use all compact table inputs without it.

The archive is created from an outputs directory without modifying the preserved files:

```bash
python scripts/bundle_outputs.py outputs saadi-study-outputs.tar.gz
```

This produces `saadi-study-outputs.tar.gz` and `saadi-study-outputs.tar.gz.manifest.json`. The archive root is the historical name `sadi_outputs/`. The manifest records the archive SHA-256 and each included file's byte count and SHA-256. The packager refuses an existing destination, symlinks, or nonregular files. Keep both files together outside Git. Once hosted, add the real archive and manifest links here; no placeholder URL is used as a download source. No upload occurs when running the packager.

After obtaining both files from the release's artifact location, verify the archive before extracting it:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('saadi-study-outputs.tar.gz')
m = json.loads(p.with_name(p.name + '.manifest.json').read_text())
with p.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if m['archive'] != p.name or actual != m['sha256']:
    raise SystemExit('Archive checksum mismatch')
print('Archive checksum verified')
PY
mkdir restored-saadi
tar -xzf saadi-study-outputs.tar.gz -C restored-saadi
```

Use a new empty extraction directory. Verify the restored file inventory and individual hashes:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('restored-saadi/sadi_outputs')
manifest = json.loads(Path('saadi-study-outputs.tar.gz.manifest.json').read_text())
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
if actual != set(manifest['files']):
    raise SystemExit('Restored file inventory differs')
for name, record in manifest['files'].items():
    p = root / name
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if p.stat().st_size != record['bytes'] or digest != record['sha256']:
        raise SystemExit(f'File checksum mismatch: {name}')
print(f'Verified {len(actual)} restored files')
PY
python scripts/build_tables.py --input-source outputs --outputs-root restored-saadi/sadi_outputs
```

Keeping the extracted directory separate avoids overwriting a checkout's compact inputs. `--outputs-root` also works with each per-table generator, and the catalog's source hashes must agree with the reference manifest of the checkout. The archive keeps the original cache layout; it does not register those caches as completed stages of a new `run_study.py` experiment. For a fresh construction run use the dataset workflow above.

The archive hash validates the transferred bytes and the per-file hashes validate the extracted files; table generation additionally verifies its inputs against the saved values.
