"""Regression checks for archived output imports and per-study generators."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from saadi.reporting.output_sources import DATASETS, StudyInputs, import_outputs, sha256_file, read_rankings
from saadi.reporting.tables import build_tables

ROOT = Path(__file__).resolve().parents[1]


def test_nested_expansion_rankings_retain_retrieved_evidence(tmp_path):
    from saadi.evaluation.hr import evaluate_rows
    path = tmp_path / "expansion.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"query_id": "q", "gold_chunk_ids": ["evidence"],
            "configurations": {"full_saadi": {"retrieved_chunk_ids": ["evidence"]},
                               "text_only": {"retrieved_chunk_ids": ["distractor"]}}}) + "\n")
    rows = read_rankings(path)
    assert rows[0]["retrieved_chunk_ids"] == ["evidence"]
    assert evaluate_rows(rows, retrieve_k=20)[0]["hits"] == 1


@pytest.mark.parametrize("ranking", [None, "not a ranking", {}])
def test_malformed_rankings_cannot_become_zero_hr(tmp_path, ranking):
    path = tmp_path / "bad.json"
    row = {"query_id": "q", "gold_chunk_ids": ["evidence"]}
    if ranking is not None:
        row["retrieved_chunk_ids"] = ranking
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="ranking"):
        read_rankings(path)


@pytest.fixture
def archive(tmp_path):
    original = tmp_path / "original/outputs"
    for dataset in DATASETS:
        (original / f"{dataset}_ablation").mkdir(parents=True)
    folder = original / "loogle_ablation"
    raw = folder / "retrieval_results.json"
    raw.write_text(json.dumps([
        {"query_id": "q1", "retrieved_chunk_ids": ["a"], "gold_chunk_ids": ["a"]},
        {"query_id": "q2", "retrieved_chunk_ids": ["b"], "silver_chunk_groups": [["b", "c"]]},
    ]))
    (folder / "memory.json").write_text('{"entity": "Aurora"}')
    (folder / "L_1.jsonl").write_text('{"summary": "Aurora is a team."}\n')
    (folder / "README.md").write_text("An obsolete experiment note")
    (folder / "retrieval_results__vanilla.json").write_text("[]")
    (folder / "report_index.json").write_text(json.dumps({
        "methods_executed": ["saadi", "vanilla"], "runs": [
            {"method": "saadi", "results_json": "retrieval_results.json"},
            {"method": "vanilla", "results_json": "retrieval_results__vanilla.json"}]}))
    reference = tmp_path / "reference"
    reference.mkdir()
    portable = reference / "ranking.jsonl.gz"
    with gzip.open(portable, "wt") as stream:
        for row in [
            {"query_id": "q1", "retrieved_chunk_ids": ["a"], "relevant_chunk_ids": ["a"], "evidence_groups": []},
            {"query_id": "q2", "retrieved_chunk_ids": ["b"], "relevant_chunk_ids": [], "evidence_groups": [["b", "c"]]},
        ]:
            stream.write(json.dumps(row) + "\n")
    identity = {"study": "main", "dataset": "loogle", "retriever": "bm25", "condition": "default", "source_id": "main_loogle"}
    source = {**identity, "path": portable.name, "sha256": sha256_file(portable),
              "original_source": "outputs/loogle_ablation/retrieval_results.json",
              "original_sha256": sha256_file(raw), "provenance": "per_query", "saved_depth": 10,
              "query_ids_sha256": hashlib.sha256(json.dumps(["q1", "q2"]).encode()).hexdigest()}
    (reference / "manifest.json").write_text(json.dumps({"schema_version": 1, "metric": "HR",
        "sources": [source], "statistics": [], "coverage": {}}))
    (reference / "expected_metrics.json").write_text(json.dumps([
        {**identity, "cutoff": k, "queries": 2, "eligible": 2, "hits": 1, "hr": 0.5} for k in (5, 10)]))
    (reference / "checksums.json").write_text(json.dumps({p.name: sha256_file(p) for p in reference.iterdir()}))
    return original, reference, tmp_path / "copied", source


def test_import_copies_full_caches_and_only_sadi_results(archive):
    original, reference, copied, _ = archive
    before = {str(p.relative_to(original)): sha256_file(p) for p in original.rglob("*") if p.is_file()}
    catalog = import_outputs(original, copied, reference, include_caches=True)
    assert catalog["includes_construction_caches"] is True
    assert (copied / "loogle_ablation/memory.json").read_bytes() == (original / "loogle_ablation/memory.json").read_bytes()
    assert (copied / "loogle_ablation/L_1.jsonl").is_file()
    assert not list(copied.rglob("*vanilla*"))
    assert not list(copied.rglob("README.md"))
    index = json.loads((copied / "loogle_ablation/report_index.json").read_text())
    assert index["methods_executed"] == ["saadi"]
    assert len(index["runs"]) == 1
    assert before == {str(p.relative_to(original)): sha256_file(p) for p in original.rglob("*") if p.is_file()}
    assert import_outputs(original, copied, reference, include_caches=True) == catalog
    (copied / "loogle_ablation/memory.json").write_text("changed")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        import_outputs(original, copied, reference, include_caches=True)


def test_tables_from_original_rankings_equal_portable_reference(archive, tmp_path):
    original, reference, copied, _ = archive
    import_outputs(original, copied, reference, include_caches=True)
    local = build_tables(reference, tmp_path / "local", outputs_root=copied, input_mode="outputs", studies=["main"])
    portable = build_tables(reference, tmp_path / "portable", input_mode="reference", studies=["main"])
    assert local["recomputed_cells"] == portable["recomputed_cells"] == 2
    assert local["input_source"] == "outputs"
    for filename in ("table.csv", "table.md", "table.tex", "metrics.csv"):
        assert (tmp_path / "local/main/sadi_retrieval" / filename).read_bytes() == (tmp_path / "portable/main/sadi_retrieval" / filename).read_bytes()
    assert not (tmp_path / "local/ablations").exists()
    assert not (tmp_path / "local/combined").exists()
    generator = tmp_path / "local/main/sadi_retrieval/generate.py"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, str(generator), "--reference", str(reference),
        "--outputs-root", str(copied), "--input-source", "outputs"], cwd=tmp_path,
        env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["verified_cells"] == 2


def test_missing_or_corrupt_outputs_never_silently_fall_back(archive):
    original, reference, copied, source = archive
    assert StudyInputs(reference, outputs_root=copied).mode == "reference"
    with pytest.raises(FileNotFoundError, match="Import"):
        StudyInputs(reference, outputs_root=copied, mode="outputs")
    import_outputs(original, copied, reference)
    assert not (copied / "loogle_ablation/memory.json").exists()
    inputs = StudyInputs(reference, outputs_root=copied)
    assert inputs.mode == "outputs"
    assert len(inputs.read(source)) == 2
    ranking = copied / "loogle_ablation/retrieval_results.json"
    ranking.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        inputs.read(source)
    ranking.unlink()
    with pytest.raises(ValueError, match="Missing"):
        inputs.read(source)
    assert len(StudyInputs(reference, outputs_root=copied, mode="reference").read(source)) == 2


def test_catalog_cannot_replace_recorded_study_inputs(archive):
    original, reference, copied, source = archive
    import_outputs(original, copied, reference)
    catalog_path = copied / "output_sources.json"
    catalog = json.loads(catalog_path.read_text())
    record = catalog["sources"][source["original_source"]]
    record["sha256"] = "0" * 64
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="differs from the saved study"):
        StudyInputs(reference, outputs_root=copied).read(source)
    catalog["reference_manifest_sha256"] = "1" * 64
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="another reference manifest"):
        StudyInputs(reference, outputs_root=copied)


@pytest.mark.parametrize("study,count", [("active_entity_budget", 16), ("active_entity_selector", 8)])
def test_paper_summaries_import_without_the_other_runs_artifacts(archive, tmp_path, study, count):
    """Paper transcriptions stay explicitly reported on both supported input routes."""
    from shutil import copyfile
    from saadi.reporting.layout import table_directory
    original, reference, copied, _ = archive
    public_reference = ROOT / "outputs/sadi/reference"
    manifest = json.loads((reference / "manifest.json").read_text())
    published = json.loads((public_reference / "manifest.json").read_text())
    source = next(s for s in published["sources"] if s["study"] == study)
    assert source["provenance"] == "paper_reported_summary"
    assert source["source_document"]["table"] == "6"
    (reference / source["path"]).parent.mkdir(exist_ok=True)
    copyfile(public_reference / source["path"], reference / source["path"])
    manifest["sources"].append(source)
    (reference / "manifest.json").write_text(json.dumps(manifest))
    expected_path = reference / "expected_metrics.json"
    expected = json.loads(expected_path.read_text())
    expected += [r for r in json.loads((public_reference / "expected_metrics.json").read_text()) if r["study"] == study]
    expected_path.write_text(json.dumps(expected))
    (reference / "checksums.json").write_text(json.dumps({str(p.relative_to(reference)): sha256_file(p)
        for p in reference.rglob("*") if p.is_file() and p.name != "checksums.json"}))
    import_outputs(original, copied, reference)
    for mode in ("reference", "outputs"):
        out = tmp_path / "tables" / mode
        report = build_tables(reference, out, outputs_root=copied, input_mode=mode, studies=[study])
        assert report["verified_cells"] == report["reported_cells"] == report["paper_reported_cells"] == count
        assert report["recomputed_cells"] == 0
        folder = table_directory(out, study)
        assert "paper_reported_summary" in (folder / "table.csv").read_text()
        evidence = json.loads((folder / "sources.json").read_text())[0]
        assert evidence["source_document"] == source["source_document"]
        assert "a transcription, not a recomputation" in (folder / "README.md").read_text()
    assert (table_directory(tmp_path / "tables/reference", study) / "table.csv").read_bytes() == (
        table_directory(tmp_path / "tables/outputs", study) / "table.csv").read_bytes()


def test_table6_has_paper_coverage_and_no_invented_query_counts():
    reference = ROOT / "outputs/sadi/reference"
    values = [r for r in json.loads((reference / "expected_metrics.json").read_text())
              if r["study"] in {"active_entity_budget", "active_entity_selector"}]
    assert len(values) == 24
    assert {r["dataset"] for r in values} == {"LooGLE", "NovelHopQA"}
    assert all(r["provenance"] == "paper_reported_summary" for r in values)
    assert all(r[k] is None for r in values for k in ("queries", "eligible", "hits"))
