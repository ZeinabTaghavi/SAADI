from pathlib import Path
import importlib.util
import json
import shutil
import hashlib
import subprocess
from types import SimpleNamespace
import sys

import pytest

from saadi.config.loader import load_config
from saadi.config.validation import validate_experiment
from saadi.evaluation.hr import evaluate_rows
from saadi.reporting.tables import build_tables, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("study_runner", ROOT / "scripts/run_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hr_complete_evidence_groups_and_eligible_denominator():
    rows = [
        {"query_id": "single", "retrieved_chunk_ids": ["a"], "relevant_chunk_ids": ["a"]},
        {"query_id": "group", "retrieved_chunk_ids": ["b", "c"], "evidence_groups": [["b", "c"]]},
        {"query_id": "partial", "retrieved_chunk_ids": ["b"], "evidence_groups": [["b", "c"]]},
        {"query_id": "unlabelled", "retrieved_chunk_ids": ["x"], "silver_chunk_ids": ["x"]},
    ]
    one, two = evaluate_rows(rows, cutoffs=(1, 2), retrieve_k=2)
    assert (one["queries"], one["eligible"], one["hits"]) == (4, 3, 1)
    assert two["hits"] == 2
    assert two["hr"] == pytest.approx(2 / 3)
    assert evaluate_rows([], cutoffs=(5,), retrieve_k=10)[0]["hr"] is None
    with pytest.raises(ValueError, match="saved"):
        evaluate_rows(rows, cutoffs=(10,), retrieve_k=5)
    with pytest.raises(ValueError, match="unique"):
        evaluate_rows(rows + rows)


def test_config_inheritance_and_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path))
    cfg = load_config(ROOT / "configs/experiments/loogle_retrieval_ablation.yaml")
    validate_experiment(cfg)
    assert cfg["ingest"]["strategy_kwargs"]["active_entity_selector"] == "bm25"
    assert cfg["model"]["sampling"]["seed"] == 42
    cfg["ingest"]["chunk_szie"] = 99
    with pytest.raises(ValueError, match="chunk_szie"):
        validate_experiment(cfg)
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_every_suite_configuration(tmp_path, monkeypatch):
    for name in ("SADI_OUTPUT_DIR", "SADI_DATA_DIR", "SADI_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name))
    module = runner()
    for path in (ROOT / "configs/suites").glob("*.yaml"):
        suite = load_config(path)
        for dataset in suite["datasets"]:
            base = load_config(ROOT / f"configs/experiments/{dataset}_retrieval_ablation.yaml")
            for variant in suite.get("variants", [{}]):
                cfg = module.merge(base, variant.get("overrides", {}))
                if variant.get("model_config"):
                    cfg["model"] = load_config(ROOT / variant["model_config"])
                validate_experiment(cfg)


def test_saved_profile_restores_recorded_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path))
    cfg = runner().published_config(load_config(ROOT / "configs/experiments/loogle_retrieval_ablation.yaml"))
    validate_experiment(cfg)
    assert cfg["ingest"]["strategy_kwargs"]["active_entity_selector"] == "word"
    assert cfg["model"]["generate"]["temperature"] == 0.6
    assert cfg["model"]["sampling"] == {"top_p": 0.95}


def test_stage_reuse_rejects_changed_artifacts_and_inputs(tmp_path):
    module = runner()
    stage = tmp_path / "stage"
    cmd = [sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(stage / "result")) + ").write_text('ok')"]
    module.execute_stage(cmd, stage, dry_run=False, inputs={"model": "a"})
    module.execute_stage(["must-not-execute"], stage, dry_run=False, inputs={"model": "a"})
    with pytest.raises(ValueError, match="changed"):
        module.execute_stage(cmd, stage, dry_run=False, inputs={"model": "b"})
    (stage / "result").write_text("corrupted")
    with pytest.raises(ValueError, match="changed"):
        module.execute_stage(cmd, stage, dry_run=False, inputs={"model": "a"})
    module.execute_stage(cmd, tmp_path / "dry", dry_run=True, inputs={})
    assert not (tmp_path / "dry").exists()


def test_frozen_results_reproduce_and_detect_corruption(tmp_path):
    reference = ROOT / "outputs/sadi/reference"
    report = build_tables(reference, tmp_path / "tables")
    assert report["verified_cells"] == 468
    assert report["recomputed_cells"] == 276
    assert report["reported_cells"] == 192
    assert report["paper_reported_cells"] == 24
    assert "HR@5 (%)" in (tmp_path / "tables/main/sadi_retrieval/table.md").read_text()
    from saadi.reporting.layout import TABLES
    for relative, _ in TABLES.values():
        directory = tmp_path / "tables" / relative
        assert (directory / "table.csv").is_file()
        assert (directory / "generate.py").is_file()
        assert json.loads((directory / "sources.json").read_text())
    assert (tmp_path / "tables/combined/all_metrics.csv").is_file()
    assert not list((tmp_path / "tables").glob("*.csv"))
    copied = tmp_path / "corrupted"
    shutil.copytree(reference, copied)
    ranking = next((copied / "rankings").iterdir())
    ranking.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(copied)
    ranking.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_bundle(copied)


def test_offline_pipeline(tmp_path):
    spec = importlib.util.spec_from_file_location("smoke", ROOT / "scripts/smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_smoke(tmp_path / "smoke")["hit_rate"] == 1.0
    from saadi.saadi_generation.run_saadi_generation import main as generate
    smoke = tmp_path / "smoke"
    result = next((smoke / "retrieval").rglob("retrieval_results__*.json"))
    for experiment in ("folded_context_block", "folded_context_nested",
                       "folded_context_local_bridge", "folded_context_generated_bridge"):
        assert generate(["--config", str(smoke / "config.json"), "--dataset", "synthetic",
                         "--retriever", "bm25", "--experiment", experiment, "--top_k", "5", "--summary_k", "2",
                         "--dataset_experiment_dir", str(tmp_path / "generation"),
                         "--mbr_results_json", str(result), "--no_auto_mbr", "--no_eval"]) == 0
    generated = list((tmp_path / "generation").rglob("generations.jsonl"))
    assert len(generated) == 4
    for path in generated:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert all(row["status"] == "ok" for row in rows)


def test_preserved_scientific_sources():
    for row in json.loads((ROOT / "docs/preserved_sources.json").read_text()):
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]


def test_stage_lock_refuses_simultaneous_writes(tmp_path):
    module = runner()
    with module.stage_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            module.execute_stage(["must-not-execute"], tmp_path, dry_run=False, inputs={})


def test_preparation_config_is_shared_across_scoring_variants(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path))
    cfg = load_config(ROOT / "configs/experiments/qasper_retrieval_ablation.yaml")
    first = module.prepare_config(module.merge(cfg, {}), dry_run=False, execute_prepare=False)
    variant = module.merge(cfg, {"cached_retrieval": {"lambda_n": 0.7}})
    second = module.prepare_config(variant, dry_run=False, execute_prepare=False)
    assert first["dataset"]["documents_json"] == second["dataset"]["documents_json"]


def test_public_ranking_and_report_schema_roundtrip(tmp_path):
    from saadi.core.types import RetrievalResult
    from saadi.evaluation.retrieval_metrics import evaluate_retrieval_results, load_retrieval_results_json_with_metadata
    result = RetrievalResult(query_id="q", doc_id="d", question="?", retrieved_chunk_ids=["a", "b"],
                             retrieved_indices=[0, 1], scores=[2.0, 1.0], silver_chunk_groups=[["a", "b"]])
    row = result.to_dict()
    assert row["evidence_groups"] == [["a", "b"]]
    assert not any("gold" in k or "silver" in k for k in row)
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps([row]))
    loaded, _ = load_retrieval_results_json_with_metadata(path)
    report = evaluate_retrieval_results(loaded, eval_k=2, retrieve_k=2).to_dict()
    assert report["hit_rate"] == 1.0
    assert report["hit_rate_at_k"]["5"] is None
    assert not any("gold" in k or "silver" in k or "union" in k for k in report)


def test_real_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("SADI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SADI_CACHE_DIR", str(tmp_path / "cache"))
    subprocess.run([sys.executable, str(ROOT / "scripts/run_study.py"), "--datasets", "qasper", "--retrievers", "bm25", "--dry-run"], check=True, capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


def test_study_runner_constructs_then_reuses_verified_stages(tmp_path, monkeypatch, capsys):
    import yaml
    module = runner()
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path / "outputs"))
    module.ROOT = tmp_path
    (tmp_path / "configs/suites").mkdir(parents=True)
    (tmp_path / "configs/experiments").mkdir()
    (tmp_path / "configs/suites/main.yaml").write_text("datasets: [synthetic]\nretrievers: [bm25]\n")
    (tmp_path / "documents.json").write_text(json.dumps({"doc": "Mira leads the Aurora team."}))
    (tmp_path / "qa.json").write_text(json.dumps([
        {"id": "q1", "document_id": "doc", "question": "Who leads Aurora?", "retrieval_spans": ["Mira"]},
        {"id": "q2", "document_id": "doc", "question": "Which team does Mira lead?", "retrieval_spans": ["Aurora"]}]))
    cfg = {"output_dir": str(tmp_path / "outputs"),
           "dataset": {"name": "synthetic", "source": "files", "documents_json": str(tmp_path / "documents.json"), "qa_json": str(tmp_path / "qa.json")},
           "model": {"backend": "mock"},
           "sample": {"max_qa_entries": 1},
           "ingest": {"strategy": "hierarchical", "enabled": True, "chunk_size": 100},
           "retrieval": {"mode": "canonicalized", "retrieve_k": 10, "eval_k": 5, "scope": "per_document"}}
    (tmp_path / "configs/experiments/synthetic_retrieval_ablation.yaml").write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(suite="main", datasets=None, retrievers=None, profile="new", tensor_parallel_size=None, max_documents=None, dry_run=False, stage="all")
    assert module.run(args) == 0
    assert len(list((tmp_path / "outputs").rglob(".stage.json"))) == 2
    assert module.run(args) == 0
    assert capsys.readouterr().out.count("Verified completed stage:") == 2
    result_path = next((tmp_path / "outputs").rglob("retrieval_results__*.json"))
    assert len(json.loads(result_path.read_text())) == 1
    index = next((tmp_path / "outputs/indexes").iterdir())
    (index / "tampered.txt").write_text("unexpected construction output")
    args.stage = "retrieve"
    with pytest.raises(ValueError, match="Completed stage changed"):
        module.run(args)
    (index / ".stage.json").unlink()
    with pytest.raises(FileNotFoundError, match="completed construction stage"):
        module.run(args)
