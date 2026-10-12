from pathlib import Path
import subprocess

import pytest

from saadi.cli import retrieve_cached
from test_release import ROOT, runner


@pytest.mark.parametrize("arguments", [[], ["--retrieve_k", "0"]])
def test_cli_preserves_explicit_zero_values(monkeypatch, arguments):
    captured = {}

    class StopBeforeInference(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise StopBeforeInference

    monkeypatch.setattr(retrieve_cached, "run_cached_retrieval_eval", capture)
    with pytest.raises(StopBeforeInference):
        retrieve_cached.main(["--llm_cache_index_json", "index.json", "--qa_json", "qa.json",
                              "--max_aliases", "0", "--max_desc", "0", *arguments])
    assert captured["max_aliases"] == captured["max_desc"] == 0
    assert captured["retrieval_scope"] == "per_document"
    # Invalid zero depths reach the pipeline's validator instead of becoming 10.
    assert captured["retrieve_k"] == (0 if arguments else 10)


def test_shell_launcher_preserves_custom_corpus_path(tmp_path, monkeypatch):
    corpus = str(tmp_path / "my books")
    monkeypatch.setenv("NOVELHOPQA_BOOKS_ROOT", corpus)
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$NOVELHOPQA_BOOKS_ROOT"',
                             "bash", str(ROOT / "scripts/env.sh")], check=True, capture_output=True, text=True)
    assert result.stdout == corpus


def test_dataset_paths_survive_runtime_config_relocation(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVELHOPQA_BOOKS_ROOT", str(tmp_path / "custom books"))
    cfg = {"dataset": {"name": "novelhopqa", "books_root": "/unused/default",
                       "documents_json": "data/documents.json", "qa_json": "data/qa.json"}}
    runner().resolve_dataset_paths(cfg, tmp_path)
    assert cfg["dataset"]["books_root"] == str(tmp_path / "custom books")
    assert cfg["dataset"]["documents_json"] == str(tmp_path / "data/documents.json")
    assert cfg["dataset"]["qa_json"] == str(tmp_path / "data/qa.json")


def test_preparation_identity_changes_when_local_source_changes(tmp_path, monkeypatch):
    from copy import deepcopy
    module = runner()
    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path / "outputs"))
    source = tmp_path / "documents.json"
    source.write_text('{"doc": "first version"}')
    cfg = {"dataset": {"name": "qasper", "source": "files", "documents_json": str(source)},
           "preparation": {"target_context_tokens": 100}}
    first = module.prepare_config(deepcopy(cfg), dry_run=False, execute_prepare=False)
    source.write_text('{"doc": "second version"}')
    second = module.prepare_config(deepcopy(cfg), dry_run=False, execute_prepare=False)
    assert first["dataset"]["documents_json"] != second["dataset"]["documents_json"]
