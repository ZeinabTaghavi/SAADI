from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from saadi.core.types import ChunkRecord, RetrievalResult
from saadi.evaluation.retrieval_metrics import (
    evaluate_retrieval_results,
    load_retrieval_results_json_with_metadata,
    write_eval_report_json,
)
from saadi.io import manifests
from saadi.llm.base import GenerateParams
from saadi.llm import factory
from saadi.llm.vllm_backend import VLLMBackend
from saadi.pipelines import retrieval_pipeline, single_document_pipeline


PROVENANCE = {
    "git_commit": "code-commit",
    "git_dirty": True,
    "git_worktree_diff_sha256": "a" * 64,
}


class _FakeVLLMModel:
    def __init__(self) -> None:
        self.params = None

    def generate(self, prompts, params):
        self.params = params
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="ok")]) for _ in prompts]


def test_vllm_backend_uses_deterministic_sampling_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))
    model = _FakeVLLMModel()
    backend = VLLMBackend(model=model)

    assert backend.generate_batch(
        ["prompt"],
        max_tokens=123,
        temperature=0.0,
        stop=["[END_JSON]"],
    ) == ["ok"]
    assert model.params.kwargs == {
        "seed": 42,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 123,
        "temperature": 0.0,
        "stop": ["[END_JSON]"],
    }


def test_ingestion_manifest_records_resolved_reproducibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifests, "git_provenance", lambda cwd=None: dict(PROVENANCE))
    backend = types.SimpleNamespace(
        runtime_metadata={
            "model_name": "org/model",
            "hf_revision": "main",
            "hf_commit_hash": "model-commit",
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
            "vllm_version": "0.10.2",
            "sampling": {
                "seed": 42,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
            },
        }
    )
    manifest = manifests.build_manifest(
        config={
            "model": {
                "backend": "vllm",
                "model_id": "org/model",
                "dtype": "bfloat16",
                "tensor_parallel_size": 4,
                "generate": {"max_tokens": 2048, "temperature": 0.0},
            }
        },
        artifacts={},
        backend=backend,
        generate=GenerateParams(max_tokens=2048, temperature=0.0),
    )

    assert manifest.metadata["git_commit"] == "code-commit"
    assert manifest.metadata["git_dirty"] is True
    assert manifest.metadata["git_worktree_diff_sha256"] == "a" * 64
    reproducibility = manifest.metadata["reproducibility"]
    assert reproducibility == {
        "git_commit": "code-commit",
        "git_dirty": True,
        "git_worktree_diff_sha256": "a" * 64,
        "model_name": "org/model",
        "hf_revision": "main",
        "hf_commit_hash": "model-commit",
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "vllm_version": "0.10.2",
        "max_tokens": 2048,
        "sampling": {
            "temperature": 0.0,
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "top_k_enabled": False,
        },
    }


def test_vllm_factory_records_resolved_model_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            model_config = types.SimpleNamespace(
                model=kwargs["model"],
                revision=kwargs.get("revision"),
                dtype="bfloat16",
                hf_config=types.SimpleNamespace(_commit_hash="resolved-hf-commit"),
            )
            parallel_config = types.SimpleNamespace(tensor_parallel_size=4)
            self.llm_engine = types.SimpleNamespace(
                model_config=model_config,
                parallel_config=parallel_config,
            )

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM))
    monkeypatch.setattr(factory, "load_hf_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(factory, "_package_version", lambda name: "0.10.2")

    backend = factory.build_backend_from_config(
        {
            "backend": "vllm",
            "model_id": "org/model",
            "revision": "requested-revision",
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
        }
    )

    assert backend.model.kwargs["revision"] == "requested-revision"
    assert backend.runtime_metadata == {
        "backend": "vllm",
        "model_name": "org/model",
        "hf_revision": "requested-revision",
        "hf_commit_hash": "resolved-hf-commit",
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "vllm_version": "0.10.2",
        "sampling": {
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
    }


def test_retrieval_artifacts_record_git_commit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval_provenance = {
        **PROVENANCE,
        "git_commit": "retrieval-commit",
    }
    monkeypatch.setattr(
        retrieval_pipeline,
        "git_provenance",
        lambda cwd=None: dict(retrieval_provenance),
    )
    result = RetrievalResult(
        query_id="q1",
        doc_id="d1",
        question="question",
        retrieved_chunk_ids=["d1::0"],
        retrieved_indices=[0],
        scores=[1.0],
        gold_chunk_ids=["d1::0"],
        silver_chunk_ids=[],
        silver_chunk_groups=[],
    )
    results_path = tmp_path / "retrieval_results.json"
    retrieval_pipeline.write_retrieval_results_json(results_path, [result])

    rows = json.loads(results_path.read_text(encoding="utf-8"))
    assert rows[0]["git_commit"] == "retrieval-commit"
    assert rows[0]["git_dirty"] is True
    assert rows[0]["git_worktree_diff_sha256"] == "a" * 64
    _, metadata = load_retrieval_results_json_with_metadata(results_path)
    assert metadata["git_commit"] == "retrieval-commit"
    assert metadata["git_dirty"] is True
    assert metadata["git_worktree_diff_sha256"] == "a" * 64

    report = evaluate_retrieval_results(
        [result], metadata=retrieval_provenance
    )
    report_path = tmp_path / "eval_report.json"
    write_eval_report_json(report_path, report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["git_commit"] == "retrieval-commit"
    assert payload["git_dirty"] is True
    assert payload["git_worktree_diff_sha256"] == "a" * 64
    assert payload["metadata"]["git_commit"] == "retrieval-commit"
    assert payload["metadata"]["git_dirty"] is True
    assert payload["metadata"]["git_worktree_diff_sha256"] == "a" * 64


def test_git_worktree_fingerprint_covers_tracked_and_untracked_content(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "SAADI Tests"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    clean = manifests.git_provenance(str(tmp_path), allow_environment=False)
    assert clean["git_dirty"] is False
    assert len(clean["git_worktree_diff_sha256"]) == 64

    tracked.write_text("value = 2\n", encoding="utf-8")
    tracked_change = manifests.git_provenance(str(tmp_path), allow_environment=False)
    assert tracked_change["git_dirty"] is True
    assert tracked_change["git_worktree_diff_sha256"] != clean["git_worktree_diff_sha256"]

    untracked = tmp_path / "new_config.yaml"
    untracked.write_text("seed: 42\n", encoding="utf-8")
    with_untracked = manifests.git_provenance(str(tmp_path), allow_environment=False)
    assert with_untracked["git_worktree_diff_sha256"] != tracked_change["git_worktree_diff_sha256"]
    assert manifests.git_provenance(
        str(tmp_path), allow_environment=False
    ) == with_untracked

    untracked.write_text("seed: 43\n", encoding="utf-8")
    changed_untracked = manifests.git_provenance(str(tmp_path), allow_environment=False)
    assert changed_untracked["git_worktree_diff_sha256"] != with_untracked["git_worktree_diff_sha256"]


def test_single_document_manual_manifest_records_reproducibility(tmp_path) -> None:
    config = single_document_pipeline.SingleDocumentPipelineConfig(
        demo_root=str(tmp_path),
        run_name="manual-manifest",
        model_config={
            "backend": "vllm",
            "model_id": "org/model",
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
            "sampling": {
                "seed": 42,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
            },
        },
        ingest_max_tokens=2048,
        ingest_temperature=0.0,
    )
    backend = types.SimpleNamespace(
        runtime_metadata={
            "model_name": "org/model",
            "hf_revision": "main",
            "hf_commit_hash": "model-commit",
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
            "vllm_version": "0.10.2",
            "sampling": {
                "seed": 42,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
            },
        }
    )

    path = single_document_pipeline.save_demo_artifacts(
        config,
        backend=backend,
        provenance=PROVENANCE,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert {field: payload[field] for field in PROVENANCE} == PROVENANCE
    reproducibility = payload["reproducibility"]
    assert {field: reproducibility[field] for field in PROVENANCE} == PROVENANCE
    assert reproducibility["model_name"] == "org/model"
    assert reproducibility["hf_commit_hash"] == "model-commit"
    assert reproducibility["max_tokens"] == 2048
    assert reproducibility["sampling"]["temperature"] == 0.0


def test_single_document_retrieval_result_records_provenance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = single_document_pipeline.SingleDocumentPipelineConfig(
        demo_root=str(tmp_path),
        run_name="retrieval-report",
        document_id="doc",
        query="Question?",
    )
    raw_chunk = ChunkRecord(
        doc_id="doc",
        chunk_id="doc:0",
        chunk_index=0,
        raw_text="Evidence.",
    )
    result = RetrievalResult(
        query_id="query_1",
        doc_id="doc",
        question="Question?",
        retrieved_chunk_ids=["doc:0"],
        retrieved_indices=[0],
        scores=[1.0],
    )
    monkeypatch.setattr(
        single_document_pipeline,
        "git_provenance",
        lambda cwd=None: dict(PROVENANCE),
    )
    monkeypatch.setattr(
        single_document_pipeline,
        "_find_cache_index_path",
        lambda config, artifacts: tmp_path / "index.json",
    )
    monkeypatch.setattr(
        single_document_pipeline,
        "build_mbr_document_memory_index",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        single_document_pipeline,
        "run_mbr_variant_retrieval_pipeline",
        lambda **kwargs: ([result], {}, "mbr_test"),
    )

    retrieval = single_document_pipeline.retrieve_for_query(
        config,
        [raw_chunk],
        [],
    )
    payload = json.loads(
        retrieval.retrieval_results_path.read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        retrieval.retrieval_diagnostics_path.read_text(encoding="utf-8")
    )
    assert {field: payload[field] for field in PROVENANCE} == PROVENANCE
    assert {field: diagnostics[field] for field in PROVENANCE} == PROVENANCE
