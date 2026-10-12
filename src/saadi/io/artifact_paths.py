"""Defines standardized output artifact paths for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ArtifactPaths:
    root: Path
    raw_chunks_jsonl: Path
    llm_cache_dir: Path
    llm_cache_index_json: Path
    retrieval_results_json: Path
    eval_report_json: Path
    run_manifest_json: Path



def default_artifact_paths(output_dir: str | Path) -> ArtifactPaths:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        root=root,
        raw_chunks_jsonl=root / "raw_chunks.jsonl",
        llm_cache_dir=root / "llm_cache",
        llm_cache_index_json=root / "llm_cache" / "index.json",
        retrieval_results_json=root / "retrieval_results.json",
        eval_report_json=root / "eval_report.json",
        run_manifest_json=root / "run_manifest.json",
    )
