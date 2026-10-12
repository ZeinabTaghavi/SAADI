"""Validated CSV inputs for comparison methods; this module never runs a method."""
import csv
import json
import math
from pathlib import Path

from .output_sources import safe_path, sha256_file

DATASET_LABELS = {"loogle": "LooGLE", "musique": "MuSiQue-32K", "qasper": "QASPER-64K", "novelhopqa": "NovelHopQA"}
RETRIEVER_LABELS = {"bm25": "BM25", "contriever": "Contriever", "bge_m3": "BGE-M3", "jina": "Jina", "qwen": "Qwen"}
METHOD_LABELS = {"sadi": "Saadi", "vanilla": "Vanilla", "llamaindex": "LlamaIndex", "lqca": "LQCA",
                 "late_chunking": "Late Chunking", "hipporag_v2": "HippoRAG v2", "raptor": "RAPTOR (Upper Bound)",
                 "rag_r1": "RAG-R1", "agentic_r": "Agentic-R", "a_rag": "A-RAG", "grip": "GRIP", "uma": "UMA"}
AGENTS = ("rag_r1", "agentic_r", "a_rag", "grip", "uma")
RETRIEVAL_FIELDS = ("study", "method", "dataset", "condition", "retriever", "variant", "cutoff", "hop",
                    "hr_percent", "mean_chunks", "queries", "eligible_queries", "hits", "provenance", "source_id")
RESOURCE_FIELDS = ("method", "dataset", "condition", "queries", "documents", "mean_chunks", "calls_per_query",
                   "input_tokens_per_query", "generated_tokens_per_query", "offline_calls_per_document",
                   "offline_input_tokens_per_document", "offline_generated_tokens_per_document",
                   "input_tokens_basis", "generated_tokens_basis", "provenance", "source_id")
IDENTITY = ("study", "method", "dataset", "condition", "retriever", "variant", "cutoff", "hop")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV column names: {path}")
        return list(reader)


def number(row, name, *, required=True):
    value = row[name]
    if value == "" and not required:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {name}: {value}")
    return result


def load_other_methods(root, *, update_checksums=False):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported comparison-input schema")
    expected_files = set(manifest["files"])
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*.csv")}
    if actual_files != expected_files:
        raise ValueError("Comparison input files are missing or unregistered")
    retrieval, resources, evidence = [], [], []
    identities = set()
    updates = {}
    for name, recorded_hash in manifest["files"].items():
        path = safe_path(root, name)
        digest = sha256_file(path)
        if not update_checksums and digest != recorded_hash:
            raise ValueError(f"Comparison input checksum changed: {name}")
        updates[name] = digest
        kind = path.stem
        if kind not in {"retrieval", "resources"}:
            raise ValueError(f"Unknown comparison CSV: {name}")
        rows = read_csv(path)
        required = set(RETRIEVAL_FIELDS if kind == "retrieval" else RESOURCE_FIELDS)
        if not rows:
            raise ValueError(f"Empty comparison CSV: {name}")
        for row in rows:
            if set(row) != required or None in row.values():
                raise ValueError(f"Unexpected or incomplete CSV columns: {name}")
            method = row["method"]
            if method == "sadi" or method not in METHOD_LABELS or method != path.parent.name:
                raise ValueError(f"Wrong method in {name}: {method}")
            if row["dataset"] not in DATASET_LABELS or row["source_id"] not in manifest["sources"]:
                raise ValueError(f"Unknown dataset/source in {name}")
            if row["provenance"] != "reported_summary":
                raise ValueError("External CSVs are saved summaries, not locally recomputed query results")
            identity = (kind,) + tuple(row[k] for k in (IDENTITY if kind == "retrieval" else ("method", "dataset", "condition")))
            if identity in identities:
                raise ValueError(f"Duplicate comparison result: {identity}")
            identities.add(identity)
            for field in ("queries", "eligible_queries", "hits", "documents"):
                if field in row and row[field] != "":
                    value = number(row, field)
                    if value != int(value):
                        raise ValueError(f"Noninteger {field} in {name}")
            number(row, "mean_chunks", required=kind == "resources" or row["study"] != "main")
            if kind == "retrieval":
                if row["study"] not in {"main", "agentic", "context_expansion", "multihop"}:
                    raise ValueError(f"Unknown comparison study: {row['study']}")
                if row["cutoff"] and (not row["cutoff"].isdigit() or int(row["cutoff"]) < 1):
                    raise ValueError("A cutoff must be a positive integer or blank for a trajectory")
                if row["retriever"] not in {*RETRIEVER_LABELS, "native"}:
                    raise ValueError("Unknown retriever")
                if (method in AGENTS) != (row["cutoff"] == ""):
                    raise ValueError("Only agentic methods use an empty trajectory cutoff")
                if row["condition"] not in ({"no_expansion", *(f"{k}K_expansion" for k in (32, 64, 128, 256))}
                                             if row["study"] == "context_expansion" else {"standard"}):
                    raise ValueError("Unknown condition for this study")
                if row["hop"] not in ({"overall", "1-hop", "2-hop", "3-hop", "4-hop"} if row["study"] == "multihop" else {"overall"}):
                    raise ValueError("Unknown hop group for this study")
                if row["variant"] not in ({"no_overlap", "low_overlap", "more_overlap"} if method == "late_chunking" else {"default"}):
                    raise ValueError("Unknown method variant")
                rate = number(row, "hr_percent")
                if rate > 100:
                    raise ValueError("HR percent must be between 0 and 100")
                eligible, hits = number(row, "eligible_queries", required=False), number(row, "hits", required=False)
                queries = number(row, "queries", required=False)
                if (eligible is None) != (hits is None):
                    raise ValueError("Hit and eligible-query counts must be supplied together")
                if queries is not None and (queries == 0 or (eligible is not None and eligible > queries)):
                    raise ValueError("Invalid query population")
                if eligible is not None and hits is not None:
                    if not eligible or hits > eligible or abs(100 * hits / eligible - rate) > 1e-8:
                        raise ValueError(f"HR disagrees with its hit/eligible counts in {name}")
                retrieval.append(row)
            else:
                if method not in AGENTS or row["condition"] != "standard" or not number(row, "queries") or not number(row, "documents"):
                    raise ValueError("Invalid resource comparison scope")
                for field in RESOURCE_FIELDS:
                    if field.endswith(("_per_query", "_per_document")):
                        number(row, field)
                resources.append(row)
        evidence.append({"path": "outputs/other_methods/" + name, "sha256": digest, "rows": len(rows),
                         "source_ids": sorted({row["source_id"] for row in rows})})
    if update_checksums:
        manifest["files"] = updates
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"retrieval": retrieval, "resources": resources, "evidence": evidence, "sources": manifest["sources"]}
