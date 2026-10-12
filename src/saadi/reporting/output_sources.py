"""Import and read the original Saadi experiment outputs used by the tables."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

DATASETS = ("loogle", "qasper", "novelhopqa", "musique")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Output path escapes its directory: {relative}")
    return path


def local_source_path(source):
    original = Path(source["original_source"])
    if original.parts[0] == "outputs" and original.parts[1] in {f"{d}_ablation" for d in DATASETS}:
        return Path(*original.parts[1:])
    if source["provenance"] == "per_query":
        return Path(f"{source['dataset']}_ablation") / "table_inputs" / source["study"] / source["source_id"] / original.name
    return Path("study_sources") / source["study"] / source["source_id"] / "values.json"


def is_baseline(value):
    text = str(value).lower()
    return any(word in text for word in ("vanilla", "baseline", "raptor", "agent_r", "a_rag", "grip"))


def clean_report_index(payload):
    payload = dict(payload)
    if "runs" in payload:
        payload["runs"] = [row for row in payload["runs"]
                           if not is_baseline(row.get("method", ""))
                           and not is_baseline(row.get("results_json", ""))]
    if isinstance(payload.get("methods_executed"), list):
        payload["methods_executed"] = [method for method in payload["methods_executed"] if not is_baseline(method)]
    return payload


def import_outputs(source_root, output_root, reference, *, include_caches=False):
    """Copy archived outputs; never remove or modify the source experiment directory."""
    from .tables import verify_bundle
    source_root, output_root, reference = map(lambda p: Path(p).resolve(), (source_root, output_root, reference))
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("The destination must be outside the source outputs")
    manifest = verify_bundle(reference)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = {}

    def copy(source, relative, *, filter_index=False):
        destination = safe_path(output_root, relative)
        source_hash = sha256_file(source)
        data = None
        if filter_index:
            payload = json.loads(source.read_text())
            filtered = clean_report_index(payload)
            if filtered != payload:
                data = (json.dumps(filtered, indent=2) + "\n").encode()
        expected = hashlib.sha256(data).hexdigest() if data is not None else source_hash
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != expected:
                raise ValueError(f"Refusing to overwrite different output: {destination}")
        else:
            temporary = destination.with_name(destination.name + ".copying")
            try:
                if data is None:
                    shutil.copyfile(source, temporary)
                else:
                    temporary.write_bytes(data)
                if sha256_file(temporary) != expected:
                    raise ValueError(f"Copy verification failed: {source}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        inventory[str(relative)] = {"sha256": expected, "bytes": destination.stat().st_size,
                                    "source_sha256": source_hash,
                                    "transform": "Saadi rows only" if data is not None else "byte-identical"}
        return inventory[str(relative)]

    for dataset in DATASETS:
        folder = source_root / f"{dataset}_ablation"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        count_before = len(inventory)
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source_root)
            if is_baseline(relative) or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.suffix in {".md", ".txt", ".log", ".py", ".pyc", ".sh"} or path.name.startswith("variance_report"):
                continue
            required = (path.name.startswith(("retrieval_results", "eval_report"))
                        or path.name in {"report_index.json", "run_manifest.json", "runtime_config.yaml"}
                        or "runtime_configs" in path.parts)
            if include_caches or required:
                copy(path, relative, filter_index=path.name == "report_index.json")
        print(f"Copied {dataset}: {len(inventory) - count_before} verified files", flush=True)

    sources = {}
    for source in manifest["sources"]:
        original = source["original_source"]
        if original in sources:
            continue
        target = local_source_path(source)
        if source["provenance"] == "per_query":
            path = safe_path(source_root, str(Path(original).relative_to("outputs")))
            if sha256_file(path) != source["original_sha256"]:
                raise ValueError(f"Original study output changed: {path}")
            info = inventory.get(str(target)) or copy(path, target)
            kind = "original_rankings"
        else:
            # The reference summaries contain only Saadi, including filtered multi-method source tables.
            info = copy(reference / source["path"], target)
            kind = "paper_reported_summary" if source["provenance"] == "paper_reported_summary" else "reported_summary"
            if kind != "paper_reported_summary":
                original_path = safe_path(source_root, str(Path(original).relative_to("outputs")))
                if original_path.is_file() and source["study"] != "multihop":
                    if sha256_file(original_path) != source["original_sha256"]:
                        raise ValueError(f"Original summary changed: {original_path}")
                    copy(original_path, target.with_name("source_values.csv"))
        sources[original] = {"path": str(target), "sha256": info["sha256"], "kind": kind,
                             "original_sha256": source["original_sha256"]}
    catalog = {"schema_version": 1, "method": "SADI", "includes_construction_caches": include_caches,
               "reference_manifest_sha256": sha256_file(reference / "manifest.json"),
               "sources": sources, "files": inventory,
               "total_files": len(inventory), "total_bytes": sum(row["bytes"] for row in inventory.values())}
    path = output_root / "output_sources.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(catalog, indent=2) + "\n")
    temporary.replace(path)
    return catalog


def read_rankings(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    return [{"query_id": str(row["query_id"]),
             "retrieved_chunk_ids": [str(x) for x in sadi_ranking(row)],
             "relevant_chunk_ids": [str(x) for x in row.get("relevant_chunk_ids", row.get("gold_chunk_ids", []))],
             "evidence_groups": row.get("evidence_groups", row.get("silver_chunk_groups", []))}
            for row in rows]


def sadi_ranking(row):
    """Read known archive schemas without interpreting missing rankings as failures."""
    for field in ("retrieved_chunk_ids", "ordered_retrieved_chunk_ids", "saadi"):
        if field in row:
            ranking = row[field]
            break
    else:
        # Context-expansion archives retain each scoring configuration separately.
        configuration = row.get("configurations", {}).get("full_saadi", {})
        if "retrieved_chunk_ids" not in configuration:
            raise ValueError(f"Missing Saadi ranking for query {row.get('query_id')!r}")
        ranking = configuration["retrieved_chunk_ids"]
    if not isinstance(ranking, list):
        raise ValueError(f"Saadi ranking must be a list for query {row.get('query_id')!r}")
    return ranking


class StudyInputs:
    def __init__(self, reference, *, outputs_root=None, mode="auto"):
        if mode not in {"auto", "outputs", "reference"}:
            raise ValueError(f"Unknown input source: {mode}")
        self.reference = Path(reference).resolve()
        self.outputs_root = Path(outputs_root).resolve() if outputs_root is not None else None
        catalog_path = self.outputs_root / "output_sources.json" if self.outputs_root else None
        self.catalog = None
        self.records = []
        if mode == "outputs" and (catalog_path is None or not catalog_path.is_file()):
            raise FileNotFoundError("Import the Saadi outputs first with scripts/import_study_outputs.py")
        if mode != "reference" and catalog_path is not None and catalog_path.is_file():
            self.catalog = json.loads(catalog_path.read_text())
            if self.catalog.get("schema_version") != 1 or self.catalog.get("method") != "SADI":
                raise ValueError("Unsupported output source catalog")
            if self.catalog["reference_manifest_sha256"] != sha256_file(self.reference / "manifest.json"):
                raise ValueError("The copied outputs belong to another reference manifest")
        self.mode = "outputs" if self.catalog is not None else "reference"

    def read(self, source):
        if self.catalog is None:
            path = safe_path(self.reference, source["path"])
            expected = source["sha256"]
            relative = "outputs/sadi/reference/" + source["path"]
        else:
            record = self.catalog["sources"].get(source["original_source"])
            if record is None:
                raise FileNotFoundError(f"Missing copied study source: {source['original_source']}")
            path = safe_path(self.outputs_root, record["path"])
            expected = record["sha256"]
            if source["provenance"] == "per_query" and expected != source["original_sha256"]:
                raise ValueError(f"Copied ranking differs from the saved study: {record['path']}")
            if source["provenance"] != "per_query" and expected != source["sha256"]:
                raise ValueError(f"Copied summary differs from the saved study: {record['path']}")
            relative = "outputs/" + record["path"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or changed study output: {relative}")
        self.records.append({"study": source["study"], "source_id": source["source_id"],
                             "path": relative, "sha256": expected, "provenance": source["provenance"],
                             **{key: source[key] for key in ("dataset", "retriever", "condition", "memory_selector", "source_document", "note") if key in source}})
        if source["provenance"] == "per_query":
            return read_rankings(path)
        return json.loads(path.read_text())
