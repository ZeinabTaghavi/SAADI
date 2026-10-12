#!/usr/bin/env python3
"""Package a selected Saadi output directory with hashes for separate distribution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = output.with_name(output.name + ".manifest.json")
    if not source.is_dir() or output.is_relative_to(source):
        raise ValueError("Select a source directory and an archive path outside it")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("The archive or its manifest already exists")
    paths = sorted(source.rglob("*"))
    if any(p.is_symlink() or (not p.is_file() and not p.is_dir()) for p in paths):
        raise ValueError("Bundles support regular files and directories only")
    records = {str(p.relative_to(source)): {"sha256": sha256(p), "bytes": p.stat().st_size}
               for p in paths if p.is_file() and p.name != ".run.lock" and "__pycache__" not in p.parts}
    if not records:
        raise ValueError("The source contains no output files")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name in records:
                archive.add(source / name, arcname="sadi_outputs/" + name, recursive=False)
                if sha256(source / name) != records[name]["sha256"]:
                    raise ValueError(f"Output changed while packaging: {name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {"schema_version": 1, "archive": output.name, "sha256": sha256(output), "files": records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(bundle(args.source, args.output))
