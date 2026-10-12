"""Loads YAML or JSON config files into Python dictionaries."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def _expand_paths(text: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Set {name} before loading this configuration; see README.md.")
        return os.environ[name]
    return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", replace, text)


def _expand_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_paths(value)
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_values(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML configs. Install with `pip install -r requirements.txt`.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = _expand_values(yaml.safe_load(f) or {})
    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dict: {path}")
    return data


def merge_config(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path: str | Path, *, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    p = Path(path).resolve()
    if p in _parents:
        raise ValueError(f"Cyclic configuration inheritance: {p}")
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(p)
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = _expand_values(json.load(f))
        if not isinstance(data, dict):
            raise ValueError(f"Config file must parse to a dict: {path}")
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        raise ValueError("extends must be a path or a list of paths")
    result = {}
    for parent in parents:
        result = merge_config(result, load_config(p.parent / parent, _parents=(*_parents, p)))
    return merge_config(result, data)
