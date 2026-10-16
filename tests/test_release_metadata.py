"""Release metadata stays consistent: the version, the citation, the changelog, the extras, and the ruff scope."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

import saadi

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _readme_bibtex() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("```bibtex") + len("```bibtex")
    return readme[start : readme.index("```", start)]


def test_the_version_is_defined_once() -> None:
    project = _pyproject()
    assert "version" not in project["project"] and "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "saadi._version.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", saadi.__version__)


def test_the_citation_file_matches_the_package_and_the_readme() -> None:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["version"] == saadi.__version__
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(cff["date-released"]))
    paper = cff["preferred-citation"]
    bibtex = _readme_bibtex()
    title = re.search(r"title\s*=\s*\{(.+)\}", bibtex).group(1).replace("{", "").replace("}", "")
    assert paper["title"] == title
    assert str(paper["year"]) == re.search(r"year\s*=\s*\{(\d{4})\}", bibtex).group(1)
    bib_authors = [name.strip() for name in re.search(r"author\s*=\s*\{(.+)\}", bibtex).group(1).split(" and ")]
    assert [f"{a['family-names']}, {a['given-names']}" for a in paper["authors"]] == bib_authors
    # Once the paper is on arXiv, both the README and the citation file must name it.
    eprint = re.search(r"eprint\s*=\s*\{([^}]+)\}", bibtex)
    arxiv_ids = [str(item.get("value", "")) for item in paper.get("identifiers", []) if "arxiv" in str(item.get("value", "")).lower()]
    assert bool(eprint) == bool(arxiv_ids)
    if eprint:
        assert any(eprint.group(1) in value for value in arxiv_ids)


def test_the_changelog_has_a_section_for_the_current_version() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{saadi.__version__}]" in text
    assert f"[{saadi.__version__}]: https://github.com/ZeinabTaghavi/SAADI/releases/tag/v{saadi.__version__}" in text


def test_extras_and_package_data() -> None:
    project = _pyproject()
    extras = project["project"]["optional-dependencies"]
    assert set(extras) == {"openai", "anthropic", "dense", "vllm", "repro", "dev"}
    assert any(req.startswith("ruff") for req in extras["dev"])
    assert any(req.startswith("saadi[") for req in extras["repro"])  # repro builds on dense and vllm
    package_data = project["tool"]["setuptools"]["package-data"]
    assert "profiles/*/*.txt" in package_data["saadi.prompts"]
    assert "data/*.txt" in package_data["saadi"]
    assert project["project"]["scripts"]["saadi"] == "saadi.cli.main:main"


def test_ruff_never_touches_the_paper_code() -> None:
    pytest.importorskip("ruff")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--show-files", "."], cwd=ROOT, capture_output=True, text=True, check=True
    )
    selected = {Path(line.strip()).resolve().relative_to(ROOT).as_posix() for line in result.stdout.splitlines() if line.strip()}
    preserved = {row["path"] for row in json.loads((ROOT / "docs/preserved_sources.json").read_text(encoding="utf-8"))}
    assert not (selected & preserved)
    assert {"src/saadi/document.py", "src/saadi/llm/base.py", "tests/test_document_api.py"} <= selected


def test_pre_commit_and_ci_pin_the_same_ruff() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    rev = next(repo["rev"] for repo in config["repos"] if "ruff-pre-commit" in repo["repo"])
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert f"ruff=={rev.lstrip('v')}" in ci


def test_the_release_workflow_uses_trusted_publishing() -> None:
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    triggers = release.get("on", release.get(True))  # YAML 1.1 reads a bare `on` as a boolean key
    assert "v*" in triggers["push"]["tags"]
    jobs = release["jobs"]
    for job_name, environment in (("publish-pypi", "pypi"), ("publish-testpypi", "testpypi")):
        job = jobs[job_name]
        assert job["environment"]["name"] == environment
        assert job["permissions"] == {"id-token": "write"}
        publish = [step for step in job["steps"] if "pypa/gh-action-pypi-publish" in step.get("uses", "")]
        assert len(publish) == 1
        assert "password" not in json.dumps(publish[0].get("with", {}))  # trusted publishing, no token
    assert jobs["publish-pypi"]["if"] == "github.ref_type == 'tag'"
    assert "verify-testpypi" in jobs and "github-release" in jobs
