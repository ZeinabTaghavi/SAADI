# Contributing

Thank you for considering a contribution. Questions go to [Discussions](https://github.com/ZeinabTaghavi/SAADI/discussions); bugs and feature requests to the issue templates.

## Setup

```bash
git clone https://github.com/ZeinabTaghavi/SAADI.git
cd SAADI
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
python -m pytest -q
```

The tests run offline: hosted providers are exercised against a fake HTTP server in `tests/_fake_api_server.py`, so no API key is needed. `pre-commit` runs ruff on the files you commit.

## Two kinds of code

- **The library** (`src/saadi/document.py`, `llm/`, `retrievers/`, `prompts/`, `inputs/`, `exports.py`, the `index`, `query`, `demo`, and `check-prompts` commands) is where new providers, retrievers, prompt profiles, and input formats go. It is formatted and linted by ruff.
- **The paper's code** is kept byte for byte so the saved results stay reproducible. The files are listed in `docs/preserved_sources.json` and checked by a test; ruff is configured to skip them. If a change there is unavoidable, remove the file from the list and add a sentence to the "Code provenance" section of `docs/results.md`.

## Pull requests

One topic per pull request, with a test for the behavior it adds or fixes. Describe what a user can now do differently. CI runs lint, the tests on Python 3.11 and 3.12, and a build job that installs the wheel in a clean environment.

Good places to start are the issues labelled [good first issue](https://github.com/ZeinabTaghavi/SAADI/labels/good%20first%20issue). Adding a provider is usually a small backend class next to the existing ones in `src/saadi/llm/`; adding an input format is an adapter in `src/saadi/inputs/`; a prompt profile for a new domain is a folder of three text files, validated with `saadi check-prompts`.
