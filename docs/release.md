# Releasing Saadi

Releases are built and published by GitHub Actions through PyPI's trusted
publishing, so no API token is stored anywhere. The version is set in one
place, `src/saadi/_version.py`.

## One-time setup

1. **PyPI.** Sign in at <https://pypi.org>, open *Account settings → Publishing*,
   and add a *pending publisher*: project name `saadi`, owner `ZeinabTaghavi`,
   repository `SAADI`, workflow `release.yml`, environment `pypi`.
2. **TestPyPI.** Do the same at <https://test.pypi.org> with environment `testpypi`.
3. **Zenodo.** Sign in at <https://zenodo.org> with GitHub, open
   *Settings → GitHub*, and switch on `ZeinabTaghavi/SAADI`. From then on every
   GitHub release gets a DOI; the first one also creates a concept DOI that always
   resolves to the latest version.

The `pypi` and `testpypi` environments are created on GitHub the first time the
workflow references them. Optionally add yourself as a required reviewer on
`pypi` under *Settings → Environments* to get a confirmation step before anything
reaches PyPI.

## Each release

1. Set the version in `src/saadi/_version.py`, move the entries from
   `[Unreleased]` in `CHANGELOG.md` into a dated `[X.Y.Z]` section, and update
   `version` and `date-released` in `CITATION.cff`. `python -m pytest -q
   tests/test_release_metadata.py` checks that the three agree.
2. Open a pull request, let CI pass (lint, tests on 3.11 and 3.12, and a build
   job that installs the wheel in a clean environment), and merge it.
3. **Rehearse on TestPyPI.** On the *Actions* tab run the *Release* workflow
   manually with target `testpypi`. It builds, publishes to TestPyPI, then
   installs `saadi==X.Y.Z` from TestPyPI into a fresh environment and runs
   `saadi demo --llm mock`. TestPyPI never accepts the same version twice, so a
   rehearsal that must be repeated needs a pre-release version such as
   `X.Y.Zrc1` on a branch.
4. **Check the GPU path.** Open
   [examples/release_check.ipynb](../examples/release_check.ipynb) on a GPU
   runtime (Colab T4 is enough), set `SAADI_INSTALL` to the TestPyPI line with
   the version under test, and run it top to bottom. It prints the resolved
   library versions, checks the files bundled in the wheel, loads a Qwen model
   through the vLLM backend, checks the prompt contract, and runs indexing,
   retrieval, answering, appending, and reloading with assertions. CI covers
   the same steps with the mock model only. When a change touches the
   backends, the prompts, or the retrievers, also run
   [examples/model_family_check.ipynb](../examples/model_family_check.ipynb):
   several light instruct models from different families index the same
   document in subprocesses, and every saved state is queried with each
   retriever; it reports extraction and fold call counts per model and a
   hit table per retriever.
5. **Publish.** Tag the merged commit and push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

   The workflow checks that the tag matches the package version, publishes to
   PyPI, and creates the GitHub release with the changelog section as notes.
   Zenodo archives that release and mints the DOI.
6. After the first release, add the Zenodo concept DOI to `CITATION.cff` under
   `identifiers` and the DOI badge to the README.

## When the paper is on arXiv

Fill in the commented `identifiers` and `url` lines under `preferred-citation`
in `CITATION.cff`, replace `notes: "Preprint"` with the arXiv entry, and add the
`eprint` field to the BibTeX block in the README; the metadata test fails until
both carry the same identifier.
