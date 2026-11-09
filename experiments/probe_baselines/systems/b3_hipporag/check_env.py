#!/usr/bin/env python3
"""B3's venv check, run once by run_all.sh after the install (in third_party/venv_hipporag).

The versions are SAADI_Hipporag's tested set (its scripts/common.sh check_runtime_versions);
the imports are the ones B3 needs: hipporag, huggingface_hub (NV-Embed-v2 download) and the
OpenIE adapter from the SAADI_Hipporag clone (PYTHONPATH=<clone>/src).
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

EXPECTED = {
    "hipporag": "2.0.0a4",
    "litellm": "1.55.8",
    "openai": "1.58.1",
    "torch": "2.5.1",
    "transformers": "4.45.2",
    "vllm": "0.6.6.post1",
}


def main() -> int:
    bad = []
    for name, want in EXPECTED.items():
        try:
            got = version(name)
        except PackageNotFoundError:
            got = None
        print(f"{name} {got} (expected {want})")
        if got != want:
            bad.append(f"{name}: {got} instead of {want}")
    import hipporag  # noqa: F401
    import huggingface_hub  # noqa: F401
    from saadi_hypporag.openie_compat import RobustOpenIE, parse_triples  # noqa: F401

    print("imports ok: hipporag, huggingface_hub, saadi_hypporag.openie_compat")
    if bad:
        print("version mismatch: " + "; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
