#!/usr/bin/env python3
"""00_check: spec hash, probe export, vLLM server and served window, scorer self-test, third-party imports.

Only a changed spec stops the run (exit code 3): every number would be meaningless. Anything
else (server not up, a third-party import failing) is reported here and makes the stage that
needs it fail on its own, while the independent stages still run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk_map as CM  # noqa: E402
import load_probe as LP  # noqa: E402
import report_utils as RPT  # noqa: E402
import scorer as SC  # noqa: E402

URL = os.environ.get("PBL_VLLM_URL", "http://127.0.0.1:8047/v1").rstrip("/")
MODEL = os.environ.get("PBL_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TP_DIR = Path(os.environ.get("PBL_TP_DIR", HERE.parent / "third_party"))
HIPPORAG_PY = os.environ.get("PBL_HIPPORAG_PY", str(TP_DIR / "venv_hipporag" / "bin" / "python"))
LQCA_PY = os.environ.get("PBL_LQCA_PY", str(TP_DIR / "venv_lqca" / "bin" / "python"))
LQCA_DIR = Path(os.environ.get("PBL_LQCA_DIR", TP_DIR / "LQCA"))
LQCA_COMMIT = os.environ.get("PBL_LQCA_COMMIT", "2d05899479d5ca08635cb544711dde32eea57b79")


def run(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()[-2000:]
    except Exception as exc:
        return 127, str(exc)


def main() -> int:
    rows, unexpected, numbers = [], [], {}
    # 1. The frozen spec (fatal) and the export every venv reads.
    try:
        spec = LP.check_spec()
    except Exception as exc:
        print(f"SPEC CHECK FAILED: {exc}", flush=True)
        CM.write_numbers("check", "check", {"spec": str(exc), "unexpected": [f"The frozen spec check failed: {exc}. Nothing else ran."]})
        (CM.system_dir("check") / "report_check.md").write_text(f"### Spec\n\n**FAILED**: {exc}\n", encoding="utf-8")
        return 3
    if spec["problems"]:
        unexpected += [f"spec: {p}" for p in spec["problems"]]
    manifest = LP.export_inputs(CM.INPUTS_DIR)
    rows.append(["frozen spec (personas.jsonl SHA-256)", f"`{spec['spec_sha256']}`", "ok"])
    rows.append(["personas / cells / books", f"{spec['personas']} / {spec['cells']} / {len(spec['books'])}", "ok"])
    rows.append(
        [
            "chunks per book equal SAADI's builds",
            ", ".join(f"{b} {v['chunks']}" for b, v in spec["books"].items()),
            "ok" if not spec["problems"] else "PROBLEM",
        ]
    )
    rows.append(["probe exported for the third-party venvs", str(manifest.parent), "ok"])
    numbers["spec"] = spec

    # 2. The vLLM server: reachable, served model, window, one call.
    try:
        import urllib.request

        with urllib.request.urlopen(URL + "/models", timeout=60) as resp:
            models = json.loads(resp.read().decode("utf-8"))["data"]
        served = next((m for m in models if m.get("id") == MODEL), models[0])
        window = int(served.get("max_model_len", 0))
        version = "?"
        try:
            with urllib.request.urlopen(URL[: -len("/v1")] + "/version", timeout=30) as resp:
                version = json.loads(resp.read().decode("utf-8")).get("version", "?")
        except Exception:
            pass
        req = urllib.request.Request(
            URL + "/chat/completions",
            data=json.dumps(
                {"model": MODEL, "messages": [{"role": "user", "content": "Reply with the word ready."}], "temperature": 0, "max_tokens": 5}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            reply = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        rows.append(["vLLM server", f"{URL}, model {served.get('id')}, vLLM {version}", "ok"])
        rows.append(["served window (max_model_len)", window, "ok" if window >= 200_000 else "SMALL: B2 sees only a prefix of most books"])
        rows.append(["test call", repr(reply[:40]), "ok"])
        numbers["vllm"] = {"url": URL, "model": served.get("id"), "window": window, "version": version}
    except Exception as exc:
        rows.append(["vLLM server", f"{URL} not reachable: {exc}", "FAILED: B2 and B3 will fail"])
        unexpected.append(f"vLLM server not reachable at {URL}: {exc}. See logs/vllm_server_*.log.")

    # 3. The scorer self-test on SAADI run 1's per-persona rankings.
    run1 = SC.SAADI_RUNS["run1"]
    if run1["per_persona"].exists():
        ok, lines = SC.self_test(run1["per_persona"], run1["table"], run1["label"])
        rows.append(
            [
                "scorer self-test: run 1's per_persona.jsonl against results/method_update/p1_ledger/table.csv",
                f"{len(lines)} cell-method rows",
                "PASSED" if ok else "FAILED",
            ]
        )
        numbers["self_test"] = {"passed": ok, "lines": lines}
        if not ok:
            unexpected.append("The scorer does not reproduce run 1's table: " + "; ".join(x for x in lines if x.startswith("DIFF"))[:1500])
    else:
        rows.append(["scorer self-test", f"{run1['per_persona']} not found", "SKIPPED"])
        unexpected.append("Scorer self-test skipped: run 1's per_persona.jsonl is not on this machine.")

    # 4. Third-party code.
    code, out = run(
        [
            HIPPORAG_PY,
            "-c",
            "import hipporag, torch, transformers, sys; print('hipporag', getattr(hipporag,'__version__','2.0.0a4'), 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), 'python', sys.version.split()[0])",
        ]
    )
    rows.append(["B3 hipporag venv imports", out.splitlines()[-1] if out else "", "ok" if code == 0 else "FAILED: B3 will fail"])
    if code != 0:
        unexpected.append(f"hipporag import failed ({HIPPORAG_PY}):\n```\n{out}\n```")
    code, out = run(
        [HIPPORAG_PY, "-c", "from huggingface_hub import snapshot_download; p = snapshot_download('nvidia/NV-Embed-v2'); print(p)"],
        timeout=3600,
    )
    rows.append(["NV-Embed-v2 weights in the HF cache", out.splitlines()[-1] if out else "", "ok" if code == 0 else "FAILED: B3 will fail"])
    if code != 0:
        unexpected.append(f"NV-Embed-v2 download failed:\n```\n{out}\n```")
    commit = ""
    if (LQCA_DIR / ".git").exists():
        _c, commit = run(["git", "-C", str(LQCA_DIR), "rev-parse", "HEAD"])
    rows.append(
        [
            "LQCA clone at the pinned commit",
            f"{LQCA_DIR} @ {commit[:12] or 'missing'}",
            "ok" if commit.strip() == LQCA_COMMIT else "FAILED: B4 will fail",
        ]
    )
    code, out = run(
        [
            LQCA_PY,
            "-c",
            "import maverick, tiktoken, torch, spacy, sys; from maverick import Maverick; spacy.load('en_core_web_sm'); tiktoken.get_encoding('cl100k_base'); m = Maverick(hf_name_or_path='sapienzanlp/maverick-mes-ontonotes', device='cpu'); print('maverick ok; torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'python', sys.version.split()[0])",
        ],
        timeout=3600,
    )
    rows.append(
        [
            "B4 venv: maverick-coref, Maverick checkpoint, spaCy, tiktoken",
            out.splitlines()[-1] if out else "",
            "ok" if code == 0 else "FAILED: B4 will fail",
        ]
    )
    if code != 0:
        unexpected.append(f"LQCA venv check failed ({LQCA_PY}):\n```\n{out}\n```")
    numbers["third_party"] = {"lqca_commit": commit.strip(), "hipporag_py": HIPPORAG_PY, "lqca_py": LQCA_PY}

    CM.write_numbers("check", "check", {**numbers, "unexpected": unexpected})
    (CM.system_dir("check") / "report_check.md").write_text(
        "### Checks\n\n" + RPT.md_table(["check", "value", "result"], rows) + "\n", encoding="utf-8"
    )
    (CM.system_dir("check") / "files_check.txt").write_text(str(CM.INPUTS_DIR) + "/<book>.json, manifest.json\n", encoding="utf-8")
    for r in rows:
        print(" | ".join(str(x) for x in r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
