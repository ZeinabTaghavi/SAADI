"""C8 step 00 (CPU only): check C5's outputs, build the full documents, count tokens, choose the
vLLM settings. No model is called.

Order (it stops at the first failed gate and still writes its report):
  1. Fixed inputs: published SAADI point (configs/methods/sadi.yaml) and docs/preserved_sources.json
     unchanged; C5's contexts for SAADI, Vanilla and closed-book exist, are complete and match
     C5's manifest (sha256); their query lists equal SAADI's saved query list.
  2. Long-context model: tokenizer, config and weights in the Hugging Face cache.
  3. Documents: the whole document of every query from SAADI's chunks in document order (C5's
     chunk pool, the text SAADI indexes); tokens per document; the position of the gold evidence
     (first / middle / last third). Distinct documents and queries per document.
  4. Prompt tokens of every condition with the long-context model's tokenizer and chat template.
  5. vLLM settings: context window = longest prompt + output tokens + margin; memory fit on 4 GPUs;
     documents beyond the model's native window and whether the installed vLLM has the dual chunk
     attention backend they need. A rough time estimate (the GPU script measures the real one).
Writes outputs/rebuttal/c8_long_context/prepared/ (git-ignored), results/rebuttal/c8_long_context/
prepare/*.csv and reports/step_00.{md,json,summary.txt}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import c8_common as c8

cc = c8.cc
s0 = c8.load_c5_step00()  # C5's step 00, read only: Dataset, Tok, checks

STEP = "00"
GIB = 1024**3


class GateFailed(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[c8-00] {cc.now()} {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# 1. C5 inputs
# ---------------------------------------------------------------------------------------------


def check_c5(paths: c8.C8Paths, key: str, qids_saved: list[str], info: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    man_path = paths.c5_manifest(key)
    if not man_path.exists():
        raise GateFailed(f"{key}: C5's contexts manifest {man_path} does not exist; run C5's step 00 first")
    man = cc.read_json(man_path)
    rows: dict[str, list[dict[str, Any]]] = {}
    checks = []
    for cond in c8.C5_CONDITIONS:
        meta = (man.get("conditions") or {}).get(cond) or {}
        path = paths.c5_context_file(key, cond)
        if not meta.get("available") or not path.exists():
            raise GateFailed(f"{key}: C5 context file for {cond} is missing ({path})")
        sha = cc.sha256_file(path)
        if sha != meta.get("sha256"):
            raise GateFailed(f"{key}/{cond}: {path} differs from C5's manifest (sha256 {sha[:12]} vs {str(meta.get('sha256'))[:12]})")
        rs = cc.read_jsonl(path)
        ids = [str(r["query_id"]) for r in rs]
        if len(set(ids)) != len(ids):
            raise GateFailed(f"{key}/{cond}: duplicate query ids in C5's contexts")
        if ids != qids_saved:
            missing = sorted(set(qids_saved) - set(ids))[:5]
            extra = sorted(set(ids) - set(qids_saved))[:5]
            if set(ids) != set(qids_saved):
                raise GateFailed(f"{key}/{cond}: C5's queries differ from SAADI's saved list (missing {missing}, extra {extra})")
        rows[cond] = rs
        checks.append({"condition": cond, "file": paths.rel(path), "rows": len(rs), "sha256": sha, "complete": bool(meta.get("complete"))})
    # Same question and gold answers in the three files.
    base = {r["query_id"]: r for r in rows["saadi"]}
    for cond in ("vanilla", "closed_book"):
        for r in rows[cond]:
            b = base[r["query_id"]]
            if r["question"] != b["question"] or r["gold_answers"] != b["gold_answers"] or r["doc_id"] != b["doc_id"]:
                raise GateFailed(f"{key}: query {r['query_id']} has another question, gold answer or document in C5's {cond} file than in its saadi file")
    info["c5_inputs"] = {"manifest": paths.rel(man_path), "manifest_created": man.get("created"), "c5_commit": man.get("commit"),
                         "c5_reader_model": man.get("reader_model"), "template_sha256": man.get("template_sha256"), "files": checks}
    import hashlib

    if man.get("template_sha256") and man["template_sha256"] != hashlib.sha256(cc.READER_TEMPLATE.encode()).hexdigest():
        raise GateFailed(f"{key}: C5's contexts were built with another prompt template than c5_common.READER_TEMPLATE")
    return rows


def load_para_labels(paths: c8.C8Paths, key: str, info: dict[str, Any]) -> dict[str, list[str]] | None:
    """NovelHopQA: C5's paragraph labels (each evidence paragraph located in SAADI's chunks)."""

    if key != "novelhopqa":
        return None
    p = paths.c5_para_labels(key)
    if not p.exists():
        info.setdefault("warnings", []).append(f"{key}: C5's paragraph labels {paths.rel(p)} are missing; evidence position uses the paper's window labels (unreliable on NovelHopQA)")
        return None
    out = {}
    for r in cc.read_jsonl(p):
        out[str(r["query_id"])] = list(dict.fromkeys(c for g in (r.get("paragraph_chunk_ids") or []) for c in g))
    info["evidence_labels_file"] = paths.rel(p)
    return out


# ---------------------------------------------------------------------------------------------
# 2. Model files
# ---------------------------------------------------------------------------------------------


def model_files(model: str) -> dict[str, Any]:
    """Config and weight sizes from the local Hugging Face cache (offline)."""

    rec: dict[str, Any] = {"model": model, "in_cache": False}
    try:
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(model, local_files_only=True))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return rec
    rec["snapshot"] = str(d)
    cfg_p = d / "config.json"
    if cfg_p.exists():
        rec["config"] = json.loads(cfg_p.read_text(encoding="utf-8"))
    idx = d / "model.safetensors.index.json"
    files = sorted(d.glob("*.safetensors"))
    if idx.exists():
        wanted = sorted(set(json.loads(idx.read_text(encoding="utf-8"))["weight_map"].values()))
        missing = [w for w in wanted if not (d / w).exists()]
        rec["weight_files"] = len(wanted)
        rec["weight_files_missing"] = missing
        files = [d / w for w in wanted if (d / w).exists()]
    else:
        rec["weight_files"] = len(files)
        rec["weight_files_missing"] = [] if files else ["*.safetensors"]
    rec["weights_bytes"] = sum(f.stat().st_size for f in files)
    rec["in_cache"] = bool(files) and not rec["weight_files_missing"] and "config" in rec
    return rec


def model_shape(rec: dict[str, Any]) -> dict[str, Any]:
    cfg = rec.get("config") or {}
    assumed = not cfg
    # Qwen2.5-7B-Instruct-1M (config.json), used only when the config is not in the cache.
    layers = int(cfg.get("num_hidden_layers", 28))
    heads = int(cfg.get("num_attention_heads", 28))
    kv_heads = int(cfg.get("num_key_value_heads", 4))
    hidden = int(cfg.get("hidden_size", 3584))
    inter = int(cfg.get("intermediate_size", 18944))
    head_dim = int(cfg.get("head_dim", hidden // heads))
    max_pos = int(cfg.get("max_position_embeddings", 1010000))
    dca = cfg.get("dual_chunk_attention_config") or {}
    native = int(dca.get("original_max_position_embeddings", 262144))
    weights = int(rec.get("weights_bytes") or 0) or int(2 * 7.62e9)
    return {"assumed_from_model_card": assumed, "layers": layers, "heads": heads, "kv_heads": kv_heads, "hidden": hidden,
            "intermediate": inter, "head_dim": head_dim, "max_position_embeddings": max_pos, "native_window": native,
            "dual_chunk_attention_config": dca, "sparse_attention_config": cfg.get("sparse_attention_config"),
            "weights_bytes": weights, "params": weights / 2, "dtype": cfg.get("torch_dtype", "bfloat16"),
            "kv_bytes_per_token": 2 * layers * kv_heads * head_dim * 2}


def vllm_check(python: str) -> dict[str, Any]:
    """Installed vLLM of the server's Python, without importing it (no CUDA): version, V0 engine,
    dual chunk attention backend."""

    code = r"""
import json, os, importlib.util as u, importlib.metadata as m
r = {}
try:
    r["version"] = m.version("vllm")
except Exception as e:
    r["version"] = None; r["error"] = repr(e)[:200]
s = u.find_spec("vllm")
if s and s.submodule_search_locations:
    d = list(s.submodule_search_locations)[0]
    r["path"] = d
    hits = []
    for dp, dn, fn in os.walk(d):
        for f in fn:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                try:
                    with open(p, encoding="utf-8", errors="ignore") as h:
                        if "DUAL_CHUNK_FLASH_ATTN" in h.read():
                            hits.append(os.path.relpath(p, d))
                except Exception:
                    pass
    r["dca_mentions"] = sorted(hits)[:20]
    r["dca_backend_file"] = os.path.exists(os.path.join(d, "attention", "backends", "dual_chunk_flash_attn.py"))
    r["v0_engine"] = os.path.exists(os.path.join(d, "engine", "llm_engine.py")) and os.path.exists(os.path.join(d, "worker", "model_runner.py"))
try:
    r["flash_attn_ext"] = u.find_spec("vllm.vllm_flash_attn") is not None
except Exception:
    r["flash_attn_ext"] = None
print(json.dumps(r))
"""
    try:
        out = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=300,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if out.returncode != 0:
            return {"python": python, "error": out.stderr.strip()[-400:]}
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        rec["python"] = python
        rec["dca_available"] = bool(rec.get("dca_backend_file")) and bool(rec.get("v0_engine"))
        return rec
    except Exception as exc:
        return {"python": python, "error": f"{type(exc).__name__}: {exc}"}


def gpu_memory(gpus: str) -> dict[str, Any]:
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpus, "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        rows = []
        for line in out.stdout.strip().splitlines():
            i, name, tot, used = [x.strip() for x in line.split(",")]
            rows.append({"index": int(i), "name": name, "total_mib": int(tot), "used_mib": int(used)})
        if rows:
            return {"source": "nvidia-smi", "gpus": rows}
    except Exception:
        pass
    mib = int(os.environ.get("C8_GPU_MEM_MIB", 49140))
    return {"source": f"C8_GPU_MEM_MIB={mib} (nvidia-smi not readable)", "gpus": [{"index": int(g), "name": "assumed RTX A6000", "total_mib": mib, "used_mib": None} for g in gpus.split(",")]}


# ---------------------------------------------------------------------------------------------
# 3. Documents and evidence position
# ---------------------------------------------------------------------------------------------


def build_documents(ds: Any, tok: Any) -> dict[str, dict[str, Any]]:
    by_doc: dict[str, list[tuple[int, str, str]]] = {}
    for cid, c in ds.chunks.items():
        by_doc.setdefault(c["doc"], []).append((int(c["index"]), cid, c["text"]))
    docs = {}
    for d in sorted(by_doc):
        cs = sorted(by_doc[d])
        idx = [i for i, _, _ in cs]
        if len(set(idx)) != len(idx):
            raise GateFailed(f"{ds.key}/{d}: two chunks share a position in the document")
        text = c8.document_passage_text([t for _, _, t in cs])
        n_chunk = [tok.count(" ".join(str(t).split())) for _, _, t in cs]
        starts, acc = [], 0
        for n in n_chunk:
            starts.append(acc)
            acc += n
        docs[d] = {"doc_id": d, "text": text, "n_tokens": tok.count(c8.format_context([{"text": text}])), "n_chunks": len(cs),
                   "chunk_ids": [cid for _, cid, _ in cs], "chunk_token_start": starts, "chunk_tokens": n_chunk,
                   "chunk_tokens_sum": acc, "contiguous": idx == list(range(idx[0], idx[0] + len(idx)))}
    return docs


def evidence_position(doc: dict[str, Any], chunk_ids: list[str]) -> dict[str, Any]:
    """Relative position (0-1, in tokens) of every evidence chunk's midpoint; the query's position
    is their mean; 'third' from it; 'spread' when the chunks fall in more than one third."""

    pos_of = {c: i for i, c in enumerate(doc["chunk_ids"])}
    total = max(1, doc["chunk_tokens_sum"])
    pts = []
    for c in chunk_ids:
        i = pos_of.get(c)
        if i is None:
            continue
        pts.append((doc["chunk_token_start"][i] + doc["chunk_tokens"][i] / 2.0) / total)
    if not pts:
        return {"n_chunks": 0, "pos_mean": None, "pos_first": None, "pos_last": None, "third": None, "spread": None}
    third = lambda p: c8.THIRDS[min(2, int(p * 3))]  # noqa: E731
    m = mean(pts)
    return {"n_chunks": len(pts), "pos_mean": round(m, 4), "pos_first": round(min(pts), 4), "pos_last": round(max(pts), 4),
            "third": third(m), "spread": len({third(p) for p in pts}) > 1}


# ---------------------------------------------------------------------------------------------
# 4. Prompt tokens of the full-document condition
# ---------------------------------------------------------------------------------------------


class FullDocCounter:
    """Prompt tokens of 'document then question' without re-tokenising the document per query.

    The rendered prompt is cut right after the blank line that follows the document, before
    'Question:'. Qwen's pre-tokeniser never joins a letter run with the newlines before it, so no
    token crosses the cut and the count is exact; it is checked against a full tokenisation of the
    first query of every document (and the chat template's own tokenisation)."""

    MARK = "\n\nQuestion: "

    def __init__(self, tok: Any):
        self.tok = tok.tok
        self.prefix_n: dict[str, int] = {}
        self.checked: list[dict[str, Any]] = []

    def _n(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def count(self, doc: dict[str, Any], question: str, verify: bool = False) -> int:
        msgs = c8.full_doc_messages(question, doc["text"])
        rendered = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        passage = c8.format_context([{"text": doc["text"]}])
        i = rendered.index(passage) + len(passage)
        if rendered[i:i + len(self.MARK)] != self.MARK:
            raise GateFailed("the prompt template no longer puts '\\n\\nQuestion: ' right after the context")
        cut = i + 2
        if doc["doc_id"] not in self.prefix_n:
            self.prefix_n[doc["doc_id"]] = self._n(rendered[:cut])
        n = self.prefix_n[doc["doc_id"]] + self._n(rendered[cut:])
        if verify:
            full = self._n(rendered)
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            self.checked.append({"doc_id": doc["doc_id"], "split_count": n, "full_count": full, "chat_template_count": len(ids)})
            if full != n or len(ids) != n:
                raise GateFailed(f"{doc['doc_id']}: split token count {n} != full tokenisation {full} / chat template {len(ids)}")
        return n


# ---------------------------------------------------------------------------------------------
# 5. Settings, memory, time
# ---------------------------------------------------------------------------------------------


def choose_max_model_len(need: int, shape: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    margin = max(int(env.get("C8_CONTEXT_MARGIN_TOKENS", 8192)), math.ceil(need * float(env.get("C8_CONTEXT_MARGIN_PCT", 2)) / 100))
    chosen = int(math.ceil((need + margin) / 8192) * 8192)
    cap = shape["max_position_embeddings"]
    capped = chosen > cap
    override = (env.get("C8_MAX_MODEL_LEN") or "").strip()
    final = int(override) if override else min(chosen, cap)
    return {"need": need, "margin": margin, "rounded": chosen, "model_max": cap, "capped_at_model_max": capped,
            "override": int(override) if override else None, "max_model_len": final}


def memory_fit(shape: dict[str, Any], gpus: dict[str, Any], max_model_len: int, env: dict[str, str]) -> dict[str, Any]:
    tp = int(env.get("C8_TENSOR_PARALLEL", 4))
    util = float(env.get("C8_GPU_MEMORY_UTILIZATION", 0.90))
    batched = int(env.get("C8_MAX_NUM_BATCHED_TOKENS", 131072))
    per_gpu = min(g["total_mib"] for g in gpus["gpus"]) * 1024**2
    weights = shape["weights_bytes"] / tp
    # Activations of one chunked-prefill step (hidden states, attention and MLP buffers; generous),
    # plus CUDA context, NCCL buffers and the dual chunk attention workspace.
    act = batched * (shape["hidden"] * 2 * 8 + shape["intermediate"] * 2 * 3 / tp) + 2.0 * GIB
    kv_tok_gpu = shape["kv_bytes_per_token"] / tp
    kv_room = per_gpu * util - weights - act
    capacity = int(kv_room / kv_tok_gpu) if kv_room > 0 else 0
    return {"tensor_parallel": tp, "gpu_memory_utilization": util, "gpu_mib": per_gpu // 1024**2,
            "weights_gib_per_gpu": weights / GIB, "activations_gib_per_gpu": act / GIB,
            "kv_kib_per_token_per_gpu": kv_tok_gpu / 1024, "kv_room_gib_per_gpu": kv_room / GIB,
            "kv_capacity_tokens": capacity, "max_model_len": max_model_len,
            "longest_sequence_kv_gib_per_gpu": max_model_len * kv_tok_gpu / GIB,
            "fits": capacity >= max_model_len, "headroom_sequences": capacity / max_model_len if max_model_len else None,
            "single_gpu_kv_capacity_tokens": int(max(0.0, per_gpu * util - shape["weights_bytes"] - act) / shape["kv_bytes_per_token"])}


# Rough time model (a priori; the GPU script's 20-query timing replaces it). 4 x RTX A6000,
# tensor parallel 4, bf16, eager mode: effective dense throughput and memory bandwidth.
EFFECTIVE_FLOPS = 1.2e14
EFFECTIVE_BW = 4 * 768e9 * 0.55
EXPECTED_OUTPUT_TOKENS = 24


def cold_seconds(n: int, shape: dict[str, Any]) -> float:
    linear = 2.0 * shape["params"] * n
    attn = 2.0 * n * n * shape["layers"] * shape["heads"] * shape["head_dim"]
    return (linear + attn) / EFFECTIVE_FLOPS


def decode_seconds(n_ctx: int, shape: dict[str, Any], n_out: int = EXPECTED_OUTPUT_TOKENS) -> float:
    return n_out * (shape["weights_bytes"] + n_ctx * shape["kv_bytes_per_token"]) / EFFECTIVE_BW


def rough_hours(key: str, docs: dict[str, dict[str, Any]], rows: list[dict[str, Any]], shape: dict[str, Any], conc: int) -> dict[str, Any]:
    per_doc = Counter(r["doc_id"] for r in rows)
    cold = sum(cold_seconds(docs[d]["n_tokens"], shape) for d in per_doc)
    warm = sum(decode_seconds(docs[r["doc_id"]]["n_tokens"], shape) + cold_seconds(64, shape) for r in rows)
    nocache = sum(cold_seconds(docs[r["doc_id"]]["n_tokens"], shape) + decode_seconds(docs[r["doc_id"]]["n_tokens"], shape) for r in rows)
    short = 0.0
    for cond in c8.C5_CONDITIONS:
        short += sum(cold_seconds(r["prompt_tokens"][cond], shape) + decode_seconds(r["prompt_tokens"][cond], shape) for r in rows) / max(1, conc)
    wall_cache = (cold + warm + short) / 3600 + 0.25
    return {"full_doc_cold_prefill_h": cold / 3600, "full_doc_warm_h": warm / 3600, "short_conditions_h": short / 3600,
            "wall_h_with_cache": wall_cache, "gpu_h_with_cache": 4 * wall_cache,
            "full_doc_wall_h_without_cache": nocache / 3600, "full_doc_gpu_h_without_cache": 4 * nocache / 3600}


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_report(paths: c8.C8Paths, info: dict[str, Any], status: str) -> None:
    f = c8.fmt
    rep = paths.reports / "step_00.md"
    L = [f"# C8 long-context reader - step 00 (prepare): {status}", "",
         f"Written {cc.now()}; commit {info.get('commit')}; host {info.get('host')}; {info.get('seconds', 0)} s. CPU only, no model was called.", "",
         "Command:", "", "```bash", "bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh " + " ".join(info.get("argv", [])), "```", ""]
    if info.get("summary"):
        L += ["## Summary", ""] + [f"- {s}" for s in info["summary"]] + [""]
    if info.get("warnings"):
        L += ["## Warnings", ""] + [f"- {w}" for w in info["warnings"]] + [""]
    # Inputs
    if info.get("per_dataset"):
        L += ["## Inputs from C5 (read only)", ""]
        rows = []
        for key, d in info["per_dataset"].items():
            for c in (d.get("c5_inputs") or {}).get("files", []):
                rows.append([c8.DATASET_DISPLAY[key], c["condition"], c["rows"], c["complete"], c["sha256"][:12], c["file"]])
        L += cc.md_table(["Dataset", "Condition", "Rows", "Complete", "sha256", "File"], rows) + [""]
        pres = info.get("preserved") or {}
        L += [f"Published SAADI point {info.get('published_point')} (configs/methods/sadi.yaml); docs/preserved_sources.json: "
              f"{pres.get('checked')} files, changed {len(pres.get('changed', []))}, missing {len(pres.get('missing', []))}.", ""]
    # Model
    m = info.get("model") or {}
    sh = info.get("shape") or {}
    if m:
        L += ["## Long-context model", "",
              f"`{m.get('model')}`: in the Hugging Face cache: {m.get('in_cache')}" + (f" ({m.get('snapshot')})" if m.get("snapshot") else "")
              + (f"; error: {m.get('error')}" if m.get("error") else "") + ".",
              f"Tokenizer used for every count: `{info.get('tokenizer')}`.", ""]
        if sh:
            L += [f"Shape{' (assumed from the model card: config not in the cache)' if sh.get('assumed_from_model_card') else ''}: {sh['layers']} layers, "
                  f"{sh['heads']} query heads, {sh['kv_heads']} KV heads, head dim {sh['head_dim']}, weights {sh['weights_bytes'] / 1e9:.1f} GB ({sh['dtype']}); "
                  f"KV cache {sh['kv_bytes_per_token'] / 1024:.0f} KiB per token; max position {sh['max_position_embeddings']:,}; "
                  f"native window {sh['native_window']:,} tokens (longer inputs need dual chunk attention: {json.dumps(sh.get('dual_chunk_attention_config'))}).", ""]
    # Documents
    if info.get("doc_stats"):
        L += ["## Document lengths (tokens of the long-context model's tokenizer)", ""]
        L += cc.md_table(["Dataset", "Queries", "Distinct documents", "Queries per document (min / median / max)", "Tokens min", "Tokens median", "Tokens mean", "Tokens max", "Chunks per document (mean)"],
                         [[c8.DATASET_DISPLAY[k], s["queries"], s["documents"], f"{s['q_min']} / {f(s['q_median'], 1)} / {s['q_max']}", f(s["min"]), f(s["median"], 0), f(s["mean"], 0), f(s["max"]), f(s["chunks_mean"], 0)]
                          for k, s in info["doc_stats"].items()]) + [""]
        L += ["Document = SAADI's chunks of that document in order, whitespace collapsed and joined by one space (the chunks are cut at whitespace), "
              "given to the reader as C5's single passage `[1] <document>`. Per-document lengths: `results/rebuttal/c8_long_context/prepare/doc_lengths.csv`.", ""]
        nov = [r for r in info.get("doc_rows", []) if r["dataset"] == "novelhopqa"]
        if nov:
            L += ["NovelHopQA books:", ""]
            L += cc.md_table(["Book", "Tokens", "Chunks", "Queries", "Prompt fits the window", "Beyond the native window"],
                             [[r["doc_id"], f(r["tokens"]), r["chunks"], r["queries"], r["fits"], r["over_native"]] for r in sorted(nov, key=lambda x: -x["tokens"])]) + [""]
    # Prompts
    if info.get("prompt_rows"):
        L += ["## Prompt tokens per condition (chat template included)", ""]
        L += cc.md_table(["Dataset", "Condition", "Queries", "Mean", "Median", "Max", "Total (M)"],
                         [[c8.DATASET_DISPLAY[r["dataset"]], c8.CONDITION_DISPLAY[r["condition"]], r["n"], f(r["mean"], 0), f(r["median"], 0), f(r["max"]), f(r["total"] / 1e6, 2)]
                          for r in info["prompt_rows"]]) + [""]
        chk = info.get("count_checks") or {}
        if chk:
            L += [f"Full-document counts: the split count equals the full tokenisation and the chat template's count on the first query of all {chk['n']} documents "
                  f"(max difference {chk['max_diff']}).", ""]
    # Settings
    s = info.get("settings") or {}
    if s:
        mm, mem, v = s["max_model_len"], s["memory"], s["vllm"]
        L += ["## vLLM settings", ""]
        L += cc.md_table(["Setting", "Value", "Why"], [
            ["--max-model-len", f(mm["max_model_len"]), f"longest prompt + {info['max_output_tokens']} output tokens = {f(mm['need'])}; + margin {f(mm['margin'])} (max of {os.environ.get('C8_CONTEXT_MARGIN_TOKENS', 8192)} tokens and {os.environ.get('C8_CONTEXT_MARGIN_PCT', 2)}%), "
             f"rounded up to a multiple of 8192" + ("; CAPPED at the model's maximum" if mm["capped_at_model_max"] else "") + (f"; override C8_MAX_MODEL_LEN={mm['override']}" if mm["override"] else "")],
            ["--tensor-parallel-size", mem["tensor_parallel"], "4 GPUs (4,5,6,7); the 4 KV heads split one per GPU"],
            ["VLLM_ATTENTION_BACKEND", s["attention_backend"], f"dual chunk attention with sparse prefill, Qwen's setup for inputs beyond {f(sh.get('native_window'))} tokens; identical to full attention inside one chunk"],
            ["VLLM_USE_V1", s["use_v1"], "the dual chunk attention backend exists only in vLLM's V0 engine"],
            ["--enable-chunked-prefill / --max-num-batched-tokens", f"on / {f(s['max_num_batched_tokens'])}", "Qwen's setting; bounds the prefill activations of a 760K-token book"],
            ["--enforce-eager", s["enforce_eager"], "Qwen's setting for this backend (no CUDA graphs)"],
            ["--enable-prefix-caching", s["prefix_caching"], "queries on the same document reuse its KV cache; the timing step checks it works"],
            ["--gpu-memory-utilization", mem["gpu_memory_utilization"], "leaves ~4.8 GB per GPU for CUDA and NCCL"],
            ["--max-num-seqs / client concurrency", f"{s['max_num_seqs']} / {s['concurrency']}", "one uncached document at a time (sent alone), then that document's questions together"],
            ["temperature / top_p / seed / max_tokens", f"0 / 1 / 42 / {info['max_output_tokens']}", "same as C5's reader"],
        ]) + [""]
        L += ["Memory fit per GPU (estimate; vLLM's own profile at start-up is the real check):", ""]
        L += cc.md_table(["GPU memory", "x utilisation", "Weights", "Activations + CUDA (est.)", "Left for KV", "KV per token", "KV capacity", "Longest sequence needs", "Fits"],
                         [[f"{mem['gpu_mib']:,} MiB", mem["gpu_memory_utilization"], f"{mem['weights_gib_per_gpu']:.1f} GiB", f"{mem['activations_gib_per_gpu']:.1f} GiB",
                           f"{mem['kv_room_gib_per_gpu']:.1f} GiB", f"{mem['kv_kib_per_token_per_gpu']:.0f} KiB", f"{mem['kv_capacity_tokens']:,} tokens ({mem['headroom_sequences']:.1f} x the window)",
                           f"{mem['longest_sequence_kv_gib_per_gpu']:.1f} GiB", "yes" if mem["fits"] else "NO"]]) + [""]
        gm = s.get("gpus") or {}
        L += [f"GPU sizes from {gm.get('source')}: " + "; ".join(f"GPU {g['index']} {g['name']} {g['total_mib']:,} MiB" + (f" ({g['used_mib']:,} MiB in use now)" if g.get("used_mib") is not None else "") for g in gm.get("gpus", [])) + ".", ""]
        L += [f"Installed vLLM (`{v.get('python')}`): version {v.get('version')}; V0 engine present: {v.get('v0_engine')}; dual chunk attention backend file: {v.get('dca_backend_file')}"
              + (f"; error: {v.get('error')}" if v.get("error") else "") + ".", ""]
        if s.get("not_fitting"):
            L += [f"Documents whose prompt does NOT fit --max-model-len (not cut; their queries cannot run the full-document condition): {s['not_fitting']}.", ""]
        ov = s.get("over_native") or {}
        for key, o in ov.items():
            if o["docs"]:
                L += [f"{c8.DATASET_DISPLAY[key]}: {len(o['docs'])} documents ({o['queries']} queries) are longer than the native window of {f(sh.get('native_window'))} tokens: "
                      f"{', '.join(o['docs'])}. They need the dual chunk attention backend" + (" (found)." if v.get("dca_available") else " (NOT found in the installed vLLM: see Summary)."), ""]
    # Evidence position
    if info.get("evidence_rows"):
        L += ["## Position of the gold evidence in the document", "",
              "Mean position (in tokens) of the evidence chunks' midpoints; MuSiQue-32K: the paper's gold and silver chunks; NovelHopQA: C5's paragraph labels "
              "(each evidence paragraph located in SAADI's chunks). 'Spread' = the evidence chunks fall in more than one third.", ""]
        L += cc.md_table(["Dataset", "First third", "Middle third", "Last third", "No located evidence", "Spread over thirds"],
                         [[c8.DATASET_DISPLAY[r["dataset"]], r["first"], r["middle"], r["last"], r["none"], r["spread"]] for r in info["evidence_rows"]]) + [""]
    # Time
    if info.get("rough_time"):
        L += ["## Rough time estimate (a priori; the GPU script's timing run replaces it)", "",
              f"Assumes {EFFECTIVE_FLOPS / 1e12:.0f} TFLOP/s effective prefill on 4 GPUs (dense attention, no sparse speed-up) and {EXPECTED_OUTPUT_TOKENS} output tokens per answer.", ""]
        L += cc.md_table(["Dataset", "Cold prefill, once per document (h)", "Questions on a cached document (h)", "SAADI + Vanilla + closed-book (h)", "Wall with cache (h)", "GPU h with cache", "Full document WITHOUT cache, wall (h)"],
                         [[c8.DATASET_DISPLAY[k], f(t["full_doc_cold_prefill_h"], 2), f(t["full_doc_warm_h"], 2), f(t["short_conditions_h"], 2), f(t["wall_h_with_cache"], 2), f(t["gpu_h_with_cache"], 1), f(t["full_doc_wall_h_without_cache"], 1)]
                          for k, t in info["rough_time"].items()]) + [""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Next", "", "```bash", "nohup bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh > /dev/null 2>&1 &", "```", "",
          "(after C5's run on GPUs 4-7 has finished)", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info.get("files", [])]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=os.environ.get("C8_DATASETS", ",".join(c8.DATASETS)))
    args = ap.parse_args(argv)
    env = dict(os.environ)
    paths = c8.C8Paths.from_env()
    datasets = [d for d in args.datasets.split(",") if d]
    t0 = time.time()
    model = env.get("C8_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M")
    tok_name = env.get("C8_TOKENIZER") or model
    max_out = int(env.get("C8_MAX_OUTPUT_TOKENS", 64))
    info: dict[str, Any] = {"step": STEP, "argv": sys.argv[1:], "commit": cc.git_commit(paths.root), "host": os.uname().nodename,
                            "datasets": datasets, "model": {}, "tokenizer": tok_name, "max_output_tokens": max_out,
                            "per_dataset": {}, "files": [], "warnings": [], "summary": []}
    status, code = "OK", 0
    try:
        from saadi.rebuttal.heldout_fusion import PUBLISHED_POINT

        point, _ = s0.published_point_from_config(Path(env.get("C5_CONFIG", paths.root / "configs/methods/sadi.yaml")))
        info["published_point"] = point
        if tuple(point) != tuple(PUBLISHED_POINT):
            raise GateFailed(f"published point {point} differs from the repo's PUBLISHED_POINT {PUBLISHED_POINT}")
        info["preserved"] = s0.check_preserved(paths.root)
        if info["preserved"]["changed"] or info["preserved"]["missing"]:
            raise GateFailed(f"preserved sources changed: {info['preserved']['changed'][:5]} missing: {info['preserved']['missing'][:5]}")

        # 2. Model files (the tokenizer must be the model's own unless C8_TOKENIZER says otherwise).
        info["model"] = model_files(model)
        shape = model_shape(info["model"])
        info["shape"] = shape
        if not env.get("C8_TOKENIZER") and not info["model"].get("snapshot"):
            raise GateFailed(f"{model} is not in the Hugging Face cache ({env.get('HF_HUB_CACHE')}); download it first, e.g. "
                             f"HF_HUB_OFFLINE=0 HF_HOME={env.get('HF_HOME')} huggingface-cli download {model}")
        log(f"tokenizer {tok_name}")
        tok = s0.Tok(tok_name)
        counter = FullDocCounter(tok)

        doc_rows_all, prompt_rows, ev_rows = [], [], []
        per_query: dict[str, list[dict[str, Any]]] = {}
        docs_all: dict[str, dict[str, dict[str, Any]]] = {}
        for key in datasets:
            d = info["per_dataset"].setdefault(key, {})
            log(f"{key}: SAADI's saved queries, labels and chunks (C5's Dataset)")
            ds = s0.Dataset(key, paths.root)
            c5rows = check_c5(paths, key, ds.qids, d)
            para = load_para_labels(paths, key, d)
            for w in d.pop("warnings", []):
                info["warnings"].append(w)
            log(f"{key}: documents and token counts")
            docs = build_documents(ds, tok)
            docs_all[key] = docs
            qn = Counter(ds.doc_of[q] for q in ds.qids)
            # Seed-42 permutation of C5's query order: the timing sample is its first C8_TIMING_QUERIES,
            # a subset (if decided) its first N.
            import numpy as np

            order = [str(r["query_id"]) for r in c5rows["saadi"]]
            perm = np.random.default_rng(c8.SEED).permutation(len(order))
            rank = {order[int(j)]: i for i, j in enumerate(perm)}
            n_timing = int(env.get("C8_TIMING_QUERIES", 20))
            by_c5 = {c: {r["query_id"]: r for r in c5rows[c]} for c in c8.C5_CONDITIONS}
            rows = []
            first_of_doc = {}
            for q in sorted(order, key=lambda x: (ds.doc_of[x], rank[x])):
                first_of_doc.setdefault(ds.doc_of[q], q)
            log(f"{key}: prompt tokens of {len(order)} queries x 4 conditions")
            for q in order:
                base = by_c5["saadi"][q]
                doc = docs[ds.doc_of[q]]
                if base["doc_id"] != doc["doc_id"]:
                    raise GateFailed(f"{key}/{q}: C5 says document {base['doc_id']}, SAADI's saved rows say {doc['doc_id']}")
                lab = ds.labels[q]
                if para is not None:
                    ev_ids, ev_src = para.get(q, []), "C5 paragraph labels"
                else:
                    ev_ids, ev_src = lab.evidence_chunks(), "paper labels (gold + silver chunks)"
                pos = evidence_position(doc, ev_ids)
                pt = {"full_doc": counter.count(doc, base["question"], verify=(first_of_doc[doc["doc_id"]] == q))}
                ct = {"full_doc": doc["n_tokens"]}
                for c in c8.C5_CONDITIONS:
                    r = by_c5[c][q]
                    pt[c] = tok.prompt(cc.build_messages(r["question"], r["passages"]))
                    ct[c] = tok.count(cc.format_context(r["passages"])) if r["passages"] else 0
                rows.append({"query_id": q, "doc_id": doc["doc_id"], "question": base["question"], "gold_answers": base["gold_answers"],
                             "subset_rank": rank[q], "timing": rank[q] < n_timing, "doc_tokens": doc["n_tokens"], "doc_chunks": doc["n_chunks"],
                             "evidence_source": ev_src, "evidence_chunk_ids": ev_ids, "evidence_eligible": bool(ev_ids), **{f"evidence_{k}": v for k, v in pos.items()},
                             "prompt_tokens": pt, "context_tokens": ct,
                             "c5_evidence_in_context": {c: by_c5[c][q].get("evidence_in_context") for c in c8.C5_CONDITIONS},
                             "c5_retrieval_hit10": {c: by_c5[c][q].get("retrieval_hit10") for c in c8.C5_CONDITIONS}})
            per_query[key] = rows
            d["queries"] = len(rows)
            d["documents"] = len(docs)
            d["non_contiguous_documents"] = [x for x, v in docs.items() if not v["contiguous"]]
            if d["non_contiguous_documents"]:
                info["warnings"].append(f"{key}: chunk positions have gaps in {d['non_contiguous_documents'][:5]} (text joined in position order)")
            for x, v in docs.items():
                doc_rows_all.append({"dataset": key, "doc_id": x, "tokens": v["n_tokens"], "chunks": v["n_chunks"], "queries": qn.get(x, 0),
                                     "chunk_tokens_sum": v["chunk_tokens_sum"], "prompt_tokens_max": max((r["prompt_tokens"]["full_doc"] for r in rows if r["doc_id"] == x), default=0)})
            for c in c8.CONDITIONS:
                vals = [r["prompt_tokens"][c] for r in rows]
                prompt_rows.append({"dataset": key, "condition": c, "n": len(vals), "mean": mean(vals), "median": median(vals), "max": max(vals), "total": sum(vals)})
            ec = Counter((r["evidence_third"] or "none") for r in rows)
            ev_rows.append({"dataset": key, "first": ec.get("first", 0), "middle": ec.get("middle", 0), "last": ec.get("last", 0), "none": ec.get("none", 0),
                            "spread": sum(1 for r in rows if r["evidence_spread"]), "label_source": rows[0]["evidence_source"]})

        info["count_checks"] = {"n": len(counter.checked), "max_diff": max((abs(c["split_count"] - c["full_count"]) for c in counter.checked), default=0)}

        # 5. Settings.
        need = max(r["prompt_tokens"][c] for k in per_query for r in per_query[k] for c in c8.CONDITIONS) + max_out
        mm = choose_max_model_len(need, shape, env)
        gpus = gpu_memory(env.get("C8_GPUS", "4,5,6,7"))
        mem = memory_fit(shape, gpus, mm["max_model_len"], env)
        vchk = vllm_check(env.get("C8_VLLM_PY") or env.get("PYTHON") or sys.executable)
        settings = {"max_model_len": mm, "memory": mem, "gpus": gpus, "vllm": vchk,
                    "attention_backend": env.get("C8_ATTENTION_BACKEND", "DUAL_CHUNK_FLASH_ATTN"), "use_v1": env.get("C8_VLLM_USE_V1", "0"),
                    "max_num_batched_tokens": int(env.get("C8_MAX_NUM_BATCHED_TOKENS", 131072)), "enforce_eager": env.get("C8_ENFORCE_EAGER", "1") == "1",
                    "prefix_caching": env.get("C8_ENABLE_PREFIX_CACHING", "1") == "1", "max_num_seqs": int(env.get("C8_MAX_NUM_SEQS", 8)),
                    "concurrency": int(env.get("C8_READER_CONCURRENCY", 8)), "over_native": {}, "not_fitting": []}
        for key, rows in per_query.items():
            for r in rows:
                r["fits"] = r["prompt_tokens"]["full_doc"] + max_out <= mm["max_model_len"]
                r["over_native"] = r["prompt_tokens"]["full_doc"] > shape["native_window"]
            over = sorted({r["doc_id"] for r in rows if r["over_native"]}, key=lambda x: -docs_all[key][x]["n_tokens"])
            settings["over_native"][key] = {"docs": over, "queries": sum(1 for r in rows if r["over_native"])}
            settings["not_fitting"] += [f"{key}:{x}" for x in sorted({r["doc_id"] for r in rows if not r["fits"]})]
        for dr in doc_rows_all:
            rr = [r for r in per_query[dr["dataset"]] if r["doc_id"] == dr["doc_id"]]
            dr["fits"] = all(r["fits"] for r in rr) if rr else None
            dr["over_native"] = any(r["over_native"] for r in rr) if rr else None
        info["settings"] = settings

        # Write prepared files and CSVs.
        for key, rows in per_query.items():
            dpath = paths.documents_file(key)
            n_docs = cc.write_jsonl(dpath, list(docs_all[key].values()))
            qpath = paths.queries_file(key)
            cc.write_jsonl(qpath, rows)
            man = {"dataset": key, "created": cc.now(), "commit": info["commit"], "model": model, "tokenizer": tok_name,
                   "template_sha256": __import__("hashlib").sha256(cc.READER_TEMPLATE.encode()).hexdigest(),
                   "documents": {"file": paths.rel(dpath), "n": n_docs, "sha256": cc.sha256_file(dpath)},
                   "queries": {"file": paths.rel(qpath), "n": len(rows), "sha256": cc.sha256_file(qpath)},
                   "c5_inputs": info["per_dataset"][key]["c5_inputs"], "max_model_len": mm["max_model_len"],
                   "timing_queries": int(env.get("C8_TIMING_QUERIES", 20))}
            cc.write_json(paths.manifest(key), man)
            info["files"] += [paths.rel(dpath), paths.rel(qpath), paths.rel(paths.manifest(key))]
        settings_lines = [
            "# Written by C8 step 00 (sourced by sh/run_gpus_4567.sh). Re-run step 00 to change.",
            f"C8_STEP00_MAX_MODEL_LEN={mm['max_model_len']}",
            f"C8_STEP00_LONGEST_PROMPT={need - max_out}",
            f"C8_STEP00_KV_CAPACITY_EST={mem['kv_capacity_tokens']}",
            f"C8_STEP00_FITS={1 if mem['fits'] and not settings['not_fitting'] else 0}",
            f"C8_STEP00_DCA_AVAILABLE={1 if vchk.get('dca_available') else 0}",
            f"C8_STEP00_OVER_NATIVE_QUERIES={sum(o['queries'] for o in settings['over_native'].values())}",
        ]
        paths.settings_file.parent.mkdir(parents=True, exist_ok=True)
        paths.settings_file.write_text("\n".join(settings_lines) + "\n", encoding="utf-8")
        info["files"].append(paths.rel(paths.settings_file))
        prep = paths.results / "prepare"
        write_csv(prep / "doc_lengths.csv", doc_rows_all)
        write_csv(prep / "prompt_tokens.csv", prompt_rows)
        write_csv(prep / "evidence_position.csv", ev_rows)
        info["files"] += [paths.rel(prep / x) for x in ("doc_lengths.csv", "prompt_tokens.csv", "evidence_position.csv")]
        info["doc_rows"], info["prompt_rows"], info["evidence_rows"] = doc_rows_all, prompt_rows, ev_rows

        # Document statistics and the rough time.
        info["doc_stats"], info["rough_time"] = {}, {}
        for key in datasets:
            lens = [v["n_tokens"] for v in docs_all[key].values()]
            qs = [x["queries"] for x in doc_rows_all if x["dataset"] == key]
            info["doc_stats"][key] = {"queries": len(per_query[key]), "documents": len(lens), "min": min(lens), "median": median(lens), "mean": mean(lens), "max": max(lens),
                                      "q_min": min(qs), "q_median": median(qs), "q_max": max(qs), "chunks_mean": mean(v["n_chunks"] for v in docs_all[key].values())}
            info["rough_time"][key] = rough_hours(key, docs_all[key], per_query[key], shape, settings["concurrency"])

        # Summary.
        for key in datasets:
            s = info["doc_stats"][key]
            t = info["rough_time"][key]
            o = settings["over_native"][key]
            info["summary"].append(f"{c8.DATASET_DISPLAY[key]}: {s['queries']} queries on {s['documents']} distinct documents; document length {s['min']:,} / {s['median']:,.0f} / {s['max']:,} tokens "
                                   f"(min / median / max); {o['queries']} queries on {len(o['docs'])} documents beyond the native {shape['native_window']:,}-token window. "
                                   f"Rough estimate {t['gpu_h_with_cache']:.1f} GPU h with the prefix cache ({t['full_doc_gpu_h_without_cache']:.0f} GPU h for the full documents without it).")
        info["summary"].append(f"--max-model-len {mm['max_model_len']:,} (longest prompt {need - max_out:,} + {max_out} output tokens + margin). Estimated KV capacity on 4 GPUs: "
                               f"{mem['kv_capacity_tokens']:,} tokens, {'enough' if mem['fits'] else 'NOT enough'} for the longest document"
                               + (f"; documents that do not fit the window: {settings['not_fitting']}" if settings["not_fitting"] else "; every document fits, none is cut") + ".")
        n_over = sum(o["queries"] for o in settings["over_native"].values())
        if n_over and not vchk.get("dca_available"):
            info["summary"].append(f"The installed vLLM ({vchk.get('version')}, `{vchk.get('python')}`) has no dual chunk attention backend (V0 engine {vchk.get('v0_engine')}); "
                                   f"{n_over} queries need it. Install vLLM 0.10.1.1 in a separate environment and set C8_VLLM_PYTHON to its python before the GPU script.")
            status = "OK (vLLM needs the dual chunk attention backend)"
        if not info["model"].get("in_cache"):
            missing = info["model"].get("weight_files_missing") or ["the whole snapshot"]
            info["summary"].append(f"Model weights not complete in the cache ({len(missing)} files missing): download {model} before the GPU script.")
            status, code = "FAILED (model weights missing)", 1
        if not mem["fits"]:
            status, code = "FAILED (does not fit in memory)", 1
        info["one_line"] = (f"docs MuSiQue {info['doc_stats'].get('musique_32k', {}).get('max', 0):,} max, NovelHopQA {info['doc_stats'].get('novelhopqa', {}).get('max', 0):,} max tokens; "
                            f"max_model_len {mm['max_model_len']:,}; {'fits' if mem['fits'] else 'does NOT fit'} on 4 GPUs" + ("" if code == 0 else f"; {status}"))
    except GateFailed as exc:
        status, code = "FAILED", 1
        info["summary"].append(f"Stopped: {exc}")
        info["one_line"] = f"FAILED: {str(exc)[:130]}"
        info["traceback"] = traceback.format_exc()
    except Exception as exc:
        status, code = "FAILED (unexpected error)", 1
        info["summary"].append(f"Unexpected error: {type(exc).__name__}: {exc}")
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
        info["traceback"] = traceback.format_exc()
    finally:
        info["seconds"] = round(time.time() - t0)
        info["status"], info["exit_code"] = status, code
        info["files"] +=[paths.rel(paths.reports / "step_00.md"), paths.rel(paths.reports / "step_00.json")]
        write_report(paths, info, status)
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
