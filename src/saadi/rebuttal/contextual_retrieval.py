"""C1: Contextual Retrieval (Anthropic, 2024) on SAADI's four benchmarks (ICLR rebuttal).

Contextual Retrieval (CR) writes, for every chunk, one short context from the whole document
and indexes ``context + "\\n\\n" + chunk``. Everything here runs under SAADI's conditions:
the saved chunks of the paper's caches, the saved queries and evidence labels, the same
construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the same five retrievers and the same
HR evaluation.

Pieces, all imported rather than copied:

- Prompt: B2's ``prompt_for`` (experiments/probe_baselines/systems/b2_contextual/build.py),
  checked against Anthropic's published text below before use.
- Chunks / queries / labels: ``frozen_records.DATASETS`` and ``load_saved_live_rows``.
- Scores: the published text channel (``mbr._score_pool``, the function the text channel
  calls) on the contextualized texts; memory and summary channels unchanged, from C2's saved
  streams when they exist, else from the published path (``heldout_fusion._score_document``).
- Rankings: ``combine_mbr_variant_chunk_scores`` through ``heldout_fusion.repo_ranking``;
  Vanilla = the text channel alone (lambda_M = lambda_S = 0), SAADI = published weights.
- HR: ``evaluation.retrieval_metrics.evaluate_retrieval_results`` (the function behind
  ``evaluation/hr.py``); paired bootstrap from ``frozen_records``.

Deliberate choices fixed before any run (nothing is tuned on evaluated queries) are listed in
``DELIBERATE_DIFFERENCES`` and repeated in every report.
"""

from __future__ import annotations

import bisect
import gzip
import importlib.util
import json
import os
import pickle
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..evaluation.retrieval_metrics import evaluate_retrieval_results
from ..indexing.retrieval_methods.mbr import _score_pool, build_mbr_document_memory_index
from . import heldout_fusion as hf
from .frozen_records import DATASETS, load_saved_live_rows, paired_bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "C1_contextual_retrieval"
SEED = 42
N_BOOT = 10_000

DATASET_KEYS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
# Cheapest first in each group (fixed by cost, not by expected results).
GROUP_DATASETS = {"A": ("loogle", "novelhopqa"), "B": ("musique_32k", "qasper_64k")}
GROUP_GPUS = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
RETRIEVERS = hf.RETRIEVERS
DENSE_RETRIEVERS = hf.DENSE_RETRIEVERS
DATASET_DISPLAY = hf.DATASET_DISPLAY
RETRIEVER_DISPLAY = hf.RETRIEVER_DISPLAY
EXPECTED_QUERIES = {"loogle": 1832, "musique_32k": 900, "qasper_64k": 1333, "novelhopqa": 908}

# Construction LLM and decoding (SAADI's saved vLLM runtime, window raised to the model's native maximum).
LLM_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CONTEXT_MAX_TOKENS = 1024
VLLM_SETTINGS: dict[str, Any] = {
    "tensor_parallel_size": 4,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.8,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 16384,
    "enable_prefix_caching": True,
    "disable_custom_all_reduce": True,
    "seed": SEED,
}
SAMPLING: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": SEED}

# Text construction.
DOC_SEPARATOR = " "  # between saved chunks when the document is rebuilt (see DELIBERATE_DIFFERENCES)
CONTEXT_SEPARATOR = "\n\n"  # the cookbook's text_to_embed: f"{context}\n\n{chunk}"

# Hybrid: the cookbook's fixed published values (retrieve_advanced).
RECALL_N = 150
SEMANTIC_WEIGHT = 0.8
BM25_WEIGHT = 0.2

VANILLA_POINT = (0.0, 0.0, 1.0, 1.0)
PUBLISHED_POINT = hf.PUBLISHED_POINT
SYSTEMS = ("vanilla", "empty", "cr", "saadi", "saadi_cr")
SYSTEM_DISPLAY = {
    "vanilla": "Vanilla",
    "empty": "CR, empty context",
    "cr": "CR",
    "saadi": "SAADI",
    "saadi_cr": "SAADI on CR text",
}

# Correctness gates (C2's rules).
DENSE_HR_TOLERANCE_POINTS = hf.DENSE_HR_TOLERANCE_POINTS
DENSE_MIN_IDENTICAL_TOP10 = hf.DENSE_MIN_IDENTICAL_TOP10
DRIFT_ACCEPTED_RETRIEVERS = hf.DRIFT_ACCEPTED_RETRIEVERS
DRIFT_MAX_HR_POINTS = hf.DRIFT_MAX_HR_POINTS
# Datasets on which Table 1's Vanilla is the text channel of these chunks (C2's server run, 2026-11-09:
# equal to the rounding on MuSiQue-32K and QASPER-64K; on LooGLE and NovelHopQA Table 1 Vanilla comes
# from another run and differs, so there the gate uses the saved per-query text-only files).
TABLE1_VANILLA_IS_TEXT_CHANNEL = ("musique_32k", "qasper_64k")

# Generation time model, from B2's server run (2026-11-09, same model, TP 4, max_num_seqs 16, served
# window 262144): 3184 calls on 10 books of 167K-269K tokens took 9.3e-6 s per visible document token
# per call (spread 8.8e-6 to 9.9e-6), median context 135 tokens. Short documents add a decode floor.
GEN_SECONDS_PER_VISIBLE_TOKEN = 9.3e-6
GEN_SECONDS_FLOOR_PER_CALL = 0.15
VLLM_STARTUP_SECONDS = 420

ANTHROPIC_PROMPT = (
    "<document>\n{doc}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document for the purposes of "
    "improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
)

DELIBERATE_DIFFERENCES = [
    "Construction LLM: Qwen3-30B-A3B-Instruct-2507 (SAADI's construction model) instead of Claude 3 Haiku; greedy decoding (temperature 0), "
    f"max_tokens {CONTEXT_MAX_TOKENS} (the cookbook allows 2048), vLLM with SAADI's saved runtime (bf16, TP 4, max_num_seqs 16) and the model's native window.",
    "No reranker: SAADI uses none, and a reranker could sit on top of either method.",
    "BM25: SAADI's own BM25 (the paper's Vanilla) on the contextualized text as one field, instead of Elasticsearch with the English analyzer "
    "searching chunk and context as two fields (multi_match); only the indexed text changes.",
    f"Document text: rebuilt from the saved chunks joined by a single space ({DOC_SEPARATOR!r}). The chunker's sentence splitter already "
    "changed some whitespace inside chunks (e.g. '1912. 02866'), so no separator restores the source exactly; the rebuilt text equals the "
    "source up to whitespace. It is the text SAADI read.",
    "Long documents: when document + prompt + chunk + 1024 output tokens exceed the window, the document is split at chunk boundaries into "
    "the smallest number k of contiguous segments of near-equal token length that each fit, and each chunk sees its own segment.",
    "Prompt sent as one user message with the document first (Anthropic sends two content blocks so the document can be cached); vLLM's "
    "prefix cache plays the role of Anthropic's prompt caching.",
    "Dense retrievers: the paper's adapters with their own truncation (Contriever, Jina and Qwen 512 tokens, BGE-M3 8192); the context comes "
    "first, so on long chunks the end of the chunk is what gets cut (counted per retriever).",
    "Hybrid: the cookbook's fusion (top 150 of each list, 0.8/rank_dense + 0.2/rank_bm25, ties as the cookbook sorts them: higher chunk "
    "position first), applied identically to Vanilla, CR and SAADI.",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C1Paths:
    repo_root: Path
    tag: str = ""  # "" for the real run, "dry_run" or "dry_run_full" for CPU tests

    @property
    def dry_run(self) -> bool:
        return bool(self.tag)

    def _tagged(self, base: Path) -> Path:
        return base / self.tag if self.tag else base

    @property
    def out_dir(self) -> Path:
        return self._tagged(self.repo_root / "outputs" / "rebuttal" / EXPERIMENT_ID)

    @property
    def results_dir(self) -> Path:
        return self._tagged(self.repo_root / "results" / "rebuttal" / EXPERIMENT_ID)

    @property
    def exp_dir(self) -> Path:
        return self.repo_root / "experiments" / "rebuttal" / EXPERIMENT_ID

    @property
    def reports_dir(self) -> Path:
        return self._tagged(self.exp_dir / "reports")

    @property
    def logs_dir(self) -> Path:
        return self._tagged(self.exp_dir / "logs")

    def dataset_dir(self, dataset: str) -> Path:
        return self.out_dir / dataset

    def contexts_path(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "contexts.jsonl"

    def census_path(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "census.json"

    def generation_summary_path(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "generation.json"

    def marker(self, dataset: str, name: str) -> Path:
        return self.dataset_dir(dataset) / "markers" / name

    def job_dir(self, dataset: str, retriever: str) -> Path:
        return self.dataset_dir(dataset) / "retrieval" / retriever

    def job_done(self, dataset: str, retriever: str) -> Path:
        return self.marker(dataset, f"{retriever}.DONE")

    def job_failed(self, dataset: str, retriever: str) -> Path:
        return self.marker(dataset, f"{retriever}.FAILED")

    def rankings_path(self, dataset: str, retriever: str) -> Path:
        return self.job_dir(dataset, retriever) / "rankings.pkl.gz"

    def memory_check_path(self, group: str) -> Path:
        return self.out_dir / f"memory_check_group_{group}.json"

    def ensure(self, datasets: Iterable[str] = ()) -> None:
        for p in (self.out_dir, self.results_dir, self.reports_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)
        for d in datasets:
            (self.dataset_dir(d) / "markers").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    hf.write_json(path, payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path = REPO_ROOT) -> str:
    return hf.rel(path, root)


# ---------------------------------------------------------------------------
# Prompt (B2's prompt_for, checked against Anthropic's text)
# ---------------------------------------------------------------------------

_B2 = None


def b2_module():
    """Import B2's build.py (probe baseline) for its prompt; nothing in B2 is changed."""

    global _B2
    if _B2 is None:
        path = REPO_ROOT / "experiments" / "probe_baselines" / "systems" / "b2_contextual" / "build.py"
        spec = importlib.util.spec_from_file_location("c1_b2_build", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import B2 from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _B2 = mod
    return _B2


def check_prompt() -> dict[str, Any]:
    b2 = b2_module()
    sample = b2.prompt_for("DOC", "CHUNK")
    expected = ANTHROPIC_PROMPT.format(doc="DOC", chunk="CHUNK")
    if sample != expected:
        raise RuntimeError(f"B2's prompt differs from Anthropic's published prompt:\n{sample!r}\nvs\n{expected!r}")
    if b2.DOC_PROMPT.format(doc="DOC") + b2.CHUNK_PROMPT.format(chunk="CHUNK") != expected:
        raise RuntimeError("B2's DOC_PROMPT + CHUNK_PROMPT do not compose to prompt_for")
    return {"source": "experiments/probe_baselines/systems/b2_contextual/build.py:prompt_for", "equals_anthropic_prompt": True}


def doc_part(doc: str) -> str:
    return b2_module().DOC_PROMPT.format(doc=doc)


def chunk_part(chunk: str) -> str:
    return b2_module().CHUNK_PROMPT.format(chunk=chunk)


def contextualize(context: str, chunk: str) -> str:
    """The cookbook's text_to_embed. An empty context leaves the chunk text exactly as it is."""

    context = str(context or "").strip()
    return f"{context}{CONTEXT_SEPARATOR}{chunk}" if context else chunk


class PromptTokens:
    """Exact token ids of the chat-formatted prompt, split into a per-segment prefix and a per-chunk suffix.

    The rendered prompt is ``pre + doc_part(doc) + chunk_part(chunk) + post``. The prefix
    ``pre + doc_part`` is identical for every chunk of a segment, so tokenizing it once and the
    suffix separately gives every request of a segment byte-identical leading token ids, which is
    what the prefix cache matches on. ``check_joint`` compares with joint tokenization.
    """

    def __init__(self, tokenizer: Any):
        self.tok = tokenizer
        marker = "⁣C1⁣"
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True)
        if rendered.count(marker) != 1:
            raise RuntimeError("could not locate the user content in the chat template")
        self.pre, self.post = rendered.split(marker)
        self.template_tokens = len(self.encode(self.pre + doc_part("") + chunk_part("") + self.post))

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def prefix_ids(self, doc: str) -> list[int]:
        return self.encode(self.pre + doc_part(doc))

    def suffix_ids(self, chunk: str) -> list[int]:
        return self.encode(chunk_part(chunk) + self.post)

    def full_text(self, doc: str, chunk: str) -> str:
        return self.pre + doc_part(doc) + chunk_part(chunk) + self.post

    def check_joint(self, doc: str, chunk: str) -> bool:
        return self.encode(self.full_text(doc, chunk)) == self.prefix_ids(doc) + self.suffix_ids(chunk)


def load_qwen_tokenizer(model_id: str = LLM_MODEL_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def native_max_model_len(model_id: str = LLM_MODEL_ID) -> dict[str, Any]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    rope = getattr(cfg, "rope_scaling", None)
    return {
        "max_model_len": int(cfg.max_position_embeddings),
        "rope_scaling": rope,
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", 0) or cfg.hidden_size // cfg.num_attention_heads),
    }


# ---------------------------------------------------------------------------
# Data: the paper's chunks, queries and labels
# ---------------------------------------------------------------------------


@dataclass
class DocChunks:
    doc_id: str
    chunk_ids: list[str]
    texts: list[str]


def dataset_docs(dataset: str, *, max_docs: int | None = None, repo_root: Path = REPO_ROOT) -> list[DocChunks]:
    """Documents of the saved results in first-appearance order, with the chunk pool of the paper's cache."""

    spec = DATASETS[dataset]
    rows = load_saved_live_rows(spec, "bm25", repo_root=repo_root)
    doc_ids = list(dict.fromkeys(str(r.get("doc_id", "")) for r in rows))
    if max_docs:
        doc_ids = doc_ids[: int(max_docs)]
    out = []
    for doc_id in doc_ids:
        idx = build_mbr_document_memory_index(repo_root / spec.index_json, doc_id=doc_id)
        out.append(DocChunks(doc_id, [str(c.chunk_id) for c in idx.chunk_pool], [str(c.text) for c in idx.chunk_pool]))
    return out


def doc_text(chunks: list[str]) -> str:
    return DOC_SEPARATOR.join(chunks)


# ---------------------------------------------------------------------------
# Token census and segmentation
# ---------------------------------------------------------------------------


def equal_token_bounds(chunk_tokens: list[int], k: int) -> list[tuple[int, int]]:
    """k contiguous segments at chunk boundaries, each boundary nearest to an equal share of the tokens."""

    n = len(chunk_tokens)
    if k <= 1:
        return [(0, n)]
    cum = [0]
    for t in chunk_tokens:
        cum.append(cum[-1] + int(t))
    total = cum[-1]
    cuts = [0]
    for m in range(1, k):
        target = m * total / k
        i = bisect.bisect_left(cum, target)
        cand = [j for j in (i - 1, i) if 0 < j < n]
        j = min(cand, key=lambda x: (abs(cum[x] - target), x)) if cand else max(1, min(n - 1, i))
        j = max(j, cuts[-1] + 1)
        cuts.append(min(j, n - (k - m)))
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(k)]


def census_document(doc: DocChunks, pt: PromptTokens, max_model_len: int) -> dict[str, Any]:
    """Exact prompt lengths and the segment rule for one document."""

    chunk_tokens = [len(pt.encode(t)) for t in doc.texts]
    suffix_tokens = [len(pt.suffix_ids(t)) for t in doc.texts]
    prefix_cache: dict[tuple[int, int], int] = {}

    def prefix_len(a: int, b: int) -> int:
        if (a, b) not in prefix_cache:
            prefix_cache[(a, b)] = len(pt.prefix_ids(doc_text(doc.texts[a:b])))
        return prefix_cache[(a, b)]

    def fits(a: int, b: int) -> bool:
        return prefix_len(a, b) + max(suffix_tokens[a:b]) + CONTEXT_MAX_TOKENS <= max_model_len

    n = len(doc.texts)
    segments: list[tuple[int, int]] | None = None
    for k in range(1, n + 1):
        bounds = equal_token_bounds(chunk_tokens, k)
        if all(fits(a, b) for a, b in bounds):
            segments = bounds
            break
    if segments is None:
        raise RuntimeError(f"{doc.doc_id}: no segmentation fits the window {max_model_len}")
    seg_rows = []
    for si, (a, b) in enumerate(segments):
        seg_rows.append(
            {
                "segment": si,
                "start": a,
                "end": b,
                "chunks": b - a,
                "prefix_tokens": prefix_len(a, b),
                "max_prompt_tokens": prefix_len(a, b) + max(suffix_tokens[a:b]),
                "nominal_input_tokens": sum(prefix_len(a, b) + suffix_tokens[i] for i in range(a, b)),
            }
        )
    return {
        "doc_id": doc.doc_id,
        "chunks": n,
        "chunk_ids": list(doc.chunk_ids),
        "chunk_tokens": chunk_tokens,
        "suffix_tokens": suffix_tokens,
        "doc_tokens": prefix_len(0, n) - len(pt.encode(pt.pre + doc_part(""))),
        "k": len(segments),
        "segments": seg_rows,
        "nominal_input_tokens": sum(s["nominal_input_tokens"] for s in seg_rows),
    }


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }


def run_census(dataset: str, *, paths: C1Paths, max_docs: int | None, log=print) -> dict[str, Any]:
    path = paths.census_path(dataset)
    if path.exists():
        cached = read_json(path)
        if cached.get("max_docs") == max_docs:
            return cached
    t0 = time.perf_counter()
    tok = load_qwen_tokenizer()
    pt = PromptTokens(tok)
    model = native_max_model_len()
    if model.get("rope_scaling"):
        raise RuntimeError(f"{LLM_MODEL_ID} config has rope_scaling {model['rope_scaling']}; the run assumes the native window")
    docs = dataset_docs(dataset, max_docs=max_docs, repo_root=paths.repo_root)
    rows = []
    for di, doc in enumerate(docs):
        rows.append(census_document(doc, pt, model["max_model_len"]))
        if (di + 1) % 10 == 0:
            log(f"census {dataset}: {di + 1}/{len(docs)} documents")
    seg_docs = [r for r in rows if r["k"] > 1]
    joint_checks = []
    for doc, r in list(zip(docs, rows, strict=True))[:3]:
        a, b = r["segments"][0]["start"], r["segments"][0]["end"]
        joint_checks.append(pt.check_joint(doc_text(doc.texts[a:b]), doc.texts[a]))
    out = {
        "dataset": dataset,
        "max_docs": max_docs,
        "tokenizer": LLM_MODEL_ID,
        "model": model,
        "template_tokens": pt.template_tokens,
        "context_max_tokens": CONTEXT_MAX_TOKENS,
        "doc_separator": DOC_SEPARATOR,
        "n_docs": len(rows),
        "n_chunks": sum(r["chunks"] for r in rows),
        "doc_tokens": _dist([r["doc_tokens"] for r in rows]),
        "chunk_tokens": _dist([t for r in rows for t in r["chunk_tokens"]]),
        "docs_exceeding_window": len(seg_docs),
        "chunks_in_segmented_docs": sum(r["chunks"] for r in seg_docs),
        "k_values": {str(k): sum(1 for r in rows if r["k"] == k) for k in sorted({r["k"] for r in rows})},
        "segmented_docs": [{"doc_id": r["doc_id"], "doc_tokens": r["doc_tokens"], "k": r["k"], "chunks": r["chunks"]} for r in seg_docs],
        "nominal_input_tokens": sum(r["nominal_input_tokens"] for r in rows),
        "visible_tokens_per_call": _dist([s["prefix_tokens"] for r in rows for s in r["segments"] for _ in range(s["chunks"])]),
        "joint_tokenization_equals_split": joint_checks,
        "docs": rows,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(path, out)
    return out


def estimate_generation_seconds(census: dict[str, Any]) -> dict[str, Any]:
    calls = 0
    secs = 0.0
    for r in census["docs"]:
        for s in r["segments"]:
            calls += s["chunks"]
            secs += s["chunks"] * (GEN_SECONDS_PER_VISIBLE_TOKEN * s["prefix_tokens"] + GEN_SECONDS_FLOOR_PER_CALL)
    return {
        "calls": calls,
        "seconds": secs + VLLM_STARTUP_SECONDS,
        "basis": (
            f"per call {GEN_SECONDS_PER_VISIBLE_TOKEN:.1e} s x visible document tokens + {GEN_SECONDS_FLOOR_PER_CALL} s, "
            f"plus {VLLM_STARTUP_SECONDS} s engine start (B2's measured rate on the same model and runtime)"
        ),
    }


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


def load_contexts(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line after a crash; that chunk is regenerated
            out[str(row["chunk_id"])] = row
    return out


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash can leave a torn last line without a newline; start on a fresh line so no new row is glued to it.
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                with path.open("a", encoding="utf-8") as fix:
                    fix.write("\n")
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class StubGenerator:
    """Dry run: deterministic placeholder contexts; prompt tokens counted exactly, cache use simulated."""

    name = "stub"

    def __init__(self, tokenizer: Any):
        self.tok = tokenizer
        self.kv_tokens = 1_000_000

    def generate(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for req in requests:
            text = "Stub context: " + " ".join(str(req["neighbour_text"]).split()[:30])
            out.append(
                {
                    "context": text,
                    "prompt_tokens": len(req["prompt_ids"]),
                    "cached_tokens": 0 if req["warmup"] else int(req["prefix_tokens"]) // 16 * 16,
                    "cached_tokens_source": "simulated (stub)",
                    "generated_tokens": len(self.tok.encode(text, add_special_tokens=False)),
                    "finish_reason": "stop",
                }
            )
        return out


class VLLMGenerator:
    """Offline vLLM engine, SAADI's saved runtime with the native window; greedy, max_tokens 1024."""

    name = "vllm"

    def __init__(self, max_model_len: int, *, log=print):
        import vllm
        from vllm import LLM, SamplingParams

        self.version = getattr(vllm, "__version__", "?")
        kwargs = dict(VLLM_SETTINGS)
        kwargs["max_num_seqs"] = int(os.environ.get("C1_MAX_NUM_SEQS", kwargs["max_num_seqs"]))
        kwargs["gpu_memory_utilization"] = float(os.environ.get("C1_GPU_MEMORY_UTILIZATION", kwargs["gpu_memory_utilization"]))
        kwargs.update({"model": LLM_MODEL_ID, "max_model_len": int(max_model_len)})
        self.settings = dict(kwargs)
        try:
            self.llm = LLM(generation_config="vllm", **kwargs)
            self.settings["generation_config"] = "vllm"
        except TypeError:
            self.llm = LLM(**kwargs)
        self.params = SamplingParams(max_tokens=CONTEXT_MAX_TOKENS, **SAMPLING)
        self.kv_tokens = self._kv_tokens()
        log(f"vLLM {self.version} ready: {json.dumps(self.settings)}; KV cache about {self.kv_tokens} tokens")

    def _kv_tokens(self) -> int:
        eng = self.llm.llm_engine
        for cfg in (getattr(getattr(eng, "vllm_config", None), "cache_config", None), getattr(eng, "cache_config", None)):
            if cfg is not None and getattr(cfg, "num_gpu_blocks", None):
                return int(cfg.num_gpu_blocks) * int(cfg.block_size)
        return 400_000

    def generate(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            from vllm.inputs import TokensPrompt

            prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in requests]
        except ImportError:
            prompts = [{"prompt_token_ids": r["prompt_ids"]} for r in requests]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        rows = []
        for o in outs:
            comp = o.outputs[0] if o.outputs else None
            cached = getattr(o, "num_cached_tokens", None)
            rows.append(
                {
                    "context": (comp.text if comp else "").strip(),
                    "prompt_tokens": len(o.prompt_token_ids or []),
                    "cached_tokens": int(cached) if cached is not None else None,
                    "cached_tokens_source": "vllm num_cached_tokens" if cached is not None else "unavailable",
                    "generated_tokens": len(comp.token_ids) if comp else 0,
                    "finish_reason": comp.finish_reason if comp else "error",
                }
            )
        return rows


def generate_contexts(
    dataset: str,
    *,
    paths: C1Paths,
    census: dict[str, Any],
    generator: Any,
    max_docs: int | None,
    log=print,
    sub_batch: int = 128,
) -> dict[str, Any]:
    """All chunks of a segment together, its first request alone first (fills the prefix cache).

    Segments are grouped into waves whose prefixes fit in half the KV cache, so each prefix is
    computed once and stays cached while its chunks run. Rows are appended as they come in;
    a rerun skips chunks already in contexts.jsonl.
    """

    tok = load_qwen_tokenizer()
    pt = PromptTokens(tok)
    docs = {d.doc_id: d for d in dataset_docs(dataset, max_docs=max_docs, repo_root=paths.repo_root)}
    out_path = paths.contexts_path(dataset)
    done = load_contexts(out_path)
    segs = []
    for r in census["docs"]:
        doc = docs[r["doc_id"]]
        for s in r["segments"]:
            todo = [i for i in range(s["start"], s["end"]) if doc.chunk_ids[i] not in done]
            if todo:
                segs.append((doc, r, s, todo))
    log(f"generation {dataset}: {sum(len(t) for *_, t in segs)} chunks to do in {len(segs)} segments ({len(done)} already done)")
    budget = max(1, int(0.5 * generator.kv_tokens))
    waves: list[list[tuple]] = []
    cur: list[tuple] = []
    cur_tokens = 0
    for item in segs:
        need = item[2]["prefix_tokens"]
        if cur and cur_tokens + need > budget:
            waves.append(cur)
            cur, cur_tokens = [], 0
        cur.append(item)
        cur_tokens += need
    if cur:
        waves.append(cur)

    t_all = time.perf_counter()
    n_done = 0
    for wi, wave in enumerate(waves):
        prepared = []
        for doc, r, s, todo in wave:
            a, b = s["start"], s["end"]
            prefix = pt.prefix_ids(doc_text(doc.texts[a:b]))
            reqs = []
            for j, i in enumerate(todo):
                neighbour = doc.texts[i - 1] if i - 1 >= a else (doc.texts[i + 1] if i + 1 < b else doc.texts[i])
                reqs.append(
                    {
                        "doc_id": doc.doc_id,
                        "chunk_id": doc.chunk_ids[i],
                        "chunk_index": i,
                        "segment": s["segment"],
                        "n_segments": r["k"],
                        "segment_start": a,
                        "segment_end": b,
                        "prefix_tokens": len(prefix),
                        "prompt_ids": prefix + pt.suffix_ids(doc.texts[i]),
                        "warmup": j == 0,
                        "neighbour_text": neighbour,
                    }
                )
            prepared.append(reqs)
        batches = [[reqs[0] for reqs in prepared]]  # warm-up: one request per segment computes its prefix
        rest = [q for reqs in prepared for q in reqs[1:]]
        batches += [rest[i : i + sub_batch] for i in range(0, len(rest), sub_batch)]
        for batch in batches:
            if not batch:
                continue
            t0 = time.perf_counter()
            results = generator.generate(batch)
            dt = time.perf_counter() - t0
            rows = []
            for req, res in zip(batch, results, strict=True):
                rows.append(
                    {
                        "doc_id": req["doc_id"],
                        "chunk_id": req["chunk_id"],
                        "chunk_index": req["chunk_index"],
                        "segment": req["segment"],
                        "n_segments": req["n_segments"],
                        "segment_start": req["segment_start"],
                        "segment_end": req["segment_end"],
                        "warmup": req["warmup"],
                        "prefix_tokens": req["prefix_tokens"],
                        **res,
                        "batch_seconds": round(dt, 3),
                        "batch_size": len(batch),
                        "generator": generator.name,
                    }
                )
            _append_rows(out_path, rows)
            n_done += len(rows)
        log(f"generation {dataset}: wave {wi + 1}/{len(waves)} done, {n_done} contexts this run, {time.perf_counter() - t_all:.0f} s")
    return {"contexts_this_run": n_done, "seconds_this_run": round(time.perf_counter() - t_all, 1), "waves": len(waves)}


def generation_stats(dataset: str, *, paths: C1Paths, census: dict[str, Any]) -> dict[str, Any]:
    rows = list(load_contexts(paths.contexts_path(dataset)).values())
    expected = {cid for r in census["docs"] for cid in r["chunk_ids"]}
    missing = sorted(expected - {r["chunk_id"] for r in rows})
    gen = [int(r.get("generated_tokens", 0)) for r in rows]
    nominal = [int(r.get("prompt_tokens", 0)) for r in rows]
    cached_known = [r for r in rows if r.get("cached_tokens") is not None]
    computed = [int(r["prompt_tokens"]) - int(r["cached_tokens"]) for r in cached_known]
    n_docs = census["n_docs"]
    calls = len(rows)
    return {
        "calls": calls,
        "calls_per_chunk": calls / max(1, census["n_chunks"]),
        "calls_per_document": calls / max(1, n_docs),
        "missing_chunks": len(missing),
        "missing_examples": missing[:5],
        "empty_contexts": sum(1 for r in rows if not str(r.get("context", "")).strip()),
        "failed_contexts": sum(1 for r in rows if r.get("finish_reason") not in ("stop", "length")),
        "cut_at_max_tokens": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "cut_share": (sum(1 for r in rows if r.get("finish_reason") == "length") / calls) if calls else None,
        "generated_tokens": _dist(gen),
        "nominal_input_tokens": _dist(nominal),
        "computed_input_tokens": _dist(computed) if cached_known else {"n": 0},
        "cached_tokens_source": sorted({str(r.get("cached_tokens_source")) for r in rows}),
        "generator": sorted({str(r.get("generator")) for r in rows}),
        "context_chars": _dist([len(str(r.get("context", ""))) for r in rows]),
    }


def sanity_sample(dataset: str, *, paths: C1Paths, n: int = 10, max_docs: int | None = None) -> list[dict[str, Any]]:
    rows = load_contexts(paths.contexts_path(dataset))
    docs = {d.doc_id: d for d in dataset_docs(dataset, max_docs=max_docs, repo_root=paths.repo_root)}
    keys = sorted(rows)
    rng = random.Random(SEED)
    picked = rng.sample(keys, min(n, len(keys)))
    out = []
    for cid in picked:
        r = rows[cid]
        d = docs.get(r["doc_id"])
        chunk = d.texts[d.chunk_ids.index(cid)] if d and cid in d.chunk_ids else ""
        out.append(
            {
                "chunk_id": cid,
                "segment": f"{int(r['segment']) + 1}/{r['n_segments']}",
                "generated_tokens": r.get("generated_tokens"),
                "finish_reason": r.get("finish_reason"),
                "context": r.get("context", ""),
                "chunk": chunk,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Retrieval (one job = one dataset x retriever)
# ---------------------------------------------------------------------------


def c2_streams_available(dataset: str, retriever: str, repo_root: Path) -> dict[str, Any] | None:
    paths = hf.C2Paths(repo_root)
    job = hf.job_name(dataset, retriever)
    files = {
        "done": paths.done_marker(job),
        "stream": paths.stream_path(job),
        "queries": paths.queries_path(job),
        "check": paths.check_path(job),
    }
    if not all(p.exists() for p in files.values()):
        return None
    try:
        check = read_json(files["check"])
    except Exception:
        return None
    if not check.get("passed"):
        return None
    return {k: str(v) for k, v in files.items()} | {"check_passed": True, "drift_accepted": bool(check.get("drift_accepted"))}


def _load_c2_docs(info: dict[str, Any], retriever: str) -> dict[str, list[dict[str, Any]]]:
    qa = hf.load_query_arrays(Path(info["stream"]), Path(info["queries"]), retriever)
    meta = read_json(Path(info["queries"]))["queries"]
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for q, m in zip(qa, meta, strict=True):
        by_doc.setdefault(q.doc_id, []).append(
            {"row_index": int(m["row_index"]), "chunk_ids": q.chunk_ids, "chunk_index": [int(x) for x in q.chunk_index], "raw": q.raw}
        )
    return by_doc


def _ranking(chunk_ids: list[str], chunk_index: list[int], raw: np.ndarray, doc_id: str, retriever: str, point) -> list[str]:
    q = SimpleNamespace(doc_id=doc_id, chunk_ids=chunk_ids, chunk_index=np.asarray(chunk_index), raw=raw)
    return hf.repo_ranking(q, retriever, point)


def _with_text(raw: np.ndarray, chunk_ids: list[str], scores: dict[str, float]) -> np.ndarray:
    out = raw.copy()
    out[:, 0] = [float(scores[c]) for c in chunk_ids]
    return out


def _truncation_stats(retriever: str, texts: list[str]) -> dict[str, Any]:
    """How many texts exceed the retriever's max_length with its own tokenizer (what the adapter cuts)."""

    from ..indexing.retriever_adapters import build_encoded_text_pool

    if retriever == "bm25" or not texts:
        return {}
    pool = build_encoded_text_pool(texts[:1], retriever_name=retriever, chunk_ids=["probe"])
    r = pool.retriever
    tok = getattr(r, "_tokenizer", None)
    max_len = int(getattr(r, "max_length", 0) or 0)
    prefix = str(getattr(r, "document_prefix", "") or "")
    if tok is None or not max_len:
        return {"note": "tokenizer not available"}
    lens = [len(x) for x in tok([prefix + t for t in texts], add_special_tokens=True, truncation=False)["input_ids"]]
    over = [n - max_len for n in lens if n > max_len]
    return {
        "max_length": max_len,
        "texts": len(lens),
        "over_max_length": len(over),
        "tokens_cut_total": int(sum(over)),
        "longest_tokens": max(lens),
    }


def run_retrieval_job(
    dataset: str,
    retriever: str,
    *,
    paths: C1Paths,
    max_docs: int | None,
    use_c2: bool = True,
    log=print,
) -> dict[str, Any]:
    """Pass 1: the empty-context check against Vanilla and the saved results. Pass 2: real contexts.

    Saves the top RECALL_N of every system's ranking per query (chunk positions, gzip pickle).
    Writes DONE or FAILED; a failing gate stops this retriever before real contexts are used.
    """

    repo = paths.repo_root
    spec = DATASETS[dataset]
    t_start = time.perf_counter()
    saved_rows = load_saved_live_rows(spec, retriever, repo_root=repo)
    by_doc = hf.rows_by_document(saved_rows)
    doc_ids = list(by_doc)[: int(max_docs)] if max_docs else list(by_doc)
    contexts = {cid: r.get("context", "") for cid, r in load_contexts(paths.contexts_path(dataset)).items()}
    c2 = c2_streams_available(dataset, retriever, repo) if use_c2 else None
    c2_docs = _load_c2_docs(c2, retriever) if c2 else {}
    work = paths.job_dir(dataset, retriever) / "work"
    work.mkdir(parents=True, exist_ok=True)
    timings = {"docs": len(doc_ids), "docs_from_checkpoint": 0, "seconds_pass1": 0.0, "seconds_pass2": 0.0}
    docs_out: list[dict[str, Any]] = []
    channel_source: dict[str, int] = {"c2_streams": 0, "published_path": 0}
    cr_texts_all: list[str] = []
    plain_texts_all: list[str] = []
    cr_texts_by_doc: dict[str, list[str]] = {}

    for di, doc_id in enumerate(doc_ids):
        ck = work / f"doc_{di:04d}.pkl"
        idx = build_mbr_document_memory_index(repo / spec.index_json, doc_id=doc_id)
        chunk_ids = [str(c.chunk_id) for c in idx.chunk_pool]
        plain = [str(c.text) for c in idx.chunk_pool]
        missing = [c for c in chunk_ids if c not in contexts]
        if missing:
            raise RuntimeError(f"{dataset}/{doc_id}: {len(missing)} chunks have no context (generation not finished), e.g. {missing[0]}")
        cr_texts = [contextualize(contexts[c], t) for c, t in zip(chunk_ids, plain, strict=True)]
        cr_texts_by_doc[doc_id] = cr_texts
        cr_texts_all += cr_texts
        plain_texts_all += plain
        if ck.exists():
            with ck.open("rb") as handle:
                doc = pickle.load(handle)
            if doc.get("doc_id") == doc_id and doc.get("chunk_ids") == chunk_ids:
                docs_out.append(doc)
                timings["docs_from_checkpoint"] += 1
                channel_source[doc["channel_source"]] += 1
                continue
        t0 = time.perf_counter()
        source = "published_path"
        qrows = None
        if doc_id in c2_docs and all(q["chunk_ids"] == chunk_ids for q in c2_docs[doc_id]):
            qrows = c2_docs[doc_id]
            source = "c2_streams"
        if qrows is None:
            scored = hf._score_document(idx, by_doc[doc_id], retriever)
            chunk_index = scored["chunk_index"]
            qrows = [
                {"row_index": q["row_index"], "chunk_ids": chunk_ids, "chunk_index": chunk_index, "raw": q["raw"]}
                for q in scored["queries"]
            ]
        channel_source[source] += 1
        queries = []
        t1 = time.perf_counter()
        for q in qrows:
            row = saved_rows[q["row_index"]]
            question = str(row.get("question", "") or "")
            raw = np.asarray(q["raw"], dtype=np.float64)
            empty = _score_pool(plain, ids=chunk_ids, query=question, retriever_name=retriever)
            r_van = _ranking(chunk_ids, q["chunk_index"], raw, doc_id, retriever, VANILLA_POINT)
            r_emp = _ranking(chunk_ids, q["chunk_index"], _with_text(raw, chunk_ids, empty), doc_id, retriever, VANILLA_POINT)
            r_saadi = _ranking(chunk_ids, q["chunk_index"], raw, doc_id, retriever, PUBLISHED_POINT)
            queries.append(
                {
                    "row_index": q["row_index"],
                    "query_id": str(row.get("query_id", "")),
                    "question": question,
                    "chunk_index": list(q["chunk_index"]),
                    "raw": raw,
                    "empty_text": np.asarray([float(empty[c]) for c in chunk_ids]),
                    "rank": {"vanilla": r_van, "empty": r_emp, "saadi": r_saadi},
                }
            )
        timings["seconds_pass1"] += time.perf_counter() - t1 + (t1 - t0)
        doc = {"doc_id": doc_id, "chunk_ids": chunk_ids, "channel_source": source, "queries": queries, "pass2": False}
        with ck.open("wb") as handle:
            pickle.dump(doc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        docs_out.append(doc)
        log(
            f"[{dataset}__{retriever}] pass 1 doc {di + 1}/{len(doc_ids)} {doc_id}: {len(queries)} queries ({source}), {time.perf_counter() - t0:.1f}s"
        )

    used_rows = [saved_rows[q["row_index"]] for d in docs_out for q in d["queries"]]
    gate = empty_context_gate(dataset, retriever, docs_out, used_rows, repo_root=repo, full_dataset=not max_docs, c2_used=bool(c2))
    summary: dict[str, Any] = {
        "dataset": dataset,
        "retriever": retriever,
        "max_docs": max_docs,
        "c2_streams": c2,
        "channel_source_docs": channel_source,
        "gate": gate,
        "queries": len(used_rows),
    }
    if not gate["passed"]:
        summary["status"] = "FAILED"
        summary["timings"] = timings | {"seconds_total": round(time.perf_counter() - t_start, 1)}
        write_json(paths.job_failed(dataset, retriever), summary)
        return summary

    # Pass 2: real contexts.
    for di, doc in enumerate(docs_out):
        if doc.get("pass2"):
            continue
        t0 = time.perf_counter()
        doc_id, chunk_ids = doc["doc_id"], doc["chunk_ids"]
        cr_texts = cr_texts_by_doc[doc_id]
        for q in doc["queries"]:
            cr = _score_pool(cr_texts, ids=chunk_ids, query=q["question"], retriever_name=retriever)
            raw_cr = _with_text(q["raw"], chunk_ids, cr)
            q["cr_text"] = raw_cr[:, 0].copy()
            q["rank"]["cr"] = _ranking(chunk_ids, q["chunk_index"], raw_cr, doc_id, retriever, VANILLA_POINT)
            q["rank"]["saadi_cr"] = _ranking(chunk_ids, q["chunk_index"], raw_cr, doc_id, retriever, PUBLISHED_POINT)
        doc["pass2"] = True
        with (work / f"doc_{di:04d}.pkl").open("wb") as handle:
            pickle.dump(doc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        timings["seconds_pass2"] += time.perf_counter() - t0
        log(f"[{dataset}__{retriever}] pass 2 doc {di + 1}/{len(docs_out)} {doc_id}: {time.perf_counter() - t0:.1f}s")

    save_rankings(paths.rankings_path(dataset, retriever), docs_out)
    summary["truncation"] = {
        "contextualized": _truncation_stats(retriever, cr_texts_all),
        "plain": _truncation_stats(retriever, plain_texts_all),
    }
    summary["status"] = "DONE"
    summary["timings"] = timings | {"seconds_total": round(time.perf_counter() - t_start, 1)}
    summary["device"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size_env": {k: v for k, v in os.environ.items() if k.endswith("_BATCH_SIZE") and k.startswith("SAADI_")},
        "device_map_env": {k: v for k, v in os.environ.items() if k.endswith("_DEVICE_MAP") and k.startswith("SAADI_")},
    }
    write_json(paths.job_done(dataset, retriever), summary)
    if paths.job_failed(dataset, retriever).exists():
        paths.job_failed(dataset, retriever).unlink()
    return summary


def save_rankings(path: Path, docs: list[dict[str, Any]]) -> None:
    payload = []
    for d in docs:
        pos = {c: i for i, c in enumerate(d["chunk_ids"])}
        payload.append(
            {
                "doc_id": d["doc_id"],
                "chunk_ids": d["chunk_ids"],
                "queries": [
                    {
                        "row_index": q["row_index"],
                        "query_id": q["query_id"],
                        "rank": {s: np.asarray([pos[c] for c in r[:RECALL_N]], dtype=np.int32) for s, r in q["rank"].items()},
                    }
                    for q in d["queries"]
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_rankings(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Correctness gate (empty context)
# ---------------------------------------------------------------------------


def _hr_of(rows: list[dict[str, Any]], rankings: list[list[str]]) -> dict[str, Any]:
    return hf._hr(rows, [r[:10] for r in rankings])


def empty_context_gate(
    dataset: str,
    retriever: str,
    docs: list[dict[str, Any]],
    used_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    full_dataset: bool,
    c2_used: bool,
) -> dict[str, Any]:
    """CR with empty contexts must reproduce Vanilla; the pool must reproduce the paper's saved rankings."""

    queries = [q for d in docs for q in d["queries"]]
    n = len(queries)
    dense = retriever != "bm25"
    # (1) The paper's chunks and queries: SAADI with published weights vs the saved rankings (C2's gate).
    docs_like = [
        {
            "doc_id": d["doc_id"],
            "queries": [
                {"published_top": q["rank"]["saadi"][:10], "published_scores": [], "vanilla_top": q["rank"]["vanilla"][:10]}
                for q in d["queries"]
            ],
            "anomalies": [],
        }
        for d in docs
    ]
    repro = hf.reproduction_check(dataset, retriever, docs_like, used_rows, repo_root=repo_root, full_dataset=full_dataset)
    problems = list(repro.get("problems", []))
    notes: list[str] = []

    # (2) Empty context vs Vanilla (the same text channel, through the CR path).
    van = [q["rank"]["vanilla"] for q in queries]
    emp = [q["rank"]["empty"] for q in queries]
    same_full = sum(1 for a, b in zip(van, emp, strict=True) if a[:RECALL_N] == b[:RECALL_N])
    same_top10 = sum(1 for a, b in zip(van, emp, strict=True) if a[:10] == b[:10])
    hr_van = _hr_of(used_rows, van)
    hr_emp = _hr_of(used_rows, emp)
    max_text_diff = max((float(np.nanmax(np.abs(q["raw"][:, 0] - q["empty_text"]))) for q in queries), default=0.0)
    if not dense:
        if same_full != n:
            problems.append(f"empty-context BM25 top-{RECALL_N} differs from Vanilla for {n - same_full} of {n} queries")
    else:
        drift = []
        if n and same_top10 / n < DENSE_MIN_IDENTICAL_TOP10:
            drift.append(f"empty-context top-10 identical to Vanilla for {same_top10}/{n} (< {DENSE_MIN_IDENTICAL_TOP10:.0%})")
        for k in ("5", "10"):
            if abs(hr_emp["hr_percent"][k] - hr_van["hr_percent"][k]) > DENSE_HR_TOLERANCE_POINTS + 1e-9:
                drift.append(f"empty-context HR@{k} {hr_emp['hr_percent'][k]:.2f} vs Vanilla {hr_van['hr_percent'][k]:.2f}")
        gap = max(abs(hr_emp["hr_percent"][k] - hr_van["hr_percent"][k]) for k in ("5", "10"))
        if drift and retriever in DRIFT_ACCEPTED_RETRIEVERS and gap <= DRIFT_MAX_HR_POINTS + 1e-9:
            notes += drift + [f"accepted as embedding drift (C2's rule for {retriever}: HR gap {gap:.2f} <= {DRIFT_MAX_HR_POINTS})"]
        else:
            problems += drift

    # (3) Saved per-query text-only rankings (BM25 on LooGLE and NovelHopQA): identical.
    pq = hf.per_query_vanilla_file(dataset, retriever, repo_root)
    pq_out: dict[str, Any] | None = None
    if pq is not None:
        saved_text = json.loads(pq.read_text(encoding="utf-8"))
        by_key = {
            (str(r.get("doc_id")), str(r.get("query_id"))): [str(x) for x in r.get("retrieved_chunk_ids", [])][:10] for r in saved_text
        }
        same = sum(
            1 for row, e in zip(used_rows, emp, strict=True) if by_key.get((str(row.get("doc_id")), str(row.get("query_id")))) == e[:10]
        )
        pq_out = {"file": rel(pq, repo_root), "identical_top10": same, "queries": n}
        if same != n:
            problems.append(f"empty-context top-10 differs from {pq.name} for {n - same} of {n} queries")

    # (4) Table 1 Vanilla where it is the text channel of these chunks (MuSiQue-32K, QASPER-64K).
    t1 = hf.published_vanilla(repo_root)
    t1_out = {}
    for k in (5, 10):
        ref = t1.get((hf.VANILLA_DATASET_NAME[dataset], retriever, k))
        ours = hr_emp["hr_percent"][str(k)]
        t1_out[str(k)] = {"table1": ref, "empty_context": ours}
        if ref is None or ours is None or not full_dataset:
            continue
        tol = 0.05 if not dense else 0.05 + DENSE_HR_TOLERANCE_POINTS
        if retriever in DRIFT_ACCEPTED_RETRIEVERS:
            tol = DRIFT_MAX_HR_POINTS
        if abs(ours - ref) > tol + 1e-9:
            msg = f"empty-context HR@{k} {ours:.2f} vs Table 1 Vanilla {ref:.1f} (tolerance {tol:.2f})"
            if dataset in TABLE1_VANILLA_IS_TEXT_CHANNEL:
                problems.append(msg)
            else:
                notes.append(msg + " - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks")

    examples = []
    for d in docs:
        for q in d["queries"]:
            if q["rank"]["vanilla"][:10] != q["rank"]["empty"][:10] and len(examples) < 3:
                examples.append(
                    {
                        "doc_id": d["doc_id"],
                        "query_id": q["query_id"],
                        "vanilla_top10": q["rank"]["vanilla"][:10],
                        "empty_top10": q["rank"]["empty"][:10],
                    }
                )
    return {
        "passed": not problems,
        "problems": problems,
        "notes": notes,
        "saadi_reproduction": {
            "identical_top10": repro.get("identical_top10"),
            "queries": repro.get("n_queries"),
            "hr_ours": repro.get("hr_ours_published"),
            "hr_saved": repro.get("hr_saved"),
            "expected_metrics": repro.get("expected_metrics"),
            "drift_accepted": repro.get("drift_accepted"),
            "drift_notes": repro.get("drift_notes"),
            "text_only_vs_table1": repro.get("text_only"),
        },
        "empty_vs_vanilla": {
            "identical_top150": same_full,
            "identical_top10": same_top10,
            "queries": n,
            "hr_vanilla": hr_van,
            "hr_empty": hr_emp,
            "max_abs_text_score_diff": max_text_diff,
            "reference": "C2 saved streams (independent run)" if c2_used else "published path in this process (shared encoding cache)",
        },
        "saved_per_query_text_only": pq_out,
        "table1_vanilla": t1_out,
        "mismatch_examples": examples,
    }


# ---------------------------------------------------------------------------
# Hybrid (cookbook retrieve_advanced) and evaluation
# ---------------------------------------------------------------------------


def hybrid_fuse(
    dense: list[int], bm25: list[int], *, n: int = RECALL_N, w_dense: float = SEMANTIC_WEIGHT, w_bm25: float = BM25_WEIGHT
) -> list[int]:
    """0.8 * 1/rank_dense + 0.2 * 1/rank_bm25 over the top n of each list (1-based ranks).

    Ties are broken as the cookbook sorts (score, doc_id, original_index) in reverse: higher
    chunk position first (one document per query here).
    """

    d = {int(c): i + 1 for i, c in enumerate(dense[:n])}
    b = {int(c): i + 1 for i, c in enumerate(bm25[:n])}
    score = {c: (w_dense / d[c] if c in d else 0.0) + (w_bm25 / b[c] if c in b else 0.0) for c in set(d) | set(b)}
    return sorted(score, key=lambda c: (score[c], c), reverse=True)[:n]


def load_job(paths: C1Paths, dataset: str, retriever: str) -> dict[tuple[str, str], dict[str, Any]]:
    """(doc_id, query_id) -> {'chunk_ids', 'rank': {system: [chunk ids]}}."""

    out = {}
    for d in load_rankings(paths.rankings_path(dataset, retriever)):
        for q in d["queries"]:
            out[(d["doc_id"], q["query_id"])] = {
                "chunk_ids": d["chunk_ids"],
                "rank_pos": {s: [int(x) for x in v] for s, v in q["rank"].items()},
                "row_index": q["row_index"],
            }
    return out


def per_query_hits(rows: list[dict[str, Any]], rankings: list[list[str]]) -> dict[str, np.ndarray]:
    """Per-query hit (0/100) at 5 and 10 for eligible queries, with the paper's HR function."""

    report = evaluate_retrieval_results(hf._results_from(rows, rankings), eval_k=10, retrieve_k=None, compute_per_query=True)
    out = {}
    for k in ("5", "10"):
        out[k] = np.asarray([100.0 * float(bool(p["hit_at_k"][k])) for p in report.per_query if p["eligible"]], dtype=float)
    out["eligible"] = int(report.n_union_eligible)
    return out


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    mean, lo, hi = paired_bootstrap_ci(a - b, n_resamples=N_BOOT, seed=SEED)
    return {"diff": mean, "ci": [lo, hi]}


def analyze_dataset(dataset: str, *, paths: C1Paths, retrievers: Iterable[str]) -> dict[str, Any]:
    """Per-retriever and hybrid HR with paired bootstrap intervals, from the saved rankings.

    In a dry run (BM25 only) the hybrid fuses BM25 with itself as a stand-in for a dense
    retriever, so the fusion code and the hybrid tables are exercised; it is labelled as such.
    """

    repo = paths.repo_root
    spec = DATASETS[dataset]
    jobs = {r: load_job(paths, dataset, r) for r in retrievers if paths.job_done(dataset, r).exists()}
    per_ret: dict[str, Any] = {}
    hits_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    keys_by_ret: dict[str, list[tuple[str, str]]] = {}
    rows_by_ret: dict[str, list[dict[str, Any]]] = {}
    for r, job in jobs.items():
        saved = load_saved_live_rows(spec, r, repo_root=repo)
        keys = [k for k in ((str(x.get("doc_id")), str(x.get("query_id"))) for x in saved) if k in job]
        lookup = {(str(x.get("doc_id")), str(x.get("query_id"))): x for x in saved}
        rows = [lookup[k] for k in keys]
        keys_by_ret[r], rows_by_ret[r] = keys, rows
        res = {"queries": len(rows)}
        for s in SYSTEMS:
            ranks = [[job[k]["chunk_ids"][p] for p in job[k]["rank_pos"][s]] for k in keys]
            h = per_query_hits(rows, ranks)
            hits_cache[(r, s)] = h
            res[s] = {
                "hr5": float(h["5"].mean()) if h["5"].size else None,
                "hr10": float(h["10"].mean()) if h["10"].size else None,
                "eligible": h["eligible"],
            }
        for k in ("5", "10"):
            res[f"cr_minus_vanilla_hr{k}"] = compare(hits_cache[(r, "cr")][k], hits_cache[(r, "vanilla")][k])
            res[f"saadi_minus_cr_hr{k}"] = compare(hits_cache[(r, "saadi")][k], hits_cache[(r, "cr")][k])
            res[f"saadi_cr_minus_cr_hr{k}"] = compare(hits_cache[(r, "saadi_cr")][k], hits_cache[(r, "cr")][k])
            res[f"saadi_cr_minus_saadi_hr{k}"] = compare(hits_cache[(r, "saadi_cr")][k], hits_cache[(r, "saadi")][k])
        per_ret[r] = res

    hybrid: dict[str, Any] = {}
    if "bm25" in jobs:
        dense_done = [r for r in DENSE_RETRIEVERS if r in jobs] or (["bm25"] if paths.dry_run else [])
        for d in dense_done:
            common = [k for k in keys_by_ret[d] if k in jobs["bm25"]]
            lookup = {(str(x.get("doc_id")), str(x.get("query_id"))): x for x in rows_by_ret[d]}
            rows = [lookup[k] for k in common]
            res = {"queries": len(rows)}
            hh = {}
            for s in ("vanilla", "cr", "saadi"):
                ranks = []
                for k in common:
                    a, b = jobs[d][k], jobs["bm25"][k]
                    if a["chunk_ids"] != b["chunk_ids"]:
                        raise RuntimeError(f"{dataset} {k}: chunk pools differ between {d} and bm25")
                    fused = hybrid_fuse(a["rank_pos"][s], b["rank_pos"][s])
                    ranks.append([a["chunk_ids"][p] for p in fused])
                h = per_query_hits(rows, ranks)
                hh[s] = h
                res[s] = {"hr5": float(h["5"].mean()), "hr10": float(h["10"].mean()), "eligible": h["eligible"]}
            for k in ("5", "10"):
                res[f"cr_minus_vanilla_hr{k}"] = compare(hh["cr"][k], hh["vanilla"][k])
                res[f"saadi_minus_cr_hr{k}"] = compare(hh["saadi"][k], hh["cr"][k])
            hybrid[d] = res
    return {"dataset": dataset, "per_retriever": per_ret, "hybrid": hybrid, "retrievers_done": sorted(jobs)}


# ---------------------------------------------------------------------------
# Published numbers next to C1
# ---------------------------------------------------------------------------


def table1_rows(repo_root: Path = REPO_ROOT) -> dict[tuple[str, str, str], dict[str, float]]:
    """(dataset, retriever, method) -> {'hr5', 'hr10'} from results/tables/main/retrieval_comparison/table.csv."""

    import csv

    path = repo_root / "results" / "tables" / "main" / "retrieval_comparison" / "table.csv"
    disp_ret = {v: k for k, v in RETRIEVER_DISPLAY.items()}
    out = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            r = disp_ret.get(row["Retriever"])
            if r is None:
                continue
            for ds, disp in DATASET_DISPLAY.items():
                try:
                    out[(ds, r, row["Method"])] = {"hr5": float(row[f"{disp} HR@5 (%)"]), "hr10": float(row[f"{disp} HR@10 (%)"])}
                except (KeyError, ValueError):
                    continue
    return out


def c2_heldout(repo_root: Path = REPO_ROOT) -> dict[tuple[str, str], dict[str, float]] | None:
    """SAADI with held-out fusion weights (C2, A1) if C2's final report exists."""

    import csv

    base = repo_root / "results" / "rebuttal" / "C2_heldout_fusion"
    table = base / "tables" / "d_per_pair.csv"
    if not (base / "REPORT.md").exists() or not table.exists():
        return None
    out = {}
    with table.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(row["dataset"], row["retriever"])] = {"hr5": float(row["a1_hr5"]), "hr10": float(row["a1_hr10"])}
            except (KeyError, ValueError):
                continue
    return out or None


def saadi_costs(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, float]]:
    """SAADI's construction cost per dataset (memory calls + hierarchy calls), from the paper's tables."""

    import csv

    out: dict[str, dict[str, float]] = {}
    path = repo_root / "results" / "tables" / "construction" / "cost" / "table.csv"
    disp = {v: k for k, v in DATASET_DISPLAY.items()}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for col, val in row.items():
                if col in disp:
                    out.setdefault(disp[col], {})[row["metric"]] = float(val)
    cpd = repo_root / "results" / "tables" / "construction" / "calls_per_document" / "table.csv"
    with cpd.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ds = disp.get(row.get("Dataset", ""))
            if ds:
                out.setdefault(ds, {})["documents"] = float(row["Documents"])
                out[ds]["total_calls_per_document"] = float(row["Total calls/document"])
    for v in out.values():
        chunks = v.get("actual_chunks") or 1.0
        v["calls_total"] = v.get("chunk_update_calls_total", 0) + v.get("hierarchy_calls_total", 0)
        v["input_tokens_total"] = v.get("chunk_update_input_tokens_total", 0) + v.get("hierarchy_input_tokens_total", 0)
        v["generated_tokens_total"] = v.get("chunk_update_generated_tokens_total", 0) + v.get("hierarchy_generated_tokens_total", 0)
        v["calls_per_chunk"] = v["calls_total"] / chunks
        v["input_tokens_per_chunk"] = v["input_tokens_total"] / chunks
        v["generated_tokens_per_chunk"] = v["generated_tokens_total"] / chunks
    return out
