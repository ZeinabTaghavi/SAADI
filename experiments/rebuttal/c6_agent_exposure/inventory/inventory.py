#!/usr/bin/env python3
# ruff: noqa: UP031, B007, B905  (kept as run on the servers; written for Python >= 3.6)
"""C6 step 1: read-only inventory of the saved outputs of the comparison methods.

SAADI ICLR rebuttal, experiment C6 (per-query agent comparison, reviewer mokb Q2).

Standard library only (Python >= 3.6). No GPU, no network, no SAADI install.

For every method folder under each --root it reports: git state (and whether the
output files are tracked, untracked or ignored), file counts and sizes per type,
the largest files, dataset and retriever coverage, query counts, the schema of
each kind of output file with one truncated sample record, which per-query trace
fields exist (retrieved ids, search calls, tokens, previews, answer, gold/hit),
and anything that looks like a run config.

STRICTLY READ-ONLY on the scanned roots: files are opened for reading only, git is
run with read-only subcommands and GIT_OPTIONAL_LOCKS=0, symlinks are not
followed, and nothing is written anywhere except the two report files in --out
(which must not be inside a scanned root). Secrets are redacted; .env and other
credential-looking files are counted but never opened.

One Markdown report (about 300 lines at most) and one JSON report are written per
root: <out>/inventory_<machine>_<root-name>.{md,json}. A failing run still writes
its report (with the traceback) and exits non-zero.

Examples
  python3 inventory.py --machine local   --root outputs/other_methods_method --out reports
  python3 inventory.py --machine server1 --root ./SAADI_UMA --out ~/c6_reports
"""

import argparse
import collections
import csv
import gzip
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime

TOOL_VERSION = "1.0"
MD_MAX_LINES = 300
STR_CUT = 200  # every string in a sample record is cut to this many characters
LIST_CUT_JSON = 20  # lists in JSON samples keep this many items (plus a marker)
LIST_CUT_MD = 5  # lists in Markdown samples keep this many items
MD_SAMPLE_CHARS = 1500  # a Markdown sample line is cut to this length (the JSON keeps more)

# --------------------------------------------------------------------------- names

# (regex on the normalized folder name, label, is query-time agent in the paper's Table 2)
METHOD_ALIASES = [
    (r"a_?rag", "A-RAG", True),
    (r"rag_?r1", "RAG-R1", True),
    (r"agent(ic)?_?r", "Agentic-R", True),
    (r"grip", "GRIP", True),
    (r"uma", "UMA", True),
    (r"raptor", "RAPTOR", False),
    (r"hippo_?rag(_?v?2)?", "HippoRAG v2", False),
    (r"llama_?index", "LlamaIndex", False),
    (r"lqca", "LQCA", False),
    (r"late_?chunk(ing)?", "Late Chunking", False),
    (r"vanilla", "Vanilla", False),
    (r"s?aadi|sadi", "SAADI", False),
]

ID_KEYS = [
    "query_id",
    "qid",
    "question_id",
    "example_id",
    "canonical_query_id",
    "sample_id",
    "query_uid",
    "uid",
    "id",
    "_id",
    "query_index",
    "idx",
]
QUESTION_KEYS = ["question", "query", "question_text", "query_text", "input"]
DATASET_KEYS = ["dataset", "dataset_name", "task", "benchmark", "corpus"]
RETRIEVER_KEYS = [
    "retriever",
    "retriever_name",
    "retrieval_backend",
    "retrieval_method",
    "backend",
    "embedding_model",
    "embedder",
    "embed_model",
    "search_backend",
    "index_type",
    "encoder",
]

PRUNE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "site-packages",
    ".ipynb_checkpoints",
    ".idea",
    ".vscode",
}

CONFIG_EXTS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".slurm", ".sbatch"}
CONFIG_NAME_RE = re.compile(
    r"(?i)(config|conf\b|manifest|args|arguments|settings|params|hparams|hyperparam"
    r"|run_info|run_meta|metadata|setup|leaderboard_row|command|cmdline)"
)
CONFIG_KEY_RE = re.compile(
    r"(?i)^(model|model_name|model_id|model_path|llm|llm_model|llm_name|generator|generator_model|reader"
    r"|reader_model|agent_model|policy_model|backbone|embedding_model|embedder|embed_model|encoder"
    r"|retriever|retriever_name|retrieval_backend|retrieval_method|top_?k|topk|retrieval_top_?k|search_top_?k"
    r"|n_docs|num_docs|k_retrieve|chunk_size|chunk_tokens|chunk_overlap|chunk_size_tokens|max_steps"
    r"|max_turns|max_iter|max_iters|max_iterations|max_rounds|max_hops|max_searches|max_search_calls"
    r"|max_tool_calls|max_reads|max_read_chunks|max_chunks|temperature|top_p|max_new_tokens|max_tokens"
    r"|max_output_tokens|max_input_tokens|context_window|max_context_tokens|target_context_tokens"
    r"|token_budget|budget|seed|dataset|dataset_name|split|n_queries|max_queries|tokenizer)$"
)
CONFIG_KEY_ORDER = [
    "model",
    "model_name",
    "llm",
    "llm_model",
    "generator",
    "agent_model",
    "policy_model",
    "embedding_model",
    "embedder",
    "retriever",
    "retriever_name",
    "top_k",
    "topk",
    "chunk_size",
    "chunk_tokens",
    "max_steps",
    "max_turns",
    "max_iterations",
    "max_rounds",
    "max_tool_calls",
    "max_reads",
    "max_new_tokens",
    "max_tokens",
    "temperature",
    "target_context_tokens",
    "dataset",
    "seed",
]

# ------------------------------------------------------------------------- secrets

SECRET_FILE_RE = re.compile(
    r"(?i)(^\.env$|^\.env\.|\.env$|^_?\.?netrc$|^\.pgpass$|^\.git-credentials$|\.pem$|\.key$|\.p12$"
    r"|\.pfx$|\.kdbx$|^id_(rsa|dsa|ecdsa|ed25519)|credential|secret|^tokens?\.(json|txt)$|htpasswd)"
)
SECRET_KEY_RE = re.compile(
    r"(?i)(api[_\-]?key|apikey|secret|passw|credential|private[_\-]?key|access[_\-]?key"
    r"|auth[_\-]?(token|header|key)|authorization|bearer|cookie|session[_\-]?(id|key|token)"
    r"|(^|[_\-.])token$|^token$|[_\-]pat$|^pat$|webhook)"
)
SECRET_VALUE_RE = re.compile(
    r"(sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_\-]{30,}|xox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|[Bb]earer\s+[A-Za-z0-9._~+/\-]{8,}=*)"
)
URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s:@]+(:[^/\s@]*)?@")
ASSIGN_RE = re.compile(r"""(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)(?P<sep>["']?\s*[:=]\s*)(?P<q>["']?)(?P<val>[^\s"',;}]+)""")
FLAG_ARG_RE = re.compile(r"""(?P<key>--?[A-Za-z][A-Za-z0-9_.\-]*)(?P<sep>\s+)(?P<q>["']?)(?P<val>[^\s"']+)""")


def is_secret_key(key):
    return bool(SECRET_KEY_RE.search(str(key).lstrip("-")))


def redact_text(s):
    if not isinstance(s, str) or not s:
        return s
    s = URL_USERINFO_RE.sub(r"\1<redacted>@", s)
    s = SECRET_VALUE_RE.sub("<redacted>", s)

    def _sub(m):
        if is_secret_key(m.group("key")):
            return m.group("key") + m.group("sep") + m.group("q") + "<redacted>"
        return m.group(0)

    s = ASSIGN_RE.sub(_sub, s)
    return FLAG_ARG_RE.sub(_sub, s)


def cut(s, n=STR_CUT):
    s = str(s)
    return s if len(s) <= n else s[:n] + "...[+%d chars]" % (len(s) - n)


def sanitize(obj, list_cut=LIST_CUT_JSON, depth=0, str_cut=STR_CUT):
    """Redact secrets, cut strings to str_cut, cap list length. Returns a new object."""
    if depth > 12:
        return "<deeper levels omitted>"
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 200:
                out["..."] = "+%d more keys" % (len(obj) - 200)
                break
            if is_secret_key(k) and v is not None and not isinstance(v, bool):
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = sanitize(v, list_cut, depth + 1, str_cut)
        return out
    if isinstance(obj, list):
        items = [sanitize(v, list_cut, depth + 1, str_cut) for v in obj[:list_cut]]
        if len(obj) > list_cut:
            items.append("... (+%d more items, %d total)" % (len(obj) - list_cut, len(obj)))
        return items
    if isinstance(obj, str):
        return cut(redact_text(obj), str_cut) if str_cut else redact_text(obj)
    return obj


# ------------------------------------------------------------------------- helpers


def human(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1000.0


def now_iso():
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def md(s):
    return str(s).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def guess_method(folder_name):
    """Return (label, known, agentic, copy_suffix)."""
    raw = folder_name.strip()
    copy_suffix = ""
    m = re.match(r"^(.*?)(\s+\(?\d+\)?|\s*copy|_copy|_old|_bak)$", raw, re.I)
    if m and m.group(1):
        raw, copy_suffix = m.group(1), m.group(2).strip()
    n = re.sub(r"[\s\-.]+", "_", raw.lower())
    stripped = re.sub(r"^(s?aadi|sadi)_", "", n)
    for cand in (stripped, n):
        for pat, label, agentic in METHOD_ALIASES:
            if re.fullmatch(pat, cand):
                return label, True, agentic, copy_suffix
    for pat, label, agentic in METHOD_ALIASES[:-1]:
        if re.search(r"(^|_)" + pat + r"($|_)", stripped):
            return label, True, agentic, copy_suffix
    return folder_name, False, False, copy_suffix


def strip_gz(name):
    return name[:-3] if name.lower().endswith(".gz") else name


def file_type(name):
    n = name.lower()
    if n.endswith(".gz"):
        n = n[:-3]
    ext = os.path.splitext(n)[1]
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".pkl", ".pickle"):
        return "pkl"
    if ext in (".pt", ".pth"):
        return "pt"
    if ext in (".log", ".out", ".err") or n == "nohup.out":
        return "log"
    if ext == ".txt":
        return "txt"
    return "other"


def is_pruned_dir(name):
    n = name.lower()
    return n in PRUNE_DIRS or n.startswith("venv") or n.startswith(".venv") or n.endswith(".egg-info")


def is_secret_file(name):
    return bool(SECRET_FILE_RE.search(name))


DS_LABEL_ORDER = ["LooGLE", "MuSiQue", "QASPER", "NovelHopQA"]


def _setting(t, key):
    m = re.search(key + r"[_/ .]*(noexp|no_?expansion|\d+k)(?![a-z0-9])", t)
    s = m.group(1) if m else None
    if s is None:
        m2 = re.search(r"(?<![a-z0-9])(\d+k)_?expansion|expansion[_/]*(\d+k)(?![a-z0-9])", t)
        if m2:
            s = m2.group(1) or m2.group(2)
        elif re.search(r"no_?exp(ansion)?", t):
            s = "noexp"
        elif re.search(r"(normal|standard|default)", t):
            s = "std"
    if s is None:
        return " (setting ?)"
    if s.startswith("no"):
        return " (no expansion)"
    if s == "std":
        return " (standard)"
    return "-" + s.upper()


def datasets_from_text(text):
    t = str(text).lower().replace("-", "_")
    out = []
    if "loogle" in t:
        out.append("LooGLE")
    if "musique" in t:
        out.append("MuSiQue" + _setting(t, "musique"))
    if "qasper" in t:
        out.append("QASPER" + _setting(t, "qasper"))
    if "novelhop" in t:
        out.append("NovelHopQA")
    return out


def ds_sort_key(label):
    for i, base in enumerate(DS_LABEL_ORDER):
        if label.startswith(base):
            return (i, label)
    return (len(DS_LABEL_ORDER), label)


RETRIEVER_PATS = [
    ("BM25", r"bm25"),
    ("Contriever", r"contriever"),
    ("BGE-M3", r"bge[_\-]?m3"),
    ("Jina", r"jina"),
    ("E5", r"(?<![a-z0-9])e5(?![a-z0-9])"),
    ("SPLADE", r"splade"),
    ("ColBERT", r"colbert"),
    ("Qwen-embedding", r"qwen3?[_\-]?emb|(?<![a-z0-9])qwen(?![a-z0-9\-])"),
    ("Hybrid", r"(?<![a-z0-9])hybrid(?![a-z0-9])"),
    ("Dense", r"(?<![a-z0-9])dense(?![a-z0-9])"),
]


def retrievers_from_text(text):
    t = str(text).lower()
    return [label for label, pat in RETRIEVER_PATS if re.search(pat, t)]


def kind_of(rel):
    base = os.path.basename(rel).lower()
    b = re.sub(r"[0-9a-f]{8,}", "<h>", base)
    b = re.sub(r"(loogle|musique|qasper|novelhopqa|novelhop)", "<ds>", b)
    b = re.sub(r"(bm25|contriever|bge_m3|bge-m3|jina)", "<r>", b)
    b = re.sub(r"\d+", "<n>", b)
    return b


def open_bin(path):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")


def count_lines(path, max_bytes):
    """Stream-count newline-terminated records. Returns (lines, complete)."""
    n, read, last = 0, 0, b"\n"
    with open_bin(path) as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
            last = chunk[-1:]
            read += len(chunk)
            if read > max_bytes:
                return n, False
    if last != b"\n":
        n += 1
    return n, True


def iter_jsonl(path, max_line, max_bytes):
    """Yield (record or None, error or None). Never holds more than one line (<= max_line)."""
    read = 0
    with open_bin(path) as f:
        while True:
            line = f.readline(max_line + 1)
            if not line:
                return
            read += len(line)
            if len(line) > max_line and not line.endswith(b"\n"):
                while True:
                    rest = f.readline(1 << 20)
                    read += len(rest)
                    if not rest or rest.endswith(b"\n"):
                        break
                yield None, "line longer than %d bytes" % max_line
                continue
            s = line.strip()
            if s:
                try:
                    yield json.loads(s.decode("utf-8", "replace")), None
                except Exception:
                    yield None, "unparseable line"
            if read > max_bytes:
                yield None, "STOP"
                return


def read_head_tail(path, head, tail):
    with open_bin(path) as f:
        h = f.read(head)
        t = b""
        if not path.lower().endswith(".gz"):
            size = os.fstat(f.fileno()).st_size
            if size > head + tail:
                f.seek(size - tail)
                t = f.read(tail)
    return h.decode("utf-8", "replace"), t.decode("utf-8", "replace")


def typename(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        inner = sorted({typename(x).split("[")[0].split("{")[0] for x in v[:20]})
        return "list[%d%s]" % (len(v), (" " + "/".join(inner)) if inner else "")
    if isinstance(v, dict):
        return "dict{%d}" % len(v)
    return type(v).__name__


def flatten(obj, max_depth=3, prefix="", depth=1, out=None):
    """path -> value, dict levels up to max_depth; lists of dicts are entered via their first dict."""
    if out is None:
        out = collections.OrderedDict()
    if not isinstance(obj, dict):
        return out
    items = list(obj.items())
    if len(items) > 50 and all(isinstance(v, dict) for _, v in items[:50]):
        items = [("{*}", items[0][1])]  # id-keyed mapping: show one representative
    for k, v in items[:300]:
        p = "%s.%s" % (prefix, k) if prefix else str(k)
        if p not in out:
            out[p] = v
        if depth < max_depth:
            if isinstance(v, dict):
                flatten(v, max_depth, p, depth + 1, out)
            elif isinstance(v, list):
                first = next((x for x in v[:50] if isinstance(x, dict)), None)
                if first is not None:
                    flatten(first, max_depth, p + "[]", depth + 1, out)
    return out


def leaf_of(path):
    parts = [p for p in re.split(r"[.\[\]]+", path) if p]
    return parts[-1] if parts else path


# --------------------------------------------------------------------- flag rules

FLAG_NAMES = ["retrieved_ids", "search_calls", "tokens", "previews", "answer", "gold"]
FLAG_TITLES = {
    "retrieved_ids": "retrieved chunk ids / passages (ordered)",
    "search_calls": "search calls / sub-queries",
    "tokens": "token counts",
    "previews": "search previews / snippets",
    "answer": "final answer",
    "gold": "gold evidence / hit flag",
}
RE_RETR = re.compile(
    r"(?i)((retriev|ordered|exposed|read|source|context|selected|top|ranked|returned|cited|visited|admitted"
    r"|used|opened|fetched|candidate|final)\w*?(chunk|passage|doc|node|para|sentence|evidence)s?"
    r"(_?(ids?|indices|index|list|keys))?$|^(chunk|passage|doc|node)_?(ids|indices)$"
    r"|^(passages|contexts|ctxs|retrieved|retrieval|retrievals|documents|chunks|retrieved_docs"
    r"|retrieved_passages|retrieved_chunks|ranking|rankings|ranked_list|top_k_ids|topk)$|retrieved_indices)"
)
RE_NOT_RETR = re.compile(r"(?i)(gold|silver|label|reference|expected|count|num_|n_|_len)")
RE_SEARCH_LIST = re.compile(
    r"(?i)(search|sub_?quer|subq|tool_?calls?|trajector|^steps?$|^actions?$|^turns?$|^rounds?$|^iterations?$"
    r"|rewrit|reformul|^queries$|query_list|generated_quer|^plan$|^thoughts?$|reasoning|^history$|^messages$"
    r"|conversation|^calls$|retrieval_calls|tool_trace|^trace$|^transcript$)"
)
RE_SEARCH_COUNT_A = re.compile(r"(?i)(search|retriev|tool|call|step|turn|round|iteration|sub_?quer)")
RE_SEARCH_COUNT_B = re.compile(r"(?i)(^n_|^num|count|calls$|_calls|steps$|turns$|rounds$|iterations$)")
TOOL_FIELD_LEAVES = {"tool", "tool_name", "action", "function_name", "function", "tool_call", "name"}
RE_PREVIEW = re.compile(
    r"(?i)(snippet|preview|search_result|search_output|search_hits|observation|tool_output|tool_result"
    r"|tool_response|tool_message|matched|highlights?)"
)
PREVIEW_CONTENT_RE = re.compile(
    r"(Chunk ID:\s*\S+.{0,40}\((score|similarity)=|Matched:\s*\.\.\.|no exact sentence snippet|\bsnippets?\b)", re.S
)
RE_ANSWER = re.compile(
    r"(?i)^(prediction|predictions|pred|preds|predicted|predicted_answer|generated_answer|generated|final_answer"
    r"|answer|model_answer|response|output|output_text|generation|completion|final_output|answer_text"
    r"|llm_answer)$"
)
RE_NOT_ANSWER = re.compile(r"(?i)(reference|gold|ground|expected|target|label|silver|tokens?)")
RE_GOLD = re.compile(r"(?i)(gold|silver|hit|evidence|supporting|relevan|recall|support_idx|is_correct|answer_containment)")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flag_hits(flat):
    """flat: path -> value. Returns {flag: [(path, typename, note)]}."""
    hits = {f: [] for f in FLAG_NAMES}
    for path, v in flat.items():
        leaf = leaf_of(path)
        if is_secret_key(leaf):
            continue
        tn = typename(v)
        if isinstance(v, list) and RE_RETR.search(leaf) and not RE_NOT_RETR.search(leaf):
            hits["retrieved_ids"].append((path, tn, ""))
        if isinstance(v, (list, dict)) and RE_SEARCH_LIST.search(leaf):
            hits["search_calls"].append((path, tn, ""))
        elif "[]" in path and leaf.lower() in TOOL_FIELD_LEAVES:
            hits["search_calls"].append((path, tn, "tool field inside a list"))
        elif _num(v) and RE_SEARCH_COUNT_A.search(leaf) and RE_SEARCH_COUNT_B.search(leaf):
            hits["search_calls"].append((path, tn, "count only"))
        if "token" in leaf.lower() and (_num(v) or isinstance(v, (list, dict))):
            hits["tokens"].append((path, tn, ""))
        elif leaf.lower() == "usage" and isinstance(v, dict):
            hits["tokens"].append((path, tn, "usage dict"))
        if RE_PREVIEW.search(leaf):
            hits["previews"].append((path, tn, "token count only" if _num(v) else ""))
        elif isinstance(v, str) and PREVIEW_CONTENT_RE.search(v[:5000]):
            hits["previews"].append((path, tn, "preview text inside value"))
        if (
            RE_ANSWER.search(leaf)
            and not RE_NOT_ANSWER.search(path)
            and (isinstance(v, str) or (isinstance(v, list) and v and isinstance(v[0], str)))
        ):
            hits["answer"].append((path, tn, ""))
        if RE_GOLD.search(leaf) and "reference" not in leaf.lower():
            hits["gold"].append((path, tn, ""))
    return hits


def config_from_flat(flat, add, source):
    for path, v in flat.items():
        leaf = leaf_of(path)
        if CONFIG_KEY_RE.match(leaf) and not is_secret_key(leaf) and (isinstance(v, (str, int, float, bool)) or v is None):
            add(leaf, v, source)


# ------------------------------------------------------------------------- git


def git_run(args, cwd, timeout=120):
    env = dict(os.environ)
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "GIT_PAGER": "cat", "PAGER": "cat"})
    try:
        p = subprocess.Popen(["git"] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err.decode("utf-8", "replace").strip()
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return 124, b"", "timeout after %ds" % timeout
    except Exception as e:  # noqa
        return 1, b"", "%s: %s" % (type(e).__name__, e)


def _lines(b, z=False):
    if z:
        return [x for x in b.decode("utf-8", "replace").split("\0") if x]
    return [x for x in b.decode("utf-8", "replace").splitlines() if x.strip()]


def git_info(path):
    info = {"is_repo": False}
    if not shutil.which("git"):
        info["error"] = "git not installed"
        return info, None
    rc, out, err = git_run(["rev-parse", "--show-toplevel"], path)
    if rc != 0:
        info["error"] = cut(redact_text(err) or "not a git repository", 300)
        if "dubious ownership" in err:
            info["hint"] = "git refused (dubious ownership); the inventory does not change git config"
        return info, None
    top = _lines(out)[0]
    info["is_repo"] = True
    info["toplevel"] = top
    info["own_repo"] = os.path.realpath(top) == os.path.realpath(path)
    rc, out, _ = git_run(["remote", "-v"], path)
    remotes = collections.OrderedDict()
    for line in _lines(out):
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in remotes:
            remotes[parts[0]] = redact_text(parts[1])
    info["remotes"] = remotes
    rc, out, _ = git_run(["symbolic-ref", "--short", "-q", "HEAD"], path)
    info["branch"] = _lines(out)[0] if rc == 0 and _lines(out) else "(detached or unknown)"
    rc, out, _ = git_run(["log", "-1", "--format=%H%x1f%cI%x1f%s"], path)
    if rc == 0 and _lines(out):
        h, d, s = (_lines(out)[0].split("\x1f") + ["", "", ""])[:3]
        info.update({"head": h, "head_date": d, "head_subject": cut(redact_text(s), 100)})
    else:
        info["head"] = "(no commits)"
    rc, out, _ = git_run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], path)
    if rc == 0 and _lines(out):
        info["upstream"] = _lines(out)[0]
        rc, out, _ = git_run(["rev-list", "--left-right", "--count", "@{u}...HEAD"], path)
        if rc == 0 and _lines(out):
            behind, ahead = (_lines(out)[0].split() + ["?", "?"])[:2]
            info["behind"], info["ahead"] = behind, ahead
        info["upstream_note"] = "ahead/behind as of the last fetch (no network used)"
    else:
        info["upstream"] = None
    status = {}
    rc, out, err = git_run(["ls-files", "-z", "--", "."], path, timeout=300)
    status["tracked"] = set(_lines(out, True)) if rc == 0 else None
    rc, out, err = git_run(["ls-files", "-z", "-o", "--exclude-standard", "--directory", "--", "."], path, 300)
    status["untracked"] = set(_lines(out, True)) if rc == 0 else None
    rc, out, err = git_run(["ls-files", "-z", "-o", "-i", "--exclude-standard", "--directory", "--", "."], path, 300)
    status["ignored"] = set(_lines(out, True)) if rc == 0 else None
    rc, out, err = git_run(["diff", "--name-only", "--relative", "-z", "--", "."], path, 300)
    status["modified"] = set(_lines(out, True)) if rc == 0 else None
    return info, status


def git_class(rel, status):
    if status is None:
        return "no-git"
    tracked = status.get("tracked")
    if tracked is not None and rel in tracked:
        mod = status.get("modified")
        return "tracked-modified" if mod and rel in mod else "tracked"
    parts = rel.split("/")
    prefixes = ["/".join(parts[:i]) + "/" for i in range(1, len(parts))]
    for key in ("ignored", "untracked"):
        s = status.get(key)
        if s is not None and (rel in s or any(p in s for p in prefixes)):
            return key
    return "unknown"


# ---------------------------------------------------------------------- scanning


class Budget:
    def __init__(self, seconds):
        self.deadline = time.time() + seconds

    def left(self):
        return time.time() < self.deadline


def walk_method(path, prune, errors):
    files, pruned, symlinks = [], [], 0

    def onerror(e):
        errors.append("walk: %s" % cut(e, 300))

    for dirpath, dirnames, filenames in os.walk(path, onerror=onerror, followlinks=False):
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                symlinks += 1
                continue
            if prune and is_pruned_dir(d):
                pruned.append(os.path.relpath(full, path).replace(os.sep, "/"))
                continue
            keep.append(d)
        dirnames[:] = sorted(keep)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            try:
                st = os.lstat(full)
            except OSError as e:
                errors.append("stat %s: %s" % (fn, e))
                continue
            if stat.S_ISLNK(st.st_mode):
                symlinks += 1
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            files.append(
                {
                    "rel": rel,
                    "full": full,
                    "size": st.st_size,
                    "type": file_type(fn),
                    "secret": is_secret_file(fn),
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
                }
            )
    return files, pruned, symlinks


def extract_records(obj):
    """Return (records, locator, mode). mode: 'list' | 'mapping' | None."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return obj, "$[]", "list"
        return None, "$ (list of %s)" % (typename(obj[0]) if obj else "nothing"), None
    if isinstance(obj, dict):
        if detect_id_key(obj) or has_question(obj):
            return None, "$ (one record)", None  # e.g. one file per query
        vals = list(obj.values())
        if len(vals) >= 20 and all(isinstance(v, dict) for v in vals[:50]) and all(isinstance(v, dict) for v in vals):
            sample_keys = list(obj.keys())[:20]
            if not all(CONFIG_KEY_RE.match(str(k)) for k in sample_keys):
                return [dict(v, **{"<mapping key>": k}) for k, v in list(obj.items())], "$.{id}", "mapping"
        best, best_key = None, None
        for k, v in obj.items():
            if isinstance(v, list) and len(v) >= 2 and isinstance(v[0], dict):
                if best is None or len(v) > len(best):
                    best, best_key = v, k
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, list) and len(v2) >= 2 and isinstance(v2[0], dict):
                        if best is None or len(v2) > len(best):
                            best, best_key = v2, "%s.%s" % (k, k2)
        if best is not None:
            return best, "$.%s[]" % best_key, "list"
    return None, "$", None


def record_id(rec, id_key):
    if id_key is None:
        return None
    v = rec.get(id_key)
    return v if isinstance(v, (str, int)) else None


def detect_id_key(rec):
    if "<mapping key>" in rec:
        return "<mapping key>"
    for k in ID_KEYS:
        if isinstance(rec.get(k), (str, int)) and not isinstance(rec.get(k), bool):
            return k
    return None


def has_question(rec):
    return any(isinstance(rec.get(k), str) and rec.get(k) for k in QUESTION_KEYS)


def load_records_sample(f, args, n=5):
    """Return dict with records (list), locator, mode, n_records, note, head_keys."""
    res = {"records": [], "locator": None, "mode": None, "n_records": None, "note": ""}
    if f["type"] == "jsonl":
        bad = 0
        for rec, err in iter_jsonl(f["full"], args.max_line_bytes, args.max_sample_bytes):
            if err == "STOP":
                break
            if err:
                bad += 1
                if bad > 50:
                    break
                continue
            if isinstance(rec, dict):
                res["records"].append(rec)
            else:
                res["records"].append({"<value>": rec})
            if len(res["records"]) >= n:
                break
        res["locator"], res["mode"] = "line", "list"
        if bad:
            res["note"] = "%d unreadable/too-long lines skipped in sample" % bad
    elif f["type"] == "json":
        if f["size"] <= args.max_json_mb * 1e6:
            with open_bin(f["full"]) as fh:
                obj = json.loads(fh.read().decode("utf-8", "replace"))
            recs, loc, mode = extract_records(obj)
            res["locator"], res["mode"] = loc, mode
            if recs is not None:
                res["records"] = recs[:n]
                res["n_records"] = len(recs)
                res["_all"] = recs
            else:
                res["records"] = [obj if isinstance(obj, dict) else {"<value>": obj}]
        else:
            head, _ = read_head_tail(f["full"], 256 * 1024, 0)
            keys = []
            for k in re.findall(r'"([^"\\]{1,80})"\s*:', head):
                if k not in keys:
                    keys.append(k)
            res["head_keys"] = keys[:80]
            res["note"] = "larger than --max-json-mb; keys guessed from the first 256 KB"
    elif f["type"] == "csv":
        delim = "\t" if strip_gz(f["rel"].lower()).endswith(".tsv") else ","
        with open_bin(f["full"]) as fh:
            text = (line.decode("utf-8", "replace") for line in fh)
            reader = csv.reader(text, delimiter=delim)
            header = next(reader, None)
            if header:
                for row in reader:
                    res["records"].append(collections.OrderedDict(zip(header, row)))
                    if len(res["records"]) >= n:
                        break
            res["header"] = header
        res["locator"], res["mode"] = "row", "list"
    return res


def scan_text_file(f, args, add_config, mentions):
    head, tail = read_head_tail(f["full"], args.log_head_bytes, args.log_tail_bytes)
    ext = os.path.splitext(strip_gz(f["rel"].lower()))[1]
    for part, where in ((head, "head"), (tail, "tail")):
        for i, line in enumerate(part.splitlines()):
            if len(line) > 4000:
                line = line[:4000]
            if ext in CONFIG_EXTS or f["type"] in ("log", "txt"):
                for m in re.finditer(r"--([A-Za-z][\w\-]*)(?:=|\s+)(\"[^\"]*\"|'[^']*'|[^\s\\]+)", line):
                    key = m.group(1).replace("-", "_")
                    if CONFIG_KEY_RE.match(key) and not is_secret_key(key):
                        add_config(key, m.group(2).strip("\"'"), f["rel"])
                m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][\w.\-]*)\s*[:=]\s*(.+?)\s*$", line)
                if m:
                    key = m.group(1).split(".")[-1].lower()
                    val = re.sub(r"\s+#.*$", "", m.group(2)).strip("\"' ,")
                    if CONFIG_KEY_RE.match(key) and not is_secret_key(key) and val:
                        add_config(key, val, f["rel"])
            if f["type"] in ("log", "txt"):
                if re.search(r"(?i)snippet|preview", line) and len(mentions["previews"]) < 5:
                    mentions["previews"].append("%s (%s line %d): %s" % (f["rel"], where, i + 1, cut(redact_text(line.strip()), 160)))
                if re.search(r"(?i)\btokens?\b", line) and len(mentions["tokens"]) < 3:
                    mentions["tokens"].append("%s (%s line %d): %s" % (f["rel"], where, i + 1, cut(redact_text(line.strip()), 160)))


def count_queries(f, kind, args):
    """Fill f['records'], f['distinct_ids'], f['content_datasets'] for a per-query file."""
    id_key = kind.get("id_key")
    ids, n, dsv = set(), 0, collections.Counter()
    complete = True

    def take(rec):
        rid = record_id(rec, id_key)
        if rid is not None:
            ids.add(rid)
        for k in DATASET_KEYS:
            v = rec.get(k)
            if isinstance(v, str):
                dsv[v[:60]] += 1
                break

    if f["type"] == "jsonl":
        if f["size"] > args.max_parse_mb * 1e6:
            return
        for rec, err in iter_jsonl(f["full"], args.max_line_bytes, args.max_parse_mb * 1e6):
            if err == "STOP":
                complete = False
                break
            if rec is None:
                continue
            n += 1
            if isinstance(rec, dict):
                take(rec)
    elif f["type"] == "json":
        if f["size"] > args.max_json_mb * 1e6:
            return
        with open_bin(f["full"]) as fh:
            obj = json.loads(fh.read().decode("utf-8", "replace"))
        recs, _, _ = extract_records(obj)
        if recs is None:
            return
        for rec in recs:
            n += 1
            take(rec)
    elif f["type"] == "csv":
        if f["size"] > args.max_parse_mb * 1e6:
            return
        with open_bin(f["full"]) as fh:
            reader = csv.DictReader(line.decode("utf-8", "replace") for line in fh)
            for rec in reader:
                n += 1
                take(rec)
    f["records"] = n
    f["distinct_ids"] = len(ids) if ids else None
    f["count_complete"] = complete
    if dsv:
        f["content_datasets"] = sorted({d for v in dsv for d in datasets_from_text(v)})


def scan_method(path, folder, args, budget):
    label, known, agentic, copy_suffix = guess_method(folder)
    if getattr(args, "method", None):
        forced = guess_method(args.method)
        label, known, agentic = (forced[0] if forced[1] else args.method), True, forced[2]
    m = collections.OrderedDict()
    m["name"], m["folder"], m["path"] = label, folder, path
    m["known_method"], m["agentic_in_paper_table"], m["copy_suffix"] = known, agentic, copy_suffix
    errors = []
    m["errors"] = errors
    sys.stderr.write("[inventory] method %s (%s)\n" % (label, path))

    files, pruned, symlinks = walk_method(path, not args.no_prune, errors)
    m["empty"] = not files
    m["pruned_dirs"] = pruned[:50]
    m["symlinks_skipped"] = symlinks

    # ---- git
    sys.stderr.write("[inventory]   git\n")
    ginfo, gstatus = git_info(path)
    m["git"] = ginfo
    for f in files:
        f["git"] = git_class(f["rel"], gstatus)
    gsum = collections.OrderedDict()
    for f in files:
        g = gsum.setdefault(f["git"], {"count": 0, "bytes": 0})
        g["count"] += 1
        g["bytes"] += f["size"]
    m["git_status_of_files"] = gsum

    # ---- file stats
    by_type = collections.OrderedDict((t, {"count": 0, "bytes": 0}) for t in ("jsonl", "json", "csv", "pkl", "pt", "log", "txt", "other"))
    other_ext = collections.Counter()
    lfs = 0
    for f in files:
        by_type[f["type"]]["count"] += 1
        by_type[f["type"]]["bytes"] += f["size"]
        if f["type"] == "other":
            other_ext[os.path.splitext(f["rel"].lower())[1] or "(none)"] += 1
        if f["size"] <= 1024 and not f["secret"]:
            try:
                with open(f["full"], "rb") as fh:
                    if fh.read(64).startswith(b"version https://git-lfs"):
                        f["lfs_pointer"] = True
                        lfs += 1
            except OSError:
                pass
    m["files"] = {
        "count": len(files),
        "bytes": sum(f["size"] for f in files),
        "by_type": by_type,
        "other_extensions": dict(other_ext.most_common(15)),
        "lfs_pointer_files": lfs,
        "secret_files_not_opened": [f["rel"] for f in files if f["secret"]][:30],
    }
    m["largest"] = [
        {"rel": f["rel"], "size": f["size"], "type": f["type"], "git": f["git"], "mtime": f["mtime"]}
        for f in sorted(files, key=lambda x: -x["size"])[:20]
    ]

    # ---- datasets / retrievers from paths
    for f in files:
        f["datasets"] = datasets_from_text(f["rel"])
    retr = collections.OrderedDict()
    for f in files:
        for r in retrievers_from_text(f["rel"]):
            e = retr.setdefault(r, {"path_files": 0, "example": f["rel"], "content_values": []})
            e["path_files"] += 1

    # ---- config
    config = collections.OrderedDict()

    def add_config(key, value, source):
        key = key.lower()
        val = cut(redact_text(str(value)), 120)
        e = config.setdefault(key, collections.OrderedDict())
        v = e.setdefault(val, {"count": 0, "sources": []})
        v["count"] += 1
        if len(v["sources"]) < 3 and source not in v["sources"]:
            v["sources"].append(source)

    mentions = {"previews": [], "tokens": []}

    # ---- kinds
    kinds = collections.OrderedDict()
    for f in files:
        if f["secret"]:
            continue
        k = (f["type"], kind_of(f["rel"]))
        kinds.setdefault(k, []).append(f)

    kind_reports = []
    method_flags = {fl: [] for fl in FLAG_NAMES}
    n_cfg = 0
    for (ftype, kname), kfiles in kinds.items():
        ext = os.path.splitext(strip_gz(kfiles[0]["rel"].lower()))[1]
        is_cfg = (ext in CONFIG_EXTS) or (ftype == "json" and CONFIG_NAME_RE.search(kname))
        if ftype not in ("jsonl", "json", "csv", "log", "txt") and not is_cfg:
            continue
        kr = collections.OrderedDict()
        kr["kind"], kr["type"], kr["files"] = kname, ftype, len(kfiles)
        kr["bytes"] = sum(f["size"] for f in kfiles)
        kr["example"] = kfiles[0]["rel"]
        kr["datasets"] = sorted({d for f in kfiles for d in f["datasets"]}, key=ds_sort_key)
        kr["git"] = dict(collections.Counter(f["git"] for f in kfiles))
        if ftype in ("log", "txt") or (ext in CONFIG_EXTS):
            for f in kfiles[:2] + kfiles[-1:] if len(kfiles) > 2 else kfiles:
                if n_cfg >= args.max_config_files or f["size"] > 64e6 and ext in CONFIG_EXTS:
                    continue
                n_cfg += 1
                try:
                    scan_text_file(f, args, add_config, mentions)
                except Exception as e:
                    errors.append("text %s: %s" % (f["rel"], cut(e, 200)))
            if ftype in ("log", "txt"):
                kind_reports.append(kr)
            continue

        # inspect the first and the last file of this kind
        inspect = [kfiles[0]] if len(kfiles) == 1 else [kfiles[0], kfiles[-1]]
        samples, union_flat, first_flat, notes, n_records, locator, mode = [], collections.OrderedDict(), None, [], None, None, None
        id_key, question = None, False
        for f in inspect:
            try:
                res = load_records_sample(f, args)
            except Exception as e:
                errors.append("read %s: %s: %s" % (f["rel"], type(e).__name__, cut(e, 200)))
                continue
            if res.get("note"):
                notes.append("%s: %s" % (f["rel"], res["note"]))
            if res.get("head_keys") and not res["records"]:
                kr.setdefault("head_keys", res["head_keys"])
            locator, mode = res["locator"], res["mode"]
            if f is inspect[0]:
                n_records = res.get("n_records")
                if f["type"] == "csv" and res.get("header"):
                    kr["csv_header"] = [cut(redact_text(h), 80) for h in res["header"][:80]]
            for rec in res["records"]:
                if not isinstance(rec, dict):
                    continue
                flat = flatten(rec, 3)
                if first_flat is None:
                    first_flat = flat
                    samples.append(rec)
                    id_key = detect_id_key(rec)
                    question = has_question(rec)
                for p, v in flat.items():
                    union_flat.setdefault(p, v)
            if is_cfg or CONFIG_NAME_RE.search(kname):
                for rec in res["records"][:1]:
                    if isinstance(rec, dict):
                        config_from_flat(flatten(rec, 5), add_config, f["rel"])
            for rec in res["records"][:3]:
                if isinstance(rec, dict):
                    for k in RETRIEVER_KEYS:
                        v = rec.get(k)
                        if isinstance(v, str) and v:
                            for r in retrievers_from_text(v) or [cut(v, 40)]:
                                e = retr.setdefault(r, {"path_files": 0, "example": f["rel"], "content_values": []})
                                if len(e["content_values"]) < 3 and cut(v, 60) not in e["content_values"]:
                                    e["content_values"].append(cut(redact_text(v), 60))
            res.pop("_all", None)
        if ftype == "jsonl" and kfiles:
            try:
                n_records, _ = count_lines(kfiles[0]["full"], args.max_parse_mb * 1e6)
            except Exception as e:
                errors.append("count %s: %s" % (kfiles[0]["rel"], cut(e, 200)))
        per_query_mode = None
        if first_flat is not None and (id_key or question):
            if mode in ("list", "mapping") and (n_records or 0) >= 2:
                per_query_mode = "records"
            elif len(kfiles) >= 5:
                per_query_mode = "file-per-query"
        kr["record_locator"], kr["records_in_example"] = locator, n_records
        kr["per_query"] = per_query_mode
        kr["id_key"] = id_key
        if first_flat is not None:
            kr["keys_first_record"] = ["%s: %s" % (p, typename(v)) for p, v in first_flat.items()][:250]
            extra = [p for p in union_flat if p not in first_flat]
            if extra:
                kr["extra_keys_in_other_samples"] = extra[:80]
            kr["sample_record"] = sanitize(samples[0])
            hits = flag_hits(union_flat)
            kr["flags"] = {fl: [{"key": p, "type": t, "note": n} for p, t, n in hits[fl][:10]] for fl in FLAG_NAMES if hits[fl]}
            scope = "per-query" if per_query_mode else "summary"
            for fl in FLAG_NAMES:
                for p, t, n in hits[fl][:10]:
                    method_flags[fl].append(
                        {"kind": kname, "file": kr["example"], "key": p, "type": t, "note": n, "scope": scope, "datasets": kr["datasets"]}
                    )
        if notes:
            kr["notes"] = notes[:5]
        kr["_files"] = kfiles
        kind_reports.append(kr)

    # ---- query counting for per-query kinds (bounded by time budget)
    sys.stderr.write("[inventory]   counting queries\n")
    skipped = 0
    for kr in kind_reports:
        if kr.get("per_query") != "records":
            continue
        for f in kr["_files"][: args.max_count_files_per_kind]:
            if not budget.left():
                skipped += 1
                continue
            try:
                count_queries(f, kr, args)
            except Exception as e:
                errors.append("count %s: %s: %s" % (f["rel"], type(e).__name__, cut(e, 200)))
        if len(kr["_files"]) > args.max_count_files_per_kind:
            skipped += len(kr["_files"]) - args.max_count_files_per_kind
    if skipped:
        errors.append("query counting skipped for %d files (time budget or --max-count-files-per-kind)" % skipped)

    # ---- coverage per dataset
    cov = collections.OrderedDict()
    for f in files:
        dss = f["datasets"] or f.get("content_datasets") or []
        f["datasets_final"] = dss
        for d in dss:
            c = cov.setdefault(
                d, {"files": 0, "bytes": 0, "per_query_files": 0, "max_queries": None, "max_queries_file": None, "per_query_dirs": set()}
            )
            c["files"] += 1
            c["bytes"] += f["size"]
    for kr in kind_reports:
        if not kr.get("per_query"):
            continue
        if kr["per_query"] == "file-per-query":
            per_dir = collections.Counter()
            for f in kr["_files"]:
                for d in f["datasets_final"] or ["(no dataset in path)"]:
                    per_dir[(d, os.path.dirname(f["rel"]))] += 1
            for (d, dname), n in per_dir.items():
                c = cov.setdefault(
                    d,
                    {"files": 0, "bytes": 0, "per_query_files": 0, "max_queries": None, "max_queries_file": None, "per_query_dirs": set()},
                )
                c["per_query_files"] += n
                c["per_query_dirs"].add(dname)
                if c["max_queries"] is None or n > c["max_queries"]:
                    c["max_queries"], c["max_queries_file"] = n, dname + "/ (one file per query)"
            continue
        for f in kr["_files"]:
            for d in f["datasets_final"] or ["(no dataset in path)"]:
                c = cov.setdefault(
                    d,
                    {"files": 0, "bytes": 0, "per_query_files": 0, "max_queries": None, "max_queries_file": None, "per_query_dirs": set()},
                )
                c["per_query_files"] += 1
                c["per_query_dirs"].add(os.path.dirname(f["rel"]))
                q = f.get("distinct_ids") or f.get("records")
                if q is not None and (c["max_queries"] is None or q > c["max_queries"]):
                    c["max_queries"], c["max_queries_file"] = q, f["rel"]
    for d, c in cov.items():
        c["per_query_dirs"] = len(c["per_query_dirs"])
    m["coverage"] = collections.OrderedDict(sorted(cov.items(), key=lambda kv: ds_sort_key(kv[0])))
    m["retrievers"] = retr

    # ---- summary marks per dataset
    marks = collections.OrderedDict()
    for kr in kind_reports:
        if "flags" not in kr:
            continue
        dss = sorted({d for f in kr["_files"] for d in (f["datasets_final"] or ["(no dataset in path)"])}, key=ds_sort_key)
        level = "Y" if kr.get("per_query") else "s"
        fl = kr["flags"]
        found = {
            "trace": bool(fl.get("retrieved_ids") or [h for h in fl.get("search_calls", []) if h["note"] != "count only"]),
            "tokens": bool(fl.get("tokens")),
            "previews": bool(fl.get("previews")),
            "answer": bool(fl.get("answer")),
        }
        for d in dss:
            row = marks.setdefault(d, {"trace": "-", "tokens": "-", "previews": "-", "answer": "-"})
            for col, ok in found.items():
                if ok and (row[col] == "-" or (row[col] == "s" and level == "Y")):
                    row[col] = level
    m["summary_marks"] = marks

    for kr in kind_reports:
        kr.pop("_files", None)
    kind_reports.sort(key=lambda k: (0 if k.get("per_query") else 1 if k["type"] in ("jsonl", "json", "csv") else 2, -k["bytes"]))
    m["kinds"] = kind_reports
    m["flags"] = method_flags
    m["log_mentions"] = mentions
    ordered_cfg = collections.OrderedDict()
    for k in CONFIG_KEY_ORDER + list(config.keys()):
        if k in config and k not in ordered_cfg:
            vals = sorted(config[k].items(), key=lambda kv: -kv[1]["count"])
            ordered_cfg[k] = [{"value": v, "count": e["count"], "sources": e["sources"]} for v, e in vals[:8]]
    m["config"] = ordered_cfg
    return m


def discover_methods(root, layout):
    base = os.path.basename(os.path.normpath(root))
    children = []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if os.path.isdir(full) and not os.path.islink(full) and not d.startswith(".") and not is_pruned_dir(d):
            children.append(d)
    if layout == "single":
        return [(root, base)], "single"
    if layout == "children":
        return [(os.path.join(root, c), c) for c in children], "children"
    if guess_method(base)[1]:
        return [(root, base)], "single (root name looks like a method)"
    if any(guess_method(c)[1] for c in children):
        return [(os.path.join(root, c), c) for c in children], "children (method-like subfolders)"
    if any(os.path.exists(os.path.join(root, x)) for x in ("pyproject.toml", "setup.py", "src", "outputs", ".git")):
        return [(root, base)], "single (root looks like one repository)"
    return [(os.path.join(root, c), c) for c in children], "children (default)"


# ---------------------------------------------------------------------- markdown

LEVELS = [
    {"largest": 20, "kinds": 8, "samples": 3, "keys": 60, "config": 10, "flag_hits": 3},
    {"largest": 10, "kinds": 6, "samples": 2, "keys": 50, "config": 8, "flag_hits": 2},
    {"largest": 5, "kinds": 4, "samples": 1, "keys": 40, "config": 6, "flag_hits": 2},
    {"largest": 3, "kinds": 3, "samples": 1, "keys": 30, "config": 4, "flag_hits": 1},
    {"largest": 0, "kinds": 2, "samples": 0, "keys": 25, "config": 3, "flag_hits": 1},
]


def compact_sample(rec):
    s = json.dumps(sanitize(rec, LIST_CUT_MD, str_cut=None), ensure_ascii=False, default=str)
    return cut(s, MD_SAMPLE_CHARS)


def render_md(rep, lv, json_name):
    L = []
    a = L.append
    a("# C6 inventory: %s / %s" % (rep["machine"], rep["root_name"]))
    a("")
    a(
        "- Root `%s` (given as `%s`), host `%s`, layout: %s."
        % (md(rep["root_resolved"]), md(rep["root_arg"]), md(rep["hostname"]), rep.get("layout", "?"))
    )
    a(
        "- Scanned %s, %.0f s, status **%s**, tool v%s, Python %s. Read-only scan; full detail in `%s`."
        % (rep["started"], rep.get("duration_sec", 0), rep["status"], TOOL_VERSION, rep["python"], json_name)
    )
    a(
        "- Limits: JSON files > %d MB not parsed (keys from head only); per-query files > %d MB not counted; "
        "samples cut strings to %d chars; secrets redacted." % (rep["args"]["max_json_mb"], rep["args"]["max_parse_mb"], STR_CUT)
    )
    if lv is not LEVELS[0]:
        a(
            "- Markdown shortened to stay under %d lines (largest files: %d, kinds per method: %d); the JSON has "
            "everything." % (MD_MAX_LINES, lv["largest"], lv["kinds"])
        )
    if rep.get("loose_root_files"):
        lr = rep["loose_root_files"]
        a("- %d loose files directly under the root (%s), not assigned to a method." % (lr["count"], human(lr["bytes"])))
    if rep["status"] != "ok":
        a("")
        a("## Failure")
        a("```")
        for line in (rep.get("traceback") or rep.get("error") or "").splitlines()[-25:]:
            a(line)
        a("```")
    a("")
    a("## Summary: method x dataset")
    a("")
    a(
        "Y = found in per-query records, s = only in summary/aggregate files, - = not found. "
        "Trace = retrieved ids or search calls per query. Queries = max distinct ids (or records) in one per-query file."
    )
    a("")
    a("| Method | Folder | Dataset | Per-query files | Queries | Trace | Tokens | Previews | Answer |")
    a("|---|---|---|---|---|---|---|---|---|")
    for m in rep["methods"]:
        rows = m.get("summary_marks") or {}
        cov = m.get("coverage") or {}
        dss = sorted(set(rows) | {d for d, c in cov.items() if c.get("per_query_files")}, key=ds_sort_key)
        if not dss:
            a("| %s | %s | (none found) | 0 | | - | - | - | - |" % (md(m["name"]), md(m["folder"])))
            continue
        for d in dss:
            r = rows.get(d, {"trace": "-", "tokens": "-", "previews": "-", "answer": "-"})
            c = cov.get(d, {})
            a(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    md(m["name"]),
                    md(m["folder"]),
                    md(d),
                    c.get("per_query_files", 0),
                    c.get("max_queries") if c.get("max_queries") is not None else "",
                    r["trace"],
                    r["tokens"],
                    r["previews"],
                    r["answer"],
                )
            )
    for m in rep["methods"]:
        a("")
        a("## %s: `%s`" % (md(m["name"]), md(m["folder"])))
        if m.get("fatal"):
            a("- **Scan failed for this method:** %s" % md(m["fatal"]))
        fi = m.get("files", {})
        extra = []
        if m.get("copy_suffix"):
            extra.append("folder name has copy suffix '%s'" % m["copy_suffix"])
        if m.get("agentic_in_paper_table"):
            extra.append("query-time agent in paper Table 2")
        if not m.get("known_method"):
            extra.append("method name not recognized")
        a(
            "- Path `%s`; %d files, %s%s."
            % (md(m["path"]), fi.get("count", 0), human(fi.get("bytes", 0)), ("; " + "; ".join(extra)) if extra else "")
        )
        if m.get("empty"):
            a("- Folder is empty (no regular files).")
            g = m.get("git", {})
            if g.get("is_repo"):
                a("- Git: inside `%s`." % md(g.get("toplevel")))
            continue
        g = m.get("git", {})
        if g.get("is_repo"):
            rem = ", ".join("%s %s" % (k, v) for k, v in (g.get("remotes") or {}).items()) or "no remote"
            up = "no upstream"
            if g.get("upstream"):
                up = "upstream `%s`, ahead %s, behind %s (as of last fetch)" % (g["upstream"], g.get("ahead", "?"), g.get("behind", "?"))
            a(
                "- Git: %s `%s`; %s; branch `%s`; HEAD `%s` %s; %s."
                % (
                    "own repo" if g.get("own_repo") else "inside repo",
                    md(g.get("toplevel")),
                    md(rem),
                    md(g.get("branch")),
                    str(g.get("head", ""))[:10],
                    g.get("head_date", ""),
                    up,
                )
            )
        else:
            a("- Git: not a git repo (%s)." % md(g.get("error", "")))
        gs = m.get("git_status_of_files") or {}
        if gs:
            a("- Git status of scanned files: " + "; ".join("%s %d (%s)" % (k, v["count"], human(v["bytes"])) for k, v in gs.items()) + ".")
        pq = collections.Counter()
        for k in m.get("kinds", []):
            if k.get("per_query"):
                pq.update(k.get("git", {}))
        if pq:
            a("- Git status of per-query files: " + "; ".join("%s %d" % kv for kv in pq.items()) + ".")
        bt = fi.get("by_type", {})
        a("- Types: " + "; ".join("%s %d (%s)" % (t, v["count"], human(v["bytes"])) for t, v in bt.items() if v["count"]) + ".")
        notes = []
        if fi.get("lfs_pointer_files"):
            notes.append("%d git-LFS pointer files (content not present)" % fi["lfs_pointer_files"])
        if fi.get("secret_files_not_opened"):
            notes.append("%d credential-like files not opened" % len(fi["secret_files_not_opened"]))
        if m.get("pruned_dirs"):
            notes.append("skipped dirs: " + ", ".join("`%s`" % md(p) for p in m["pruned_dirs"][:4]))
        if notes:
            a("- Notes: " + "; ".join(notes) + ".")
        if lv["largest"] and m.get("largest"):
            a("")
            a("| Largest files | Size | Git |")
            a("|---|---|---|")
            for f in m["largest"][: lv["largest"]]:
                a("| `%s` | %s | %s |" % (md(cut(f["rel"], 140)), human(f["size"]), f["git"]))
        cov = m.get("coverage") or {}
        a("")
        if cov:
            a("| Dataset | Files | Size | Per-query files | Run dirs | Queries (max per file) |")
            a("|---|---|---|---|---|---|")
            for d, c in cov.items():
                a(
                    "| %s | %d | %s | %d | %d | %s |"
                    % (
                        md(d),
                        c["files"],
                        human(c["bytes"]),
                        c["per_query_files"],
                        c["per_query_dirs"],
                        "" if c["max_queries"] is None else c["max_queries"],
                    )
                )
        else:
            a("- Coverage: no dataset name found in paths or records.")
        rt = m.get("retrievers") or {}
        if rt:
            a(
                "- Retrievers (guessed): "
                + "; ".join(
                    "%s (%d path files%s)"
                    % (k, v["path_files"], (", values " + "/".join(v["content_values"])) if v["content_values"] else "")
                    for k, v in rt.items()
                )
                + "."
            )
        a("- Per-query trace flags:")
        for fl in FLAG_NAMES:
            hits = m.get("flags", {}).get(fl, [])
            pq_hits = [h for h in hits if h["scope"] == "per-query"]
            sm_hits = [h for h in hits if h["scope"] != "per-query"]
            seen, parts = set(), []
            for h in pq_hits:
                key = (h["kind"], h["key"])
                if key in seen:
                    continue
                seen.add(key)
                parts.append("`%s` -> `%s` (%s%s)" % (md(h["kind"]), md(h["key"]), md(h["type"]), (", " + h["note"]) if h["note"] else ""))
            txt = "; ".join(parts[: lv["flag_hits"]])
            if len(parts) > lv["flag_hits"]:
                txt += "; +%d more" % (len(parts) - lv["flag_hits"])
            if not parts:
                if sm_hits:
                    h = sm_hits[0]
                    txt = "summary only: `%s` -> `%s`" % (md(h["kind"]), md(h["key"]))
                else:
                    txt = "not found"
            if fl == "previews" and m.get("log_mentions", {}).get("previews") and not parts:
                txt += "; log mention: %s" % md(m["log_mentions"]["previews"][0])
            a("  - %s: %s" % (FLAG_TITLES[fl], txt))
        kinds = [k for k in m.get("kinds", []) if k["type"] in ("jsonl", "json", "csv")]
        if kinds:
            a("- Output kinds (%d data kinds; %d shown, per-query first):" % (len(kinds), min(len(kinds), lv["kinds"])))
            for i, k in enumerate(kinds[: lv["kinds"]]):
                desc = "%s, %d files, %s" % (k["type"], k["files"], human(k["bytes"]))
                if k.get("per_query"):
                    desc += ", per-query (%s), id `%s`" % (k["per_query"], k.get("id_key"))
                if k.get("records_in_example") is not None:
                    desc += ", %s records in `%s`" % (k["records_in_example"], md(cut(k["example"], 90)))
                else:
                    desc += ", e.g. `%s`" % md(cut(k["example"], 90))
                a("  - `%s` (%s)" % (md(k["kind"]), desc))
                keys = k.get("keys_first_record") or k.get("csv_header") or k.get("head_keys") or []
                if keys:
                    shown = ", ".join(md(x) for x in keys[: lv["keys"]])
                    more = " (+%d more)" % (len(keys) - lv["keys"]) if len(keys) > lv["keys"] else ""
                    a("    - keys: %s%s" % (shown, more))
                if i < lv["samples"] and k.get("sample_record") is not None and k.get("per_query"):
                    a("    - sample: `%s`" % compact_sample(k["sample_record"]).replace("`", "'"))
        cfg = m.get("config") or {}
        if cfg:
            parts = []
            for key, vals in list(cfg.items())[: lv["config"]]:
                parts.append("%s = %s" % (key, " / ".join("%s (x%d)" % (md(v["value"]), v["count"]) for v in vals[:3])))
            a("- Config seen: " + "; ".join(parts) + ".")
        if m.get("errors"):
            a("- Warnings (%d): %s" % (len(m["errors"]), md("; ".join(m["errors"][:3]))))
    if rep.get("warnings"):
        a("")
        a("## Warnings")
        for w in rep["warnings"][:10]:
            a("- %s" % md(w))
    return L


def write_reports(rep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = "inventory_%s_%s" % (rep["machine_safe"], rep["root_name"])
    json_path = os.path.join(out_dir, stem + ".json")
    md_path = os.path.join(out_dir, stem + ".md")
    lines = None
    for lv in LEVELS:
        try:
            lines = render_md(rep, lv, os.path.basename(json_path))
        except Exception:
            lines = [
                "# C6 inventory: %s / %s" % (rep.get("machine"), rep.get("root_name")),
                "",
                "Markdown rendering failed; see the JSON report.",
                "",
                "```",
                traceback.format_exc(),
                "```",
            ]
            rep.setdefault("warnings", []).append("markdown rendering failed: " + traceback.format_exc()[-300:])
            break
        if len(lines) <= MD_MAX_LINES:
            break
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, json_path)
    tmp = md_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, md_path)
    return md_path, json_path


# ------------------------------------------------------------------------ main


def scan_root(root_arg, args, out_dir, used_names):
    root = os.path.realpath(os.path.expanduser(root_arg))
    name = re.sub(r"[^A-Za-z0-9._\-]+", "_", os.path.basename(root.rstrip(os.sep)) or "root")
    if name in used_names:
        name = "%s_%d" % (name, len(used_names))
    used_names.add(name)
    rep = collections.OrderedDict()
    rep["tool"], rep["tool_version"] = "c6_inventory", TOOL_VERSION
    rep["machine"] = args.machine
    rep["machine_safe"] = re.sub(r"[^A-Za-z0-9._\-]+", "_", args.machine)
    rep["root_arg"], rep["root_resolved"], rep["root_name"] = root_arg, root, name
    rep["hostname"] = socket.gethostname()
    rep["python"] = sys.version.split()[0]
    rep["started"] = now_iso()
    rep["status"] = "running"
    rep["args"] = {k: v for k, v in vars(args).items() if k not in ("root",)}
    rep["methods"] = []
    rep["warnings"] = []
    t0 = time.time()
    try:
        if not os.path.isdir(root):
            raise RuntimeError("root is not a directory: %s" % root)
        methods, layout = discover_methods(root, args.layout)
        rep["layout"] = layout
        loose = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f)) and not os.path.islink(os.path.join(root, f))
        ]
        if loose and not layout.startswith("single"):
            rep["loose_root_files"] = {
                "count": len(loose),
                "bytes": sum(os.lstat(p).st_size for p in loose),
                "names": [os.path.basename(p) for p in loose][:30],
            }
        budget = Budget(args.time_budget_sec)
        for mpath, folder in methods:
            try:
                rep["methods"].append(scan_method(mpath, folder, args, budget))
            except Exception as e:
                rep["methods"].append(
                    {
                        "name": guess_method(folder)[0],
                        "folder": folder,
                        "path": mpath,
                        "fatal": "%s: %s" % (type(e).__name__, cut(e, 300)),
                        "traceback": traceback.format_exc(),
                        "errors": [],
                    }
                )
                rep["warnings"].append("method %s failed: %s" % (folder, cut(e, 200)))
        failed = [m for m in rep["methods"] if m.get("fatal")]
        rep["status"] = "partial" if failed else "ok"
    except Exception as e:
        rep["status"] = "failed"
        rep["error"] = "%s: %s" % (type(e).__name__, e)
        rep["traceback"] = traceback.format_exc()
    rep["finished"] = now_iso()
    rep["duration_sec"] = round(time.time() - t0, 1)
    try:
        md_path, json_path = write_reports(rep, out_dir)
        sys.stderr.write("[inventory] wrote %s and %s (status %s)\n" % (md_path, json_path, rep["status"]))
    except Exception:
        sys.stderr.write("[inventory] FAILED to write the report:\n%s\n" % traceback.format_exc())
        return False
    return rep["status"] == "ok"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="C6: read-only inventory of saved comparison-method outputs.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--machine", required=True, help="label for this machine, e.g. local or server1")
    p.add_argument("--root", action="append", required=True, help="folder to scan (repeatable); one report per root")
    p.add_argument("--out", required=True, help="folder for the reports (must not be inside a scanned root)")
    p.add_argument(
        "--layout",
        choices=("auto", "single", "children"),
        default="auto",
        help="root is one method folder (single) or contains method folders (children)",
    )
    p.add_argument(
        "--method", default=None, help="method name to report (overrides the guess from the folder name; use with --layout single)"
    )
    p.add_argument("--max-json-mb", type=int, default=64, help="parse .json files up to this size")
    p.add_argument("--max-parse-mb", type=int, default=1024, help="count queries in per-query files up to this size")
    p.add_argument("--max-sample-bytes", type=int, default=64 * 1024 * 1024, help="bytes read from a JSONL file to collect sample records")
    p.add_argument("--max-line-bytes", type=int, default=32 * 1024 * 1024, help="skip JSONL lines longer than this")
    p.add_argument("--log-head-bytes", type=int, default=1024 * 1024, help="bytes read from the head of a log/txt")
    p.add_argument("--log-tail-bytes", type=int, default=256 * 1024, help="bytes read from the tail of a log/txt")
    p.add_argument("--max-config-files", type=int, default=400, help="max text/config files scanned per method")
    p.add_argument("--max-count-files-per-kind", type=int, default=500, help="max files counted per kind")
    p.add_argument("--time-budget-sec", type=int, default=3600, help="stop query counting after this per root")
    p.add_argument("--no-prune", action="store_true", help="also walk venv/.git/__pycache__/node_modules dirs")
    args = p.parse_args(argv)

    out_dir = os.path.realpath(os.path.expanduser(args.out))
    for r in args.root:
        rr = os.path.realpath(os.path.expanduser(r))
        if out_dir == rr or out_dir.startswith(rr.rstrip(os.sep) + os.sep):
            sys.stderr.write(
                "[inventory] refusing: --out %s is inside the scanned root %s (read-only rule). "
                "Pass --out somewhere else.\n" % (out_dir, rr)
            )
            return 2
    ok, used = True, set()
    for r in args.root:
        ok = scan_root(r, args, out_dir, used) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("[inventory] interrupted\n")
        sys.exit(130)
