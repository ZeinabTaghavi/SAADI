#!/usr/bin/env python3
"""C4 steps 03 (BM25) and 04 (dense): per-query ranks inside the gold sub-document.

Only pairs that passed the reproduction gate of step 02 are measured. For each pair the
per-query ranks (full ranking, top-10 sub-document share, inside rank for set S and set R,
restriction keep and renorm, Vanilla and SAADI) go to work/perquery/<setting>__<retriever>.csv.gz;
the report shows the point estimates. Step 05 adds the paired bootstrap and the summary.

Before measuring, the full-ranking HR@5/HR@10 hit counts are recomputed and must equal the
counts step 02 gated on (the scores have not changed since).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from c4_common import (
    ALL_SETTINGS,
    REPORT_DIR,
    RETRIEVER_DISPLAY,
    StepReport,
    fnum,
    published_settings,
    read_json,
    rel,
    require_step_ok,
    run_step,
    selected_retrievers,
    selected_settings,
)
from c4_measures import SETS, compute_pair, load_perquery, perquery_path, summarize
from c4_streams import job_name, stream_paths

MAIN_SET = os.environ.get("C4_MAIN_SET", "R")


def _job(key: str, retriever: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
    except Exception as exc:
        return {"setting": key, "retriever": retriever, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def body(report: StepReport, args: argparse.Namespace, family: str) -> None:
    require_step_ok(1)
    s02 = require_step_ok(2)
    cfg = published_settings()
    if s02.get("published_settings") != cfg:
        raise RuntimeError("the config's published weights changed since step 02; rerun step 02")
    settings = selected_settings(args.settings)
    retrievers = [r for r in selected_retrievers(args.retrievers) if (r == "bm25") == (family == "bm25")]
    pairs02 = {f"{p['setting']}__{p['retriever']}": p for p in s02.get("pairs", [])}
    ready = set(s02.get("ready_pairs", []))

    todo, skipped = [], []
    for s in settings:
        for r in retrievers:
            name = job_name(s, r)
            if name not in pairs02:
                continue  # the paper has no saved ranking for this pair
            if name in ready:
                todo.append((s, r))
                continue
            sp, qp = stream_paths(s, r)
            st = pairs02[name]["status"]
            if st == "waiting_for_scores" and sp.exists() and qp.exists():
                skipped.append((s, r, "scores now present but not yet checked: rerun sh/02_check.sh"))
            elif st == "waiting_for_scores":
                skipped.append((s, r, "waiting for scores: " + ", ".join(f"`{f}`" for f in pairs02[name].get("missing_files", []))))
            else:
                skipped.append((s, r, f"step 02 status {st}"))
    report.data["main_set"] = MAIN_SET
    report.data["skipped"] = [{"pair": job_name(s, r), "reason": why} for s, r, why in skipped]

    results: dict[str, dict[str, Any]] = {}
    if todo:
        print(f"[c4] step {report.step:02d}: {len(todo)} pairs, {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(todo)))) as pool:
            futs = {pool.submit(_job, s.key, r, cfg): job_name(s, r) for s, r in todo}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()

    summary_rows: list[dict[str, Any]] = []
    for s, r in todo:
        name = job_name(s, r)
        res = results[name]
        if "error" in res:
            report.fail(f"{name}: {res['error']}")
            continue
        for p in res["problems"][:10]:
            report.fail(f"{name}: {p}")
        if len(res["problems"]) > 10:
            report.fail(f"{name}: ... {len(res['problems']) - 10} more problems")
        frame = load_perquery(perquery_path(s, r))
        # Same scores as gated in step 02: full-ranking hit counts on the paper's eligible queries.
        ref = pairs02[name]["hits"]
        for meth in ("pub", "van"):
            for k in (5, 10):
                ours = int((frame[f"full_rank_{meth}"] <= k).sum())
                if ours != int(ref[meth][str(k)]):
                    report.fail(f"{name}: {meth} HR@{k} hits {ours} != step 02 ({ref[meth][str(k)]}); scores changed, rerun step 02")
        if len(frame) != int(pairs02[name]["eligible"]):
            report.fail(f"{name}: {len(frame)} queries != step 02 eligible {pairs02[name]['eligible']}")
        for set_name in ("all",) + SETS:
            for meth in ("van", "pub"):
                summary_rows.append({"setting": s.key, "display": s.display, "retriever": r, "set": set_name, "method": meth,
                                     **summarize(frame, set_name, meth)})
        res["diag"]["queries"] = len(frame)
        report.wrote(perquery_path(s, r))
    report.data["pairs"] = {k: {kk: vv for kk, vv in v.items() if kk != "traceback"} for k, v in results.items()}
    report.data["summary"] = summary_rows

    # ---------------- report
    report.p(f"Pairs measured: {len(todo)}. Main query set: {MAIN_SET} (set with `C4_MAIN_SET`; both sets are always computed). "
             "Vanilla = the text channel alone; SAADI = the published fusion "
             f"(lambda_M={cfg['lambda_n']}, lambda_S={cfg['lambda_h']}, mu=({cfg['memory_name_weight']}, {cfg['memory_alias_weight']}, "
             f"{cfg['memory_description_weight']}), from `configs/methods/sadi.yaml`).")
    if skipped:
        report.h("Pairs not measured")
        report.table(["Setting", "Retriever", "Reason"], [[s.display, r, why] for s, r, why in skipped])
        for s, r, why in skipped:
            report.warn(f"{job_name(s, r)}: {why.split(':')[0]}")
    if not todo:
        report.p("No pair of this family is ready; nothing measured.")
        return

    def get(setting, retr, set_name, meth):
        for row in summary_rows:
            if (row["setting"], row["retriever"], row["set"], row["method"]) == (setting, retr, set_name, meth):
                return row
        return None

    report.h("1. Query sets")
    rows = []
    for s, r in todo:
        a, S_, R_ = (get(s.key, r, x, "pub") for x in ("all", "S", "R"))
        if a is None:
            continue
        d = results[job_name(s, r)]["diag"]
        rows.append([s.display, RETRIEVER_DISPLAY[r], a["n"], S_["n"], R_["n"], S_.get("n_gt10"), R_.get("n_gt10"),
                     d["S_relabel_differs_from_paper"], d["n_gold_chunks_mismatch"]])
    report.table(["Setting", "Retriever", "Eligible (paper)", "Set S", "Set R", "S with > 10 chunks", "R with > 10 chunks",
                  "S: inside relabel != paper labels", "Chunk-count mismatches"], rows)

    for set_name in (MAIN_SET,) + tuple(x for x in SETS if x != MAIN_SET):
        report.h(f"2{'a' if set_name == MAIN_SET else 'b'}. Set {set_name}: full ranking (A), sub-document (C), inside, keep scores (B)")
        rows = []
        for s, r in todo:
            for meth in ("van", "pub"):
                x = get(s.key, r, set_name, meth)
                if x is None:
                    continue
                rows.append([s.display, RETRIEVER_DISPLAY[r], {"van": "Vanilla", "pub": "SAADI"}[meth], x["n"],
                             fnum(x["full_hr5"]), fnum(x["full_hr10"]), fnum(x["full_median_rank"], 1),
                             fnum(x["subdoc_hit10"]), fnum(x["subdoc_share10"]),
                             fnum(x["keep_in_hr1"]), fnum(x["keep_in_hr3"]), fnum(x["keep_in_hr5"]), fnum(x["keep_in_hr10"]),
                             f"{fnum(x['keep_in_hr10_gt'])} (n={x['n_gt10']})", fnum(x["keep_in_mrr"]), fnum(x["keep_in_median_rank"], 1)])
        report.table(["Setting", "Retriever", "Method", "n", "Full HR@5", "Full HR@10", "Full median rank", "Sub-doc hit@10",
                      "Top-10 share from gold sub-doc", "In HR@1", "In HR@3", "In HR@5", "In HR@10", "In HR@10 (>10 chunks)",
                      "In MRR", "In median rank"], rows)

    report.h(f"3. Set {MAIN_SET}: HR@k on queries whose gold sub-document has more than k chunks (keep / renorm)")
    rows = []
    for s, r in todo:
        for meth in ("van", "pub"):
            x = get(s.key, r, MAIN_SET, meth)
            if x is None:
                continue
            rows.append([s.display, RETRIEVER_DISPLAY[r], {"van": "Vanilla", "pub": "SAADI"}[meth]]
                        + [f"{fnum(x[f'keep_in_hr{k}_gt'])} / {fnum(x[f'renorm_in_hr{k}_gt'])} (n={x[f'n_gt{k}']})" for k in (1, 3, 5, 10)]
                        + [f"{fnum(x['keep_in_mrr'])} / {fnum(x['renorm_in_mrr'])}"])
    report.table(["Setting", "Retriever", "Method", "HR@1 (>1)", "HR@3 (>3)", "HR@5 (>5)", "HR@10 (>10)", "MRR"], rows)
    report.p("keep = full-document scores, chunks outside the gold sub-document dropped (main). renorm = every channel "
             "z-scored again over the gold sub-document's chunks only. Vanilla's ranking is the same under both (z-scoring one "
             "channel does not change its order). All values in percent except ranks (1-based). The full-ranking columns "
             "use the paper's labels on the whole document.")
    for name, res in results.items():
        if res.get("traceback"):
            report.h(f"Traceback ({name})")
            report.code(res["traceback"])
    report.data["measured_pairs"] = [job_name(s, r) for s, r in todo if "error" not in results[job_name(s, r)]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=("bm25", "dense"), required=True)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--retrievers", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("C4_WORKERS", "4")))
    args = ap.parse_args()
    step = 3 if args.family == "bm25" else 4
    script = "03_bm25.sh" if step == 3 else "04_dense.sh"
    extra = [a for a in sys.argv[1:] if not a.startswith("--family")]
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        extra = sys.argv[1:i] + sys.argv[i + 2:]
    cmd = f"bash experiments/rebuttal/c4_subdoc_control/sh/{script}" + ("" if not extra else " " + " ".join(extra))
    title = "BM25 ranks inside the gold sub-document" if step == 3 else "dense-retriever ranks inside the gold sub-document"
    report = StepReport(step, title, command=cmd)
    return run_step(report, lambda r: body(r, args, args.family))


if __name__ == "__main__":
    raise SystemExit(main())
