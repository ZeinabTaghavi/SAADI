"""``saadi index``: build or extend the state of a document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..document import Saadi
from ..inputs import KNOWN_SUFFIXES
from ._common import LLM_HELP, add_verbose_argument, configure_logging
from ._render import format_index_report, format_memory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saadi index",
        description="Build the SAADI state of a document: one state-update call per chunk, then retrieve as often as you like.",
    )
    parser.add_argument("source", help="A .txt or .md file, a conversation .csv, or a directory of such files.")
    parser.add_argument("--llm", required=True, metavar="PROVIDER:MODEL", help=LLM_HELP)
    parser.add_argument(
        "--state",
        metavar="DIR",
        help="State directory (default saadi_state/<document_id>). For a directory source, the root under which one state per file is created.",
    )
    parser.add_argument("--retriever", default="bm25", metavar="NAME", help="Query-time retriever (default bm25).")
    parser.add_argument("--prompts", default="default", metavar="PROFILE", help="Prompt profile: default, conversation, or a directory.")
    parser.add_argument("--append", action="store_true", help="Extend an existing state with the source instead of building it.")
    parser.add_argument("--document-id", metavar="ID", help="Override the document id derived from the file name.")
    parser.add_argument("--kind", choices=["text", "conversation"], help="Force the input kind; by default .csv means conversation.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens (default 500).")
    parser.add_argument("--branching-factor", type=int, default=5, help="Chunks per fold summary (default 5).")
    parser.add_argument("--max-entities", type=int, default=5, help="Entities recalled per state update (default 5).")
    parser.add_argument("--summary", action="store_true", help="Also write a whole-document summary (needs a profile with document_summary.txt).")
    parser.add_argument("--json", action="store_true", help="Print the reports as JSON.")
    add_verbose_argument(parser)
    return parser


def _sources(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in KNOWN_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No {', '.join(KNOWN_SUFFIXES)} files in {path}")
        return files
    return [path]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    source = Path(args.source).expanduser()
    try:
        files = _sources(source)
        reports = []
        for file in files:
            state_dir = (Path(args.state) / file.stem) if (args.state and source.is_dir()) else (Path(args.state) if args.state else None)
            if args.append:
                if state_dir is None:
                    raise ValueError("--append needs --state pointing at an existing state directory")
                doc = Saadi.load(state_dir, llm=args.llm, retriever=args.retriever, prompts=args.prompts)
                report = doc.append(file, kind=args.kind) if args.kind else doc.append(file)
            else:
                doc = Saadi(
                    args.llm,
                    args.retriever,
                    args.prompts,
                    state_dir=state_dir,
                    chunk_size=args.chunk_size,
                    branching_factor=args.branching_factor,
                    max_entities=args.max_entities,
                )
                options = {"kind": args.kind} if args.kind else {}
                report = doc.index(file, document_id=args.document_id if len(files) == 1 else None, **options)
            summary = doc.summarize() if args.summary else None
            reports.append({"report": report, "memory": doc.memory, "summary": summary})
    except (ValueError, FileNotFoundError, RuntimeError, TypeError) as exc:
        print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps([{**r["report"].__dict__, "state_dir": str(r["report"].state_dir), "summary": r["summary"]} for r in reports], indent=2))
        return 0
    for item in reports:
        print(format_index_report(item["report"]))
        print(format_memory(item["memory"]))
        if item["summary"]:
            print(f"Summary: {item['summary']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
