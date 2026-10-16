"""``saadi query``: retrieve chunks and optionally answer a question over an indexed state."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ..document import Saadi
from ._common import LLM_HELP, add_verbose_argument, configure_logging
from ._render import format_answer, format_chunks, format_memory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saadi query", description="Retrieve the best chunks for a question and, with --llm, answer it.")
    parser.add_argument("question")
    parser.add_argument("--state", required=True, metavar="DIR", help="State directory written by `saadi index`.")
    parser.add_argument("--llm", metavar="PROVIDER:MODEL", help=LLM_HELP + " Without it, only retrieval runs.")
    parser.add_argument("--k", type=int, default=5, help="Chunks to return and to read for the answer (default 5).")
    parser.add_argument("--retriever", metavar="NAME", help="Override the retriever stored in the state.")
    parser.add_argument("--memory", action="store_true", help="Also print the entity memory.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    add_verbose_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        doc = Saadi.load(Path(args.state), llm=args.llm, retriever=args.retriever)
        if args.llm:
            result = doc.answer(args.question, k=args.k)
            payload = {
                "question": result.question,
                "answer": result.text,
                "status": result.status,
                "chunks": [dataclasses.asdict(c) for c in result.chunks],
            }
            text = format_answer(result)
        else:
            chunks = doc.retrieve(args.question, k=args.k)
            payload = {"question": args.question, "chunks": [dataclasses.asdict(c) for c in chunks]}
            text = f"Question: {args.question}\n" + format_chunks(chunks)
    except Exception as exc:  # the command boundary: one line for the user, the traceback behind --verbose
        if args.verbose:
            raise
        print(f"error: {exc}")
        return 1
    if args.json:
        if args.memory:
            payload["memory"] = [dataclasses.asdict(e) for e in doc.memory]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(text)
    if args.memory:
        print()
        print(format_memory(doc.memory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
