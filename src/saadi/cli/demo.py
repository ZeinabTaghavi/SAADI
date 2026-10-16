"""``saadi demo``: index a bundled report and answer a question whose evidence sits mid-document."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from ..document import Saadi
from ._render import format_answer, format_index_report, format_memory
from .index import LLM_HELP

DEFAULT_QUESTION = "Who keeps the Corvin station?"


def bundled_report() -> str:
    return resources.files("saadi").joinpath("data", "aurora_report.txt").read_text(encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saadi demo",
        description="Index a bundled synthetic report, show the entity memory, and answer a question from the middle of it.",
    )
    parser.add_argument("--llm", default="mock", metavar="PROVIDER:MODEL", help=LLM_HELP + " Default: mock (placeholder outputs).")
    parser.add_argument("--retriever", default="bm25", metavar="NAME")
    parser.add_argument("--prompts", default="default", metavar="PROFILE")
    parser.add_argument("--state", metavar="DIR", default="saadi_state/aurora_report", help="Where to keep the state (default saadi_state/aurora_report).")
    parser.add_argument("--query", default=DEFAULT_QUESTION)
    parser.add_argument("--chunk-size", type=int, default=100, help="Small chunks so the report folds into summaries (default 100).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        doc = Saadi(args.llm, args.retriever, args.prompts, state_dir=Path(args.state), chunk_size=args.chunk_size)
        print("Indexing the bundled report (a synthetic account of a fictional research team)...")
        report = doc.index(bundled_report(), document_id="aurora_report")
        print(format_index_report(report))
        print()
        print(format_memory(doc.memory, limit=8))
        print()
        result = doc.answer(args.query)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    print(format_answer(result))
    if str(args.llm).strip().lower() in {"mock", "dummy"}:
        print()
        print("The mock model returns placeholder outputs; pass --llm openai:gpt-4.1-mini (or ollama:..., anthropic:...) for real results.")
    print()
    print(f"Inspect the state in {doc.state_dir}: memory.json, history/, queries/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
