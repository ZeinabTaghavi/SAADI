"""``saadi index``: build or extend the state of a document."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ..document import DEFAULT_STATE_ROOT, Saadi
from ..inputs import KNOWN_SUFFIXES
from ._common import LLM_HELP, add_verbose_argument, configure_logging
from ._render import format_index_report, format_memory

CONSTRUCTION_FLAGS = ("--chunk-size", "--branching-factor", "--max-entities", "--document-id")


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
        help=f"State directory (default {DEFAULT_STATE_ROOT}/<document_id>). For a directory source, the root under which one state per file is created.",
    )
    parser.add_argument("--retriever", metavar="NAME", help="Query-time retriever (default bm25; with --append, the one stored in the state).")
    parser.add_argument("--prompts", metavar="PROFILE", help="Prompt profile: default, conversation, or a directory (with --append, the stored one).")
    parser.add_argument("--append", action="store_true", help="Extend the state in --state with the source instead of building it.")
    parser.add_argument("--document-id", metavar="ID", help="Override the document id derived from the file name.")
    parser.add_argument("--kind", choices=["text", "conversation"], help="Force the input kind; by default .csv means conversation.")
    parser.add_argument("--roles", metavar="ROLE,ROLE", help="Accepted speaker roles of a conversation CSV (default: any role).")
    parser.add_argument("--role-alias", action="append", default=[], metavar="OLD=NEW", help="Rename a speaker role, for example clinician=therapist; repeatable.")
    parser.add_argument("--timestamps", action="store_true", help="Keep the start column of a conversation CSV in each turn.")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in tokens (default 500).")
    parser.add_argument("--branching-factor", type=int, help="Chunks per fold summary (default 5).")
    parser.add_argument("--max-entities", type=int, help="Entities recalled per state update (default 5).")
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
    if not path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {path}")
    return [path]


def _state_names(files: list[Path]) -> list[str]:
    """One state per file; files that share a stem keep their suffix in the name."""

    stems = [file.stem for file in files]
    return [file.stem if stems.count(file.stem) == 1 else file.name.replace(".", "_") for file in files]


def _input_options(args: argparse.Namespace) -> dict:
    options: dict = {}
    if args.kind:
        options["kind"] = args.kind
    if args.roles:
        options["allowed_roles"] = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    if args.role_alias:
        aliases = {}
        for item in args.role_alias:
            if "=" not in item:
                raise ValueError(f"--role-alias expects OLD=NEW, got {item!r}")
            old, new = item.split("=", 1)
            aliases[old.strip()] = new.strip()
        options["role_aliases"] = aliases
    if args.timestamps:
        options["include_timestamps"] = True
    return options


def _construction_kwargs(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in (("chunk_size", args.chunk_size), ("branching_factor", args.branching_factor), ("max_entities", args.max_entities))
        if value is not None
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.append:
            given = [flag for flag, value in zip(CONSTRUCTION_FLAGS, (args.chunk_size, args.branching_factor, args.max_entities, args.document_id)) if value is not None]
            if given:
                raise ValueError(f"{', '.join(given)} cannot change on --append; the stored construction settings apply")
            if not args.state:
                raise ValueError("--append needs --state pointing at an existing state directory")
        source = Path(args.source).expanduser()
        files = _sources(source)
        options = _input_options(args)
        reports = []
        if source.is_dir():
            root = Path(args.state) if args.state else Path(DEFAULT_STATE_ROOT)
            targets = [(file, root / name, name) for file, name in zip(files, _state_names(files))]
        else:
            targets = [(files[0], Path(args.state) if args.state else None, args.document_id)]
        for file, state_dir, document_id in targets:
            if args.append:
                doc = Saadi.load(state_dir, llm=args.llm, retriever=args.retriever, prompts=args.prompts)
                report = doc.append(file, **options)
            else:
                doc = Saadi(args.llm, args.retriever or "bm25", args.prompts or "default", state_dir=state_dir, **_construction_kwargs(args))
                report = doc.index(file, document_id=document_id, **options)
            summary = doc.summarize() if args.summary else None
            reports.append({"report": report, "memory": doc.memory, "summary": summary})
    except Exception as exc:  # the command boundary: one line for the user, the traceback behind --verbose
        if args.verbose:
            raise
        print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps([{**dataclasses.asdict(r["report"]), "state_dir": str(r["report"].state_dir), "summary": r["summary"]} for r in reports], indent=2))
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
