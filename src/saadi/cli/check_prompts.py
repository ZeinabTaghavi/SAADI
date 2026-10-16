"""``saadi check-prompts``: run one chunk through a prompt profile and validate the parse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..prompts import builtin_profile_names, check_prompt_profile, format_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saadi check-prompts",
        description=(
            "Load a prompt profile, send one entity-extraction prompt and one fold prompt, and check that "
            "the answers follow the output contract."
        ),
    )
    parser.add_argument(
        "profile",
        help=f"Built-in profile name ({', '.join(builtin_profile_names())}) or a directory with the guidance files.",
    )
    parser.add_argument(
        "--llm",
        default="mock",
        metavar="PROVIDER:MODEL",
        help="Language model to test with, for example openai:gpt-4.1-mini or ollama:qwen3:8b. Default: mock.",
    )
    parser.add_argument("--text", metavar="FILE", help="Text file whose beginning is used as the sample chunk.")
    parser.add_argument("--words", type=int, default=120, help="Words of --text to use as the chunk (default 120).")
    parser.add_argument("--max-entities", type=int, default=5, help="Entity cap passed to the extraction prompt.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    text = None
    if args.text:
        words = Path(args.text).expanduser().read_text(encoding="utf-8").split()
        text = " ".join(words[: max(1, int(args.words))])
    try:
        report = check_prompt_profile(args.profile, backend=args.llm, text=text, max_entities=int(args.max_entities))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) if args.json else format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
