#!/usr/bin/env python
"""Run one of the two public single-document SAADI workflows."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from saadi.llm import model_config_from_spec
from single_document_demo.pipeline import load_demo_config, run_demo


DEFAULT_CONFIGS = {
    "text": REPO_ROOT / "single_document_demo" / "configs" / "text_document.toml",
    "therapy": REPO_ROOT
    / "single_document_demo"
    / "configs"
    / "therapy_conversation.toml",
}


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Run default SAADI over TXT input, or conversation profile over role-labeled CSV input."
        )
    )
    parser.add_argument("workflow", choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--config", help="Optional custom TOML for the selected workflow.")
    parser.add_argument("--input", help="Input file or directory; overrides the TOML path.")
    parser.add_argument(
        "--output-root",
        help="Output root; defaults to single_document_demo/output.",
    )
    parser.add_argument(
        "--run-name",
        help="Batch label under output/_batches/; document folders mirror input folders.",
    )
    parser.add_argument(
        "--prompt-profile",
        help="Advanced override: prompt profile name or directory.",
    )
    answer_group = parser.add_mutually_exclusive_group()
    answer_group.add_argument("--query", help="One question for the text workflow.")
    answer_group.add_argument(
        "--questions",
        metavar="JSONL",
        help="Question JSONL for the text workflow.",
    )
    parser.add_argument(
        "--llm",
        metavar="PROVIDER:MODEL",
        help=(
            "Language model, for example openai:gpt-4.1-mini, ollama:qwen3:8b, "
            "anthropic:claude-opus-5-5, vllm:Qwen/Qwen3.5-27B, or mock. "
            "Overrides the TOML [model] table."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the CPU-safe mock backend; the same as --llm mock.",
    )
    parser.add_argument(
        "--retriever",
        metavar="NAME",
        help=(
            "Query-time retriever: bm25 (default), contriever, bge_m3, e5, jina, qwen, "
            "openai_embedding, ollama_embedding, or a name registered with "
            "saadi.retrievers.register_retriever."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop a directory run after its first failed document.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIGS[args.workflow]
    config = load_demo_config(config_path)
    if config.workflow != args.workflow:
        raise ValueError(
            f"Config workflow is {config.workflow!r}, but the CLI selected {args.workflow!r}"
        )
    if args.input:
        path = Path(args.input).expanduser()
        config.input_path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    if args.output_root:
        path = Path(args.output_root).expanduser()
        config.output_root = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    if args.run_name:
        config.run_name = args.run_name
    if args.prompt_profile:
        config.prompt_profile = args.prompt_profile
    if args.retriever:
        config.saadi_options["retriever"] = args.retriever.strip()
    if args.query is not None:
        config.query = args.query.strip()
        config.questions_path = ""
    if args.questions is not None:
        config.questions_path = args.questions
        config.query = ""
    if args.llm:
        config.model_config = model_config_from_spec(args.llm)
    if args.mock:
        if args.llm and config.model_config.get("backend") != "mock":
            raise ValueError("--mock and --llm select different backends; pass only one of them")
        config.model_config = {"backend": "mock"}

    result = run_demo(config, continue_on_error=False if args.fail_fast else None)
    for record in result.records:
        print(f"[{record.status}] {record.document_id}: {record.run_dir}")
        if record.cleaned_outputs_manifest:
            print(
                "  cleaned outputs: "
                f"{Path(record.cleaned_outputs_manifest).parent}"
            )
        if record.memory_summary:
            print(f"  cleaned memory: {record.memory_summary}")
        if record.document_summary:
            print(f"  document summary: {record.document_summary}")
        if record.questions_summary:
            print(f"  questions: {record.questions_summary}")
        if record.error:
            print(f"  error: {record.error}")
    print(f"Batch manifest: {result.batch_manifest_path}")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
