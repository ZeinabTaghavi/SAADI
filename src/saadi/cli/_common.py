"""Shared pieces of the index, query, and demo commands."""

from __future__ import annotations

import argparse
import logging

LLM_HELP = "Language model, for example openai:gpt-4.1-mini, ollama:qwen3:8b, anthropic:claude-opus-5-5, or mock."
SDK_LOGGERS = ("httpx", "httpcore", "openai", "anthropic")


def add_verbose_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Show the pipeline's progress log lines.")


def configure_logging(verbose: bool) -> None:
    """Keep the terminal readable: warnings and the progress bar by default, pipeline logs with --verbose."""

    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)
    for name in SDK_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)  # one line per HTTP request is never useful here


__all__ = ["LLM_HELP", "SDK_LOGGERS", "add_verbose_argument", "configure_logging"]
