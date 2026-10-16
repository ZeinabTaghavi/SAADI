"""Saadi command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from saadi._version import __version__

COMMANDS = {
    "index": "index",
    "query": "query",
    "demo": "demo",
    "check-prompts": "check_prompts",
    "run-experiment": "run_experiment",
    "retrieve-cached": "retrieve_cached",
    "eval": "eval",
    "eval-faithfulness-linking-sentence": "eval_faithfulness_linking_sentence",
    "eval-static-statistics-index-expansion-statistics": "eval_static_statistics_index_expansion_statistics",
    "eval-faithfulness-entity-disambiguation": "eval_faithfulness_entity_disambiguation",
    "eval-static-statistics-chunk-level-changes": "eval_static_statistics_chunk_level_changes",
    "eval-faithfulness-hierarchical-summary": "eval_faithfulness_hierarchical_summary",
    "eval-static-statistics-memory-augmentation-statistics": "eval_static_statistics_memory_augmentation_statistics",
    "eval-faithfulness-active-entity": "eval_faithfulness_active_entity",
    "eval-cumulative-growth-per-document-logging": "eval_cumulative_growth_per_document_logging",
    "eval-static-statistics-active-entity-statistics": "eval_static_statistics_active_entity_statistics",
    "eval-cumulative-growth-reporting-summaries": "eval_cumulative_growth_reporting_summaries",
    "eval-cumulative-growth-comparative-plotting": "eval_cumulative_growth_comparative_plotting",
    "eval-cumulative-growth-aggregated-curves": "eval_cumulative_growth_aggregated_curves",
    "eval-static-statistics-hierarchical-summary-statistics": "eval_static_statistics_hierarchical_summary_statistics",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="saadi",
        description="Saadi: index a document once, retrieve many times. Commands: index, query, demo, check-prompts, and the experiment tools.",
    )
    parser.add_argument("--version", action="version", version=f"saadi {__version__}")
    parser.add_argument("command", choices=sorted(COMMANDS))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    rest = argv[1:]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    module = importlib.import_module("saadi.cli." + COMMANDS[args.command])
    return module.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
