"""Cumulative-growth analysis runners for updated chunks."""

from .aggregated_curves import run_dataset_level_aggregated_cumulative_curves_eval
from .comparative_plotting import run_comparative_plotting_across_datasets_eval
from .per_document_logging import run_per_document_cumulative_growth_logging_eval
from .reporting_summaries import run_methodology_summary_and_reporting_outputs_eval

__all__ = [
    "run_dataset_level_aggregated_cumulative_curves_eval",
    "run_comparative_plotting_across_datasets_eval",
    "run_per_document_cumulative_growth_logging_eval",
    "run_methodology_summary_and_reporting_outputs_eval",
]
