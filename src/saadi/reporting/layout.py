"""Study names and the public table directory structure."""

TABLES = {
    "main": ("main/sadi_retrieval", "Saadi retrieval across datasets and retrievers"),
    "active_entity_budget": ("ablations/memory_budget", "Active memory budget"),
    "active_entity_selector": ("ablations/memory_selector", "Active memory selection"),
    "branching_factor": ("ablations/hierarchy_branching", "Hierarchy branching factor"),
    "llm_ablation": ("ablations/construction_model", "Construction language model"),
    "saadi_score_formulation_and_normalization": ("ablations/score_formulation", "Score formulation and normalization"),
    "saadi_score_fusion_weights": ("ablations/score_fusion", "Score fusion weights"),
    "saadi_score_level_2_memory_weights": ("ablations/memory_field_weights", "Memory field weights"),
    "state_conditioning_ablation": ("ablations/state_conditioning", "History and memory conditioning"),
    "context_expansion": ("robustness/context_expansion", "Context expansion"),
    "multihop": ("robustness/multihop", "Retrieval by question hop count"),
    "retrieval_depth": ("robustness/retrieval_depth", "Retrieval depth"),
    "active_entities_cumulative_growth": ("construction/memory_growth", "Cumulative memory growth"),
    "active_entities_expansion_statistics": ("construction/memory_expansion", "Memory expansion statistics"),
    "active_entities_statistics": ("construction/memory_statistics", "Entity memory statistics"),
    "hierarchical_summary_statistics": ("construction/hierarchy_statistics", "Hierarchical summary statistics and growth"),
    "construction_cost": ("construction/cost", "Saadi construction cost"),
}

STATISTICS = {
    "appendix_construction_active_entities_cumulative_growth_final_values": "table",
    "appendix_construction_active_entities_cumulative_growth_values": "growth",
    "appendix_construction_active_entities_expansion_statistics_values": "table",
    "appendix_construction_active_entities_statistics_values": "table",
    "appendix_construction_hierarchical_summary_growth_values": "growth",
    "appendix_construction_hierarchical_summary_statistics_values": "table",
    "construction_cost": "table",
    "multihop": "table",
}


def table_directory(root, study):
    return root / TABLES[study][0]


def generator_script(study):
    return f'''#!/usr/bin/env python3
"""Generate the Saadi {TABLES[study][1].lower()} table from verified study outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").is_file() and (p / "src/saadi").is_dir()), None)
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.cli import table_main

if __name__ == "__main__":
    raise SystemExit(table_main({study!r}, table_dir=HERE, project_root=ROOT))
'''
