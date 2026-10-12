import json
from pathlib import Path

import pytest

from saadi.reporting.paper_consistency import audit_paper, compare_cell
from saadi.reporting.release_tables import build_release_tables

ROOT = Path(__file__).resolve().parents[1]


def test_paper_rounding_uses_full_precision_and_detects_different_values():
    cell = {"result_file": "table.csv", "row_selector": '{"Dataset":"NovelHopQA"}',
            "column": "HR@5 (%)", "paper_value": "13.5", "decimals": "1", "scale": "1"}
    rows = [{"Dataset": "NovelHopQA", "HR@5 (%)": "13.54625550660793"}]
    assert compare_cell(cell, rows) == ("match", "13.5")
    rows[0]["HR@5 (%)"] = "12.7"
    assert compare_cell(cell, rows) == ("mismatch", "12.7")


def test_paper_check_distinguishes_missing_rows_columns_and_ambiguity():
    cell = {"result_file": "table.csv", "row_selector": '{"Dataset":"LooGLE"}',
            "column": "HR@15 (%)", "paper_value": "98.3", "decimals": "1", "scale": "1"}
    assert compare_cell(cell, []) == ("missing_row", "")
    row = {"Dataset": "LooGLE"}
    assert compare_cell(cell, [row]) == ("missing_column", "")
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_cell(cell, [row, row])


def test_fresh_main_and_depth_tables_match_the_independent_paper_cells(tmp_path):
    tables = tmp_path / "tables"
    build_release_tables(ROOT / "outputs/sadi/reference", tables, outputs_root=ROOT / "outputs", input_mode="reference")
    report = audit_paper(tables, ROOT / "results/paper_consistency", tmp_path / "audit")
    for table in ("1", "2", "6A", "6B", "7", "8", "9", "10", "11", "12", "16", "17", "21", "22", "23", "24", "26", "27a"):
        assert set(report["by_table"][table]) == {"match"}, table
    assert report["by_table"]["1"]["match"] == 184
    assert report["by_table"]["23"]["match"] == 80
    assert report["by_table"]["26"]["match"] == 60
    assert report["by_table"]["6A"]["match"] == 16
    assert report["by_table"]["6B"]["match"] == 8
    # Standard expansion conditions must use the same query population and HR
    # as the main study, rather than an earlier subset with a similar score.
    metrics = json.loads((ROOT / "outputs/sadi/reference/expected_metrics.json").read_text())
    for dataset, condition in (("musique", "32K_expansion"), ("qasper", "64K_expansion")):
        for k in (5, 10):
            main = next(r for r in metrics if r["study"] == "main" and r["dataset"] == dataset
                        and r["retriever"] == "bm25" and r["cutoff"] == k)
            expansion = next(r for r in metrics if r["study"] == "context_expansion" and r["dataset"] == dataset
                             and r["condition"] == condition and r["cutoff"] == k)
            assert {key: main[key] for key in ("queries", "eligible", "hits", "hr")} == {
                key: expansion[key] for key in ("queries", "eligible", "hits", "hr")}
