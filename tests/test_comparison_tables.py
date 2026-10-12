import csv
import json
from pathlib import Path
import shutil

import pytest

from saadi.reporting.comparison_inputs import load_other_methods, read_csv
from saadi.reporting.comparison_tables import build_comparison_tables, verify_auxiliary, only

ROOT = Path(__file__).resolve().parents[1]


def build(output, **kwargs):
    return build_comparison_tables(ROOT / "outputs/sadi/reference", output, outputs_root=ROOT / "outputs",
                                   input_mode="reference", **kwargs)


def save(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("fault", ["checksum", "duplicate", "invalid_hr", "sadi", "count_disagreement", "missing_file", "missing_row"])
def test_comparison_inputs_fail_closed(tmp_path, fault):
    other = tmp_path / "other_methods"
    shutil.copytree(ROOT / "outputs/other_methods", other)
    path = other / "vanilla/retrieval.csv"
    rows = read_csv(path)
    if fault == "missing_file":
        path.unlink()
    elif fault == "duplicate":
        save(path, rows + [rows[0]])
    elif fault == "sadi":
        rows[0]["method"] = "sadi"
        save(path, rows)
    elif fault == "invalid_hr":
        rows[0]["hr_percent"] = "nan"
        save(path, rows)
    elif fault == "count_disagreement":
        rows[0].update(queries="10", eligible_queries="10", hits="1")
        save(path, rows)
    elif fault == "missing_row":
        save(path, rows[1:])
        load_other_methods(other, update_checksums=True)
        with pytest.raises(ValueError, match="Expected one input"):
            build(tmp_path / "tables", studies=["retrieval_comparison"], other_methods_root=other)
        return
    else:
        rows[0]["hr_percent"] = "90"
        save(path, rows)
    with pytest.raises(ValueError):
        load_other_methods(other, update_checksums=fault != "checksum")


def test_changing_a_baseline_cannot_change_sadi_results(tmp_path):
    other = tmp_path / "other_methods"
    shutil.copytree(ROOT / "outputs/other_methods", other)
    build(tmp_path / "before", studies=["retrieval_comparison"])
    path = other / "vanilla/retrieval.csv"
    rows = read_csv(path)
    target = only(rows, study="main", dataset="loogle", retriever="bm25", cutoff="5")
    target["hr_percent"] = "12.3"
    save(path, rows)
    load_other_methods(other, update_checksums=True)
    build(tmp_path / "after", studies=["retrieval_comparison"], other_methods_root=other)
    before = read_csv(tmp_path / "before/main/retrieval_comparison/table.csv")
    after = read_csv(tmp_path / "after/main/retrieval_comparison/table.csv")
    assert [r for r in before if r["Method"] == "Saadi"] == [r for r in after if r["Method"] == "Saadi"]
    assert only(after, Method="Vanilla", Retriever="BM25")["LooGLE HR@5 (%)"] == "12.3"
    assert sum(a != b for a, b in zip(before, after)) == 1


def test_matched_depth_uses_unrounded_exposure_and_method_specific_costs(tmp_path):
    build(tmp_path, studies=["matched_exposure", "resource_use"])
    matched = read_csv(tmp_path / "main/matched_exposure/table.csv")
    # 3.99 must bracket depths 3/4; rounding exposure first would incorrectly use 4/4.
    grip = only(matched, Dataset="MuSiQue-32K", Method="GRIP")
    assert (grip["Lower k"], grip["Upper k"]) == ("3", "4")
    uma = only(matched, Dataset="MuSiQue-32K", Method="UMA")
    assert (uma["Lower k"], uma["Upper k"]) == ("19", "20")
    arag = only(matched, Dataset="LooGLE", Method="A-RAG")
    assert (arag["Lower k"], arag["Upper k"]) == ("1", "2")
    costs = read_csv(tmp_path / "comparisons/resource_use/metrics.csv")
    uma = only(costs, dataset="novelhopqa", method="uma")
    assert uma["documents"] == "16"
    assert float(uma["sadi_calls_per_document"]) == pytest.approx(545.8125)
    assert round(float(uma["calls_crossover"])) == 169


def test_extra_sadi_rankings_are_checked_and_expansion_preserves_population(tmp_path):
    aux = tmp_path / "comparison_inputs"
    shutil.copytree(ROOT / "outputs/sadi/comparison_inputs", aux)
    verify_auxiliary(aux)
    next((aux / "rankings").glob("*.gz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_auxiliary(aux)
    build(tmp_path / "tables", studies=["context_comparison"])
    rows = read_csv(tmp_path / "tables/comparisons/context_expansion/metrics.csv")
    assert len([r for r in rows if r["method"] == "sadi"]) == 60
    for retriever in ("bm25", "contriever"):
        standard = only(rows, method="sadi", dataset="qasper", retriever=retriever, condition="64K_expansion", cutoff="10")
        alternate = only(rows, method="sadi", dataset="qasper", retriever=retriever, condition="32K_expansion", cutoff="10")
        assert (standard["queries"], alternate["queries"]) == ("1333", "1321")


def test_missing_trajectory_depth_is_not_silently_clamped(tmp_path):
    other = tmp_path / "other_methods"
    shutil.copytree(ROOT / "outputs/other_methods", other)
    for name in ("retrieval", "resources"):
        path = other / f"uma/{name}.csv"
        rows = read_csv(path)
        for row in rows:
            if row["dataset"] == "musique" and (name == "resources" or row["study"] == "agentic"):
                row["mean_chunks"] = "21.2"
        save(path, rows)
    load_other_methods(other, update_checksums=True)
    with pytest.raises(ValueError, match="Expected one input"):
        build(tmp_path / "tables", studies=["matched_exposure"], other_methods_root=other)
