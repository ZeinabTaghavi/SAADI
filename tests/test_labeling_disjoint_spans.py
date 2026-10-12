from saadi.core.types import ChunkRecord
from saadi.indexing.labeling import _build_support_targets


def _chunks(texts: list[str]) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            doc_id="group_001",
            chunk_id=f"group_001:{index}",
            chunk_index=index,
            raw_text=text,
        )
        for index, text in enumerate(texts)
    ]


def test_disjoint_full_spans_do_not_label_the_interval_between_them() -> None:
    chunks = _chunks(
        [
            "unrelated zero",
            "prefix alpha evidence complete suffix",
            "unrelated middle two",
            "unrelated middle three",
            "prefix omega evidence complete suffix",
        ]
    )
    qa_entry = {
        "retrieval_spans": [
            "alpha evidence complete",
            "omega evidence complete",
        ]
    }

    gold, silver, silver_groups, labels = _build_support_targets(qa_entry, chunks)

    assert gold == ["group_001:1", "group_001:4"]
    assert silver == []
    assert silver_groups == []
    assert labels == [0, 2, 0, 0, 2]


def test_disjoint_split_spans_create_separate_local_silver_groups() -> None:
    chunks = _chunks(
        [
            "unrelated zero",
            "prefix alpha evidence",
            "continues here suffix",
            "completely unrelated middle",
            "prefix omega evidence",
            "continues there suffix",
        ]
    )
    qa_entry = {
        "retrieval_spans": [
            "alpha evidence continues here",
            "omega evidence continues there",
        ]
    }

    gold, silver, silver_groups, labels = _build_support_targets(qa_entry, chunks)

    assert gold == []
    assert silver == ["group_001:1", "group_001:2", "group_001:4", "group_001:5"]
    assert silver_groups == [
        ["group_001:1", "group_001:2"],
        ["group_001:4", "group_001:5"],
    ]
    assert labels == [0, 1, 1, 0, 1, 1]
