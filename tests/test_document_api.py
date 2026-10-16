"""The public Saadi class and the index, query, and demo commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _fake_api_server import saadi_like_reply
from saadi import Answer, IndexReport, MemoryEntity, RetrievedChunk, Saadi
from saadi.document import STATE_LAYOUT
from saadi.llm import CallableBackend

ROOT = Path(__file__).resolve().parents[1]
AURORA = ROOT / "examples" / "aurora_report" / "document.txt"
SAMPLE = ROOT / "examples" / "sample_text" / "document.txt"
TRANSCRIPT = ROOT / "examples" / "sample_conversation" / "conversation.csv"
QUESTION = "Who keeps the Corvin station?"


def cooperative() -> CallableBackend:
    """A model that extracts capitalized names, folds summaries, and answers the sample question."""

    return CallableBackend(lambda system, user: saadi_like_reply(user))


def test_index_writes_a_flat_documented_state(tmp_path: Path) -> None:
    state = tmp_path / "state"
    doc = Saadi(llm=cooperative(), state_dir=state, chunk_size=20)
    report = doc.index(AURORA)
    assert isinstance(report, IndexReport)
    assert report.document_id == "document" and report.state_dir == state
    assert report.chunks >= 10 and report.new_chunks == report.chunks
    assert report.entities > 0 and report.history_levels[:2] == (0, 1)
    assert report.prompts_sent >= report.chunks  # one state update per chunk plus the folds

    assert sorted(p.name for p in state.iterdir()) == ["cache", "chunks.jsonl", "document.txt", "history", "memory.json", "saadi.json"]
    manifest = json.loads((state / "saadi.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["document_id"] == "document"
    assert manifest["settings"]["chunk_size"] == 20 and manifest["settings"]["retriever"] == "bm25"
    assert manifest["settings"]["prompts"]["name"] == "default"
    assert manifest["counts"]["chunks"] == report.chunks and manifest["counts"]["entities"] == report.entities
    assert set(manifest["layout"]) == set(STATE_LAYOUT)
    assert (state / manifest["cache_dir"]).is_dir()

    chunks = [json.loads(line) for line in (state / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["chunk_index"] for c in chunks] == list(range(report.chunks))
    assert len(doc.chunks) == report.chunks and doc.chunks[0].raw_text == chunks[0]["text"]
    assert doc.document_text == AURORA.read_text(encoding="utf-8")

    memory = doc.memory
    assert memory and all(isinstance(e, MemoryEntity) for e in memory)
    assert any(e.name == "Mira Chen" for e in memory)
    history = doc.history
    assert set(history) >= {0, 1}
    assert len(history[0]) == report.chunks
    assert all("alpine lake water quality" in item.text for item in history[1])
    assert (state / "history" / "L_1.txt").read_text(encoding="utf-8").startswith("[L1 item 0 | steps 0-4]")


def test_append_extends_the_state_without_rebuilding(tmp_path: Path) -> None:
    words = AURORA.read_text(encoding="utf-8").split()
    first, second = " ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2 :])

    doc = Saadi(llm=cooperative(), state_dir=tmp_path / "grow", chunk_size=20)
    initial = doc.index(first, document_id="aurora")
    entities_before = {e.name for e in doc.memory}
    update = doc.append(second)

    assert update.chunks > initial.chunks
    assert update.new_chunks == update.chunks - initial.chunks
    assert update.prompts_sent < initial.prompts_sent  # only the tail was processed
    assert update.prompts_sent <= update.new_chunks + 1 + update.chunks // 5 + 2  # new chunks, one boundary chunk, folds
    assert {e.name for e in doc.memory} > entities_before
    assert doc.document_text.startswith(first[:40]) and doc.document_text.rstrip().endswith(second[-20:])
    assert len(doc.history[0]) == update.chunks

    fresh = Saadi(llm=cooperative(), state_dir=tmp_path / "full", chunk_size=20)
    full = fresh.index(doc.document_text, document_id="aurora")
    assert full.chunks == update.chunks
    assert {e.name for e in fresh.memory} == {e.name for e in doc.memory}


def test_retrieve_and_answer_write_one_folder_per_query(tmp_path: Path) -> None:
    doc = Saadi(llm=cooperative(), state_dir=tmp_path / "state", chunk_size=20)
    doc.index(AURORA)

    hits = doc.retrieve(QUESTION, k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(isinstance(h, RetrievedChunk) for h in hits)
    assert "Corvin station" in hits[0].text
    assert hits[0].score is not None and hits[0].score >= hits[1].score

    reply = doc.answer(QUESTION, k=3)
    assert isinstance(reply, Answer)
    assert reply.text == "Mira Chen leads the Aurora team."
    assert reply.status == "ok" and len(reply.chunks) == 3
    assert "Corvin" in reply.context
    assert (reply.directory / "generation" / "answer.txt").read_text(encoding="utf-8").strip() == reply.text
    assert (reply.directory / "retrieval" / "retrieval_result.json").is_file()
    query_dirs = sorted(p.name for p in (tmp_path / "state" / "queries").iterdir())
    assert len(query_dirs) == 2 and query_dirs[0].startswith("001_who-keeps")
    assert json.loads((reply.directory / "query.json").read_text(encoding="utf-8"))["mode"] == "answer"


def test_load_roundtrip_and_save_copy(tmp_path: Path) -> None:
    state = tmp_path / "state"
    doc = Saadi(llm=cooperative(), state_dir=state, chunk_size=20)
    doc.index(AURORA)

    again = Saadi.load(state)
    assert again.backend is None
    assert again.document_id == "document"
    assert again.memory == doc.memory
    assert {k: len(v) for k, v in again.history.items()} == {k: len(v) for k, v in doc.history.items()}
    assert len(again.chunks) == len(doc.chunks)
    assert "Corvin station" in again.retrieve(QUESTION, k=2)[0].text  # no model needed
    with pytest.raises(RuntimeError, match="language model"):
        again.answer(QUESTION)
    assert Saadi.load(state, llm=cooperative()).answer(QUESTION).text == "Mira Chen leads the Aurora team."

    copy = doc.save(tmp_path / "copy")
    assert copy == tmp_path / "copy"
    copied = Saadi.load(copy)
    assert len(copied.chunks) == len(doc.chunks) and copied.memory == doc.memory
    assert "Saadi(" in repr(copied) and "chunks=" in repr(copied)


def test_summarize_needs_a_profile_with_a_document_summary(tmp_path: Path) -> None:
    plain = Saadi(llm=cooperative(), state_dir=tmp_path / "plain", chunk_size=20)
    plain.index(SAMPLE)
    with pytest.raises(ValueError, match="document_summary.txt"):
        plain.summarize()

    session = Saadi(llm=cooperative(), prompts="conversation", state_dir=tmp_path / "session", chunk_size=50)
    session.index(TRANSCRIPT)
    assert session.document_text.startswith("therapist: ")
    manifest = json.loads((tmp_path / "session" / "saadi.json").read_text(encoding="utf-8"))
    assert manifest["source"]["input_format"] == "conversation_csv"
    summary = session.summarize()
    assert summary and session.summary == summary
    assert (tmp_path / "session" / "summary.txt").is_file()
    session.append("patient: I also want to talk about work.")
    assert session.summary is None  # stale after the document changed


def test_load_with_a_changed_profile_keeps_retrieval_and_rebuilds_on_append(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "entity_extraction.txt").write_text("Entities are people and places.\n", encoding="utf-8")
    (profile_dir / "hierarchy_summary.txt").write_text("Keep the names.\n", encoding="utf-8")
    doc = Saadi(llm=cooperative(), prompts=profile_dir, state_dir=tmp_path / "state", chunk_size=20)
    first = doc.index(AURORA)
    old_cache = doc.cache_dir

    (profile_dir / "entity_extraction.txt").write_text("Entities are people, places, and instruments.\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="changed since this state was built"):
        again = Saadi.load(tmp_path / "state", llm=cooperative())
    assert again.cache_dir != old_cache
    assert "Corvin station" in again.retrieve(QUESTION, k=2)[0].text  # the stored construction still serves retrieval

    update = again.append("Mira Chen retired in 2031.")
    assert update.chunks >= first.chunks
    assert update.prompts_sent > first.chunks  # the whole document was re-read with the new prompts
    assert (again.cache_dir / "raw_chunks.jsonl").is_file()
    manifest = json.loads((tmp_path / "state" / "saadi.json").read_text(encoding="utf-8"))
    assert manifest["cache_dir"] == str(again.cache_dir.relative_to(tmp_path / "state"))


def test_query_folders_never_collide(tmp_path: Path) -> None:
    doc = Saadi(llm="mock", state_dir=tmp_path / "state", chunk_size=50)
    doc.index(SAMPLE)
    doc.retrieve("first question")
    doc.retrieve("second question")
    import shutil

    shutil.rmtree(next((tmp_path / "state" / "queries").glob("001_*")))
    doc.retrieve("third question")
    names = sorted(p.name for p in (tmp_path / "state" / "queries").iterdir())
    assert names == ["002_second-question", "003_third-question"]


def test_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="language model"):
        Saadi(state_dir=tmp_path / "x").index("Some text.")
    with pytest.raises(RuntimeError, match="No indexed document"):
        Saadi(llm="mock", state_dir=tmp_path / "y").retrieve("anything")
    doc = Saadi(llm="mock", state_dir=tmp_path / "z", chunk_size=50)
    doc.index(SAMPLE)
    with pytest.raises(ValueError, match="query must not be empty"):
        doc.answer("   ")
    with pytest.raises(FileNotFoundError, match="not a Saadi state directory"):
        Saadi.load(tmp_path / "nowhere")


def test_default_state_dir_and_document_id_sanitizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    doc = Saadi(llm="mock", chunk_size=50)
    report = doc.index("Some text about Mira Chen and the Aurora team.", document_id="My Doc!")
    assert report.document_id == "My_Doc"
    assert doc.state_dir == Path("saadi_state") / "My_Doc"
    assert (tmp_path / "saadi_state" / "My_Doc" / "saadi.json").is_file()


def test_cli_index_query_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from saadi.cli.main import main

    state = tmp_path / "state"
    assert main(["index", str(AURORA), "--llm", "mock", "--state", str(state), "--chunk-size", "100"]) == 0
    out = capsys.readouterr().out
    assert "Indexed document:" in out and "Entity memory: empty" in out and "State:" in out

    assert main(["query", QUESTION, "--state", str(state), "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "Top chunks:" in out and "Corvin" in out and "Answer:" not in out

    assert main(["query", QUESTION, "--state", str(state), "--llm", "mock", "--memory"]) == 0
    out = capsys.readouterr().out
    assert "Answer: Mock answer" in out and "Top chunks:" in out and "Entity memory" in out

    assert main(["index", str(SAMPLE), "--llm", "mock", "--state", str(state), "--append"]) == 0
    out = capsys.readouterr().out
    assert "new)" in out
    assert len(Saadi.load(state).chunks) > 10

    assert main(["demo", "--llm", "mock", "--state", str(tmp_path / "demo"), "--chunk-size", "100"]) == 0
    out = capsys.readouterr().out
    assert "Question: Who keeps the Corvin station?" in out and "placeholder" in out
    assert (tmp_path / "demo" / "saadi.json").is_file()

    assert main(["query", QUESTION, "--state", str(tmp_path / "missing")]) == 1
    assert "not a Saadi state directory" in capsys.readouterr().out


def test_cli_index_directory_creates_one_state_per_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from saadi.cli.main import main

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Alpha Centauri is far away.\n", encoding="utf-8")
    (folder / "b.md").write_text("Beta Pictoris has a disk.\n", encoding="utf-8")
    (folder / "notes.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "states"
    assert main(["index", str(folder), "--llm", "mock", "--state", str(root), "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert [r["document_id"] for r in reports] == ["a", "b"]
    assert (root / "a" / "saadi.json").is_file() and (root / "b" / "saadi.json").is_file()
