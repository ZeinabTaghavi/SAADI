"""Tests for the pluggable retriever contract, the registry, fusion dispatch, and hosted embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _fake_api_server import FakeAPIServer, fake_embedding
from saadi.indexing.retriever_adapters import BM25Retriever, get_retriever
from saadi.indexing.retriever_adapters.scoring import build_encoded_text_pool, compute_similarity, encode_query
from saadi.retrievers import (
    BaseRetriever,
    EmbeddingRetriever,
    OllamaEmbeddingRetriever,
    OpenAIEmbeddingRetriever,
    make_embedding_retriever,
    register_retriever,
    registered_retrievers,
    resolve_retriever,
    unregister_retriever,
)

ROOT = Path(__file__).resolve().parents[1]


class KeywordOverlap(BaseRetriever):
    """Counts shared whitespace-separated words; good enough to check the plumbing."""

    def score(self, query: str) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in self.corpus]


@pytest.fixture
def registered_overlap():
    register_retriever("keyword_overlap_test", KeywordOverlap)
    try:
        yield "keyword_overlap_test"
    finally:
        unregister_retriever("keyword_overlap_test")


@pytest.fixture
def server():
    fake = FakeAPIServer()
    try:
        yield fake
    finally:
        fake.close()


def test_base_retriever_ranks_by_score() -> None:
    retriever = KeywordOverlap(["alpha beta", "beta gamma delta", "epsilon"], chunk_ids=["a", "b", "c"], k=2)
    ids, scores, indices = retriever.retrieve("beta delta")
    assert (ids, scores, indices) == (["b", "a"], [2.0, 1.0], [1, 0])
    assert retriever.retrieve("zzz") == (["a", "b"], [0.0, 0.0], [0, 1])  # ties keep corpus order
    assert retriever.retrieve_data("gamma") == (["beta gamma delta", "alpha beta"], [1.0, 0.0])
    with pytest.raises(ValueError, match="chunk_ids length"):
        KeywordOverlap(["x"], chunk_ids=["a", "b"])
    with pytest.raises(TypeError, match="unexpected options"):
        KeywordOverlap(["x"], bogus=1)


def test_registry_and_name_lookup(registered_overlap: str) -> None:
    assert get_retriever("bm25") is BM25Retriever
    assert get_retriever("Keyword_Overlap_Test") is KeywordOverlap
    assert registered_overlap in registered_retrievers()
    with pytest.raises(ValueError, match="Unknown retriever: nope. Known retrievers: bm25"):
        get_retriever("nope")
    with pytest.raises(ValueError, match="already registered"):
        register_retriever(registered_overlap, BM25Retriever)
    register_retriever(registered_overlap, KeywordOverlap)  # the same class again is fine


def test_fusion_scoring_dispatches_to_registered_retrievers(registered_overlap: str) -> None:
    texts = ["Mira Chen founded the team", "Leon Park keeps the station", "The lake froze"]
    pool = build_encoded_text_pool(texts, retriever_name=registered_overlap, chunk_ids=["c0", "c1", "c2"])
    assert isinstance(pool.retriever, KeywordOverlap)
    assert pool.chunk_ids == ["c0", "c1", "c2"]
    query = encode_query("who keeps the station", pool.retriever)
    assert compute_similarity(query, pool.document_representations, pool.retriever) == [1.0, 3.0, 1.0]


def test_embedding_retriever_runs_on_the_dense_fusion_path() -> None:
    Toy = make_embedding_retriever(
        lambda texts: np.array([[len(t), t.count("e")] for t in texts], dtype=float), name="toy_embedding"
    )
    assert issubclass(Toy, EmbeddingRetriever)
    register_retriever("toy_embedding_test", Toy)
    try:
        pool = build_encoded_text_pool(["eee", "a", "ee"], retriever_name="toy_embedding_test", chunk_ids=["0", "1", "2"])
        assert pool.representation_kind == "embedding"
        query = encode_query("ee", pool.retriever)
        scores = compute_similarity(query, pool.document_representations, pool.retriever)
        assert scores[0] == pytest.approx(1.0)  # same direction as the query
        assert scores[2] == pytest.approx(1.0)
        assert scores[1] < scores[0]
        ids, _scores, _ = pool.retriever.retrieve("ee", k=2)
        assert set(ids) == {"0", "2"}
    finally:
        unregister_retriever("toy_embedding_test")


def test_resolve_retriever_accepts_names_classes_and_embed_functions() -> None:
    assert resolve_retriever("BM25") == "bm25"
    name = resolve_retriever(KeywordOverlap)
    try:
        assert get_retriever(name) is KeywordOverlap
        assert resolve_retriever(KeywordOverlap) == name
    finally:
        unregister_retriever(name)

    def toy_embed(texts):
        return np.array([[len(t), t.count("a")] for t in texts], dtype=float)

    name = resolve_retriever(toy_embed)
    try:
        cls = get_retriever(name)
        assert issubclass(cls, EmbeddingRetriever)
        ids, _scores, _ = cls(["aa", "bbbb"], chunk_ids=["x", "y"]).retrieve("a")
        assert ids[0] == "x"
    finally:
        unregister_retriever(name)
    with pytest.raises(ValueError, match="Unknown retriever"):
        resolve_retriever("not_a_retriever")
    with pytest.raises(TypeError):
        resolve_retriever(42)


def test_openai_embedding_retriever_batches_requests_and_scores_by_cosine(server: FakeAPIServer) -> None:
    pytest.importorskip("openai")
    corpus = [f"text {i}" for i in range(7)]
    retriever = OpenAIEmbeddingRetriever(
        corpus,
        chunk_ids=[str(i) for i in range(7)],
        base_url=server.url + "/v1",
        api_key="test-key",
        batch_size=3,
        model="fake-embedding",
    )
    assert [request["path"] for request in server.requests] == ["/v1/embeddings"] * 3
    assert [len(request["body"]["input"]) for request in server.requests] == [3, 3, 1]
    assert server.requests[0]["body"]["model"] == "fake-embedding"
    assert server.requests[0]["headers"]["authorization"] == "Bearer test-key"

    ids, scores, _ = retriever.retrieve("text 3", k=2)
    matrix = np.array([fake_embedding(text) for text in corpus])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = np.array(fake_embedding("text 3"))
    query /= np.linalg.norm(query)
    expected = matrix @ query
    assert ids[0] == str(int(np.argmax(expected)))
    assert scores[0] == pytest.approx(float(expected.max()), abs=1e-5)
    assert len(server.requests) == 4  # one more call for the query


def test_ollama_embedding_retriever_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("SAADI_OLLAMA_EMBEDDING_BASE_URL", raising=False)
    retriever = OllamaEmbeddingRetriever([], chunk_ids=[])  # an empty corpus needs no server
    assert retriever.base_url == "http://localhost:11434/v1"
    assert retriever.model == "nomic-embed-text"
    assert retriever.retrieve("anything") == ([], [], [])
    assert get_retriever("ollama_embedding") is OllamaEmbeddingRetriever


def test_single_document_pipeline_uses_a_registered_retriever(registered_overlap: str, tmp_path: Path) -> None:
    from single_document_demo.pipeline import load_demo_config, run_demo

    config = load_demo_config("single_document_demo/configs/text_document.toml")
    config.demo_root = tmp_path / "demo"
    config.output_root = config.demo_root / "output"
    config.prompt_profile = str(ROOT / "single_document_demo" / "memory_prompts" / "default")
    config.run_name = "overlap"
    config.query = "Who leads the Aurora team?"
    config.model_config = {"backend": "mock"}
    config.saadi_options.update({"use_cache": False, "chunk_size": 20, "retriever": registered_overlap})
    result = run_demo(config)
    assert not result.failures, [record.error for record in result.failures]
    payload = json.loads((Path(result.records[0].run_dir) / "retrieval" / "retrieval_result.json").read_text(encoding="utf-8"))
    assert payload["retriever"] == registered_overlap
    ranked = payload["result"].get("ranked_chunk_ids") or payload["result"].get("retrieved_chunk_ids")
    assert ranked and ranked[0].endswith(":0")  # the chunk naming the Aurora team


def test_demo_cli_parses_retriever_option() -> None:
    from single_document_demo.run import build_arg_parser

    args = build_arg_parser().parse_args(["text", "--retriever", "openai_embedding"])
    assert args.retriever == "openai_embedding"
