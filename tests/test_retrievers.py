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


def test_resolve_retriever_accepts_scorer_and_embedding_model_instances() -> None:
    import saadi

    assert saadi.register_retriever is register_retriever
    created: list[str] = []
    try:
        class Reranker:
            def score(self, query: str, texts: list[str]) -> list[float]:
                return [float(text.lower().count(query.lower())) for text in texts]

        reranker = Reranker()
        name = resolve_retriever(reranker)
        created.append(name)
        assert resolve_retriever(reranker) == name  # same object, same name
        retriever = get_retriever(name)(["a a", "b", "a"], chunk_ids=["x", "y", "z"])
        assert retriever.retrieve("a", k=2)[0] == ["x", "z"]

        class Model:
            def encode(self, sentences: list[str], batch_size: int = 32):
                return np.array([[len(s), s.count("e")] for s in sentences], dtype=float)

        name = resolve_retriever(Model())
        created.append(name)
        assert issubclass(get_retriever(name), EmbeddingRetriever)
        assert "model" in name  # names are normalized to lower case

        name = resolve_retriever(lambda query, texts: [float(query in text) for text in texts])
        created.append(name)
        assert get_retriever(name)(["q here", "no"]).score("q") == [1.0, 0.0]

        first = resolve_retriever(lambda texts: np.ones((len(texts), 2)))
        second = resolve_retriever(lambda texts: np.zeros((len(texts), 2)))
        created.extend([first, second])
        assert first != second  # same __name__, different objects

        with pytest.raises(TypeError, match="pass the class"):
            resolve_retriever(KeywordOverlap(["x"]))
    finally:
        for name in created:
            unregister_retriever(name)


# --- correctness of the plug-in inside the fusion step ----------------------------


class SpyRetriever(BaseRetriever):
    """Records every (corpus, query) it scores so a test can see which pools used it."""

    calls: list[tuple[tuple[str, ...], str]] = []

    def score(self, query: str) -> list[float]:
        type(self).calls.append((tuple(self.corpus), query))
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in self.corpus]


def _clear_pool_cache() -> None:
    from saadi.indexing.retriever_adapters.scoring import _ENCODED_POOL_CACHE

    _ENCODED_POOL_CACHE.clear()


def _run_demo(server: FakeAPIServer, tmp_path: Path, *, retriever: str, run_name: str, model_config: dict | None = None):
    from _fake_api_server import saadi_like_responder
    from single_document_demo.pipeline import load_demo_config, run_demo

    server.responder = saadi_like_responder
    config = load_demo_config("single_document_demo/configs/text_document.toml")
    config.demo_root = tmp_path / "demo"
    config.output_root = config.demo_root / "output"
    config.prompt_profile = str(ROOT / "single_document_demo" / "memory_prompts" / "default")
    config.run_name = run_name
    config.query = "Who leads the Aurora team?"
    config.model_config = model_config or {
        "backend": "openai",
        "model_id": "fake-model",
        "base_url": server.url + "/v1",
        "api_key": "test-key",
        "timeout": 10,
    }
    config.saadi_options.update({"use_cache": False, "chunk_size": 20, "retriever": retriever})
    result = run_demo(config)
    assert not result.failures, [record.error for record in result.failures]
    return Path(result.records[0].run_dir)


def test_registered_retriever_scores_all_three_fusion_channels(server: FakeAPIServer, tmp_path: Path) -> None:
    pytest.importorskip("openai")
    _clear_pool_cache()
    SpyRetriever.calls = []
    register_retriever("spy_retriever_test", SpyRetriever)
    try:
        run_dir = _run_demo(server, tmp_path, retriever="spy_retriever_test", run_name="spy")
    finally:
        unregister_retriever("spy_retriever_test")

    chunks = [json.loads(line)["raw_text"] for line in (run_dir / "raw_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    memory = json.load((run_dir / "cleaned_outputs" / "memory_summary.json").open(encoding="utf-8"))
    entity_names = {entity["name"] for entity in memory.get("entities", [])}
    assert len(chunks) >= 5 and entity_names  # five chunks of twenty words; the fold happens at five

    corpora = [set(corpus) for corpus, _query in SpyRetriever.calls]
    assert all(query == "Who leads the Aurora team?" for _corpus, query in SpyRetriever.calls)
    # Chunk channel: one pool holds exactly the document's chunks.
    assert any(len(corpus) == len(chunks) and any("Aurora" in text for text in corpus) for corpus in corpora)
    # Memory channel: one pool consists of entity names from the memory.
    assert any(corpus and corpus <= entity_names for corpus in corpora)
    # Summary channel: one pool consists of the fold summaries the fake model produced.
    assert any(corpus and all("alpine lake water quality" in text for text in corpus) for corpus in corpora)

    payload = json.loads((run_dir / "retrieval" / "retrieval_result.json").read_text(encoding="utf-8"))
    assert payload["retriever"] == "spy_retriever_test"
    ranked = payload["result"].get("ranked_chunk_ids") or payload["result"].get("retrieved_chunk_ids")
    assert len(ranked) == len(set(ranked)) == min(len(chunks), 10)


def test_full_hosted_stack_runs_llm_and_embeddings_on_the_same_server(
    server: FakeAPIServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("openai")
    _clear_pool_cache()
    monkeypatch.setenv("SAADI_OPENAI_EMBEDDING_BASE_URL", server.url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    run_dir = _run_demo(server, tmp_path, retriever="openai_embedding", run_name="hosted_stack")

    paths = {request["path"] for request in server.requests}
    assert paths == {"/v1/chat/completions", "/v1/embeddings"}
    embedded_texts = {text for request in server.requests if request["path"] == "/v1/embeddings" for text in request["body"]["input"]}
    chunks = [json.loads(line)["raw_text"] for line in (run_dir / "raw_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(chunks) <= embedded_texts  # the chunk pool was embedded
    assert "Who leads the Aurora team?" in embedded_texts  # and so was the query
    memory = json.load((run_dir / "cleaned_outputs" / "memory_summary.json").open(encoding="utf-8"))
    assert {entity["name"] for entity in memory["entities"]} <= embedded_texts  # entity names too

    payload = json.loads((run_dir / "retrieval" / "retrieval_result.json").read_text(encoding="utf-8"))
    assert payload["retriever"] == "openai_embedding"
    assert "Mira Chen" in (run_dir / "generation" / "answer.txt").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reproducibility"]["llm_backend"]["backend"] == "openai"


def test_cli_runs_with_a_registered_retriever_and_rejects_unknown_names(tmp_path: Path) -> None:
    from single_document_demo.run import main

    register_retriever("cli_overlap_test", KeywordOverlap)
    try:
        exit_code = main(
            ["text", "--llm", "mock", "--retriever", "cli_overlap_test", "--query", "Who leads the Aurora team?",
             "--output-root", str(tmp_path), "--run-name", "cli_retriever"]
        )
        assert exit_code == 0
        payload = json.loads((tmp_path / "sample_text" / "retrieval" / "retrieval_result.json").read_text(encoding="utf-8"))
        assert payload["retriever"] == "cli_overlap_test"
    finally:
        unregister_retriever("cli_overlap_test")

    with pytest.raises(ValueError, match="Unknown retriever: not_a_retriever. Known retrievers: bm25"):
        main(["text", "--llm", "mock", "--retriever", "not_a_retriever", "--query", "Who?", "--output-root", str(tmp_path / "bad")])


# --- registry and base-class edge cases ----------------------------------------


def test_builtin_names_take_precedence_over_registered_ones() -> None:
    register_retriever("bm25", KeywordOverlap)
    try:
        assert get_retriever("bm25") is BM25Retriever
        assert get_retriever("BM25S") is BM25Retriever
    finally:
        unregister_retriever("bm25")
    from saadi.indexing.retriever_adapters import BUILTIN_RETRIEVERS

    for name in BUILTIN_RETRIEVERS:
        assert isinstance(get_retriever(name), type)


def test_register_retriever_works_as_a_decorator() -> None:
    @register_retriever("decorated_test")
    class Decorated(KeywordOverlap):
        pass

    try:
        assert Decorated.__name__ == "Decorated"  # the decorator returns the class unchanged
        assert get_retriever("decorated_test") is Decorated
        assert resolve_retriever(Decorated) == "decorated_test"
    finally:
        unregister_retriever("decorated_test")
    with pytest.raises(ValueError, match="must not be empty"):
        register_retriever("   ", KeywordOverlap)
    with pytest.raises(TypeError, match="expects a class"):
        register_retriever("fn_test", lambda corpus: None)  # type: ignore[arg-type]


def test_base_retriever_validates_score_output_and_k() -> None:
    class Broken(BaseRetriever):
        def score(self, query: str) -> list[float]:
            return [1.0]

    with pytest.raises(ValueError, match="returned 1 scores for 2 texts"):
        Broken(["a", "b"]).retrieve("q")
    retriever = KeywordOverlap(["a b", "b c", "c d"], k=2)
    assert len(retriever.retrieve("b")[0]) == 2  # k from the constructor
    assert len(retriever.retrieve("b", k=10)[0]) == 3  # capped at the corpus size
    assert retriever.retrieve("b", k=0)[0] == retriever.retrieve("b")[0]  # k=0 means "use the default", like the built-ins
    assert KeywordOverlap([]).retrieve("q") == ([], [], [])


def test_embedding_retriever_validates_vectors_and_handles_edge_cases() -> None:
    Bad = make_embedding_retriever(lambda texts: np.ones((len(texts) + 1, 3)), name="bad_shape")
    with pytest.raises(ValueError, match="shape"):
        Bad(["a", "b"])

    Raw = make_embedding_retriever(lambda texts: np.array([[2.0, 0.0] if "a" in t else [0.0, 1.0] for t in texts]), name="raw", normalize=False)
    raw = Raw(["a", "b"], chunk_ids=["x", "y"])
    assert raw.score("a") == [4.0, 0.0]  # dot products, no normalization
    Cos = make_embedding_retriever(lambda texts: np.array([[2.0, 0.0] if "a" in t else [0.0, 1.0] for t in texts]), name="cos")
    assert Cos(["a", "b"]).score("a") == pytest.approx([1.0, 0.0])  # cosine

    Single = make_embedding_retriever(lambda texts: np.array([1.0, 2.0]) if len(texts) == 1 else np.ones((len(texts), 2)), name="single")
    assert Single(["only"]).score("q") == pytest.approx([1.0])  # a 1-D vector for one text is accepted

    empty = Cos([])
    assert empty.score("q") == []
    assert empty.retrieve("q") == ([], [], [])

    with pytest.raises(NotImplementedError):
        EmbeddingRetriever(["a"])  # no embed_fn and no embed() override
    with pytest.raises(TypeError):
        make_embedding_retriever("not callable")  # type: ignore[arg-type]


def test_make_scorer_retriever_requires_a_scorer() -> None:
    from saadi.retrievers import make_scorer_retriever

    with pytest.raises(TypeError, match="scorer"):
        make_scorer_retriever(object())
    Scorer = make_scorer_retriever(lambda query, texts: [len(t) for t in texts], name="length")
    assert Scorer.__name__ == "length_scorer_retriever"
    assert Scorer(["aa", "a"]).retrieve("q")[0] == ["0", "1"]


# --- hosted embedding retrievers: options, errors, caching -----------------------


def test_openai_embedding_retriever_reads_env_overrides_and_options(server: FakeAPIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("SAADI_OPENAI_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("SAADI_OPENAI_EMBEDDING_BASE_URL", server.url + "/v1")
    monkeypatch.setenv("SAADI_OPENAI_EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    retriever = OpenAIEmbeddingRetriever(["one", "two", "three"], dimensions=16)
    assert retriever.model == "env-model" and retriever.batch_size == 2
    assert [len(r["body"]["input"]) for r in server.requests] == [2, 1]
    assert all(r["body"]["model"] == "env-model" and r["body"]["dimensions"] == 16 for r in server.requests)
    assert server.requests[0]["headers"]["authorization"] == "Bearer env-key"

    explicit = OpenAIEmbeddingRetriever(["x"], model="explicit", base_url=server.url + "/v1", api_key="k", batch_size=50)
    assert explicit.model == "explicit" and explicit.batch_size == 50  # keywords win over the environment


def test_openai_embedding_retriever_blank_texts_and_bad_responses(server: FakeAPIServer) -> None:
    pytest.importorskip("openai")
    OpenAIEmbeddingRetriever(["", "   ", "text"], base_url=server.url + "/v1", api_key="k")
    assert server.requests[0]["body"]["input"] == [" ", " ", "text"]  # blanks are sent as a space, never empty

    one_vector = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}], "model": "m", "usage": {}}
    server.responder = lambda path, body: (200, {}, one_vector)
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        OpenAIEmbeddingRetriever(["a", "b"], base_url=server.url + "/v1", api_key="k")


def test_openai_embedding_retriever_requires_a_key_for_remote_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SAADI_OPENAI_EMBEDDING_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="openai embedding retriever needs an API key: set OPENAI_API_KEY"):
        OpenAIEmbeddingRetriever(["a"], base_url="https://api.example.com/v1")


def test_openai_embedding_retriever_shares_one_client_per_endpoint(server: FakeAPIServer) -> None:
    pytest.importorskip("openai")
    first = OpenAIEmbeddingRetriever(["a"], base_url=server.url + "/v1", api_key="shared")
    second = OpenAIEmbeddingRetriever(["b", "c"], base_url=server.url + "/v1", api_key="shared")
    other = OpenAIEmbeddingRetriever(["d"], base_url=server.url + "/v1", api_key="different")
    assert first.client() is second.client()
    assert first.client() is not other.client()


def test_ollama_embedding_retriever_env_and_keyword_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434/v1")
    assert OllamaEmbeddingRetriever([]).base_url == "http://ollama-host:11434/v1"
    assert OllamaEmbeddingRetriever([], base_url="http://explicit:1/v1").base_url == "http://explicit:1/v1"
    monkeypatch.setenv("SAADI_OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    assert OllamaEmbeddingRetriever([]).model == "mxbai-embed-large"
