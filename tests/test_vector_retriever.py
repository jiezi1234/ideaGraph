from src.schemas.document import RetrievedPaperDocument

from src.retrieval.base import PaperRetriever
from src.retrieval.embedding_cache import PaperEmbeddingCache
from src.retrieval.vector_retriever import VectorEntityRetriever, VectorPaperRetriever


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    @property
    def model_name(self):
        return "qwen3-embedding:4b"

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.mapping[text]


class OrderedFakeEmbedder(FakeEmbedder):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.active_item_embeddings = []
        self._query_returned = False

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self._query_returned:
            self.active_item_embeddings.append(text)
        self._query_returned = True
        return self.mapping[text]


class FakeRepository:
    def __init__(self, papers, entities=None):
        self._papers = papers
        self._entities = entities or []

    def list_papers_with_abstracts(self):
        return list(self._papers)

    def list_entities(self):
        return list(self._entities)


def test_vector_retriever_is_a_paper_retriever(tmp_path):
    retriever = VectorPaperRetriever(
        repository=FakeRepository([]),
        embedder=FakeEmbedder({"query": [1.0, 0.0]}),
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
    )

    assert isinstance(retriever, PaperRetriever)


def test_vector_retriever_ranks_by_score_and_limits_top_k(tmp_path):
    papers = [
        RetrievedPaperDocument(
            document_id="p1",
            title="Paper One",
            abstract="abstract-1",
            cited_by_count=10,
        ),
        RetrievedPaperDocument(
            document_id="p2",
            title="Paper Two",
            abstract="abstract-2",
            cited_by_count=20,
        ),
        RetrievedPaperDocument(
            document_id="p3",
            title="Paper Three",
            abstract="abstract-3",
            cited_by_count=30,
        ),
    ]
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "abstract-1": [1.0, 0.0],
            "abstract-2": [0.6, 0.8],
            "abstract-3": [0.0, 1.0],
        }
    )
    retriever = VectorPaperRetriever(
        repository=FakeRepository(papers),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
    )

    results = retriever.retrieve("query", top_k=2)

    assert [paper.paper_id for paper in results] == ["p1", "p2"]
    assert results[0].score > results[1].score > 0
    assert results[0].cited_by_count == 10


def test_vector_retriever_skips_papers_without_abstracts(tmp_path):
    papers = [
        RetrievedPaperDocument(
            document_id="p1",
            title="Useful Paper",
            abstract="useful abstract",
        ),
        RetrievedPaperDocument(
            document_id="p2",
            title="Empty Paper",
            abstract="",
        ),
    ]
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "useful abstract": [1.0, 0.0],
        }
    )
    retriever = VectorPaperRetriever(
        repository=FakeRepository(papers),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
    )

    results = retriever.retrieve("query", top_k=2)

    assert [paper.paper_id for paper in results] == ["p1"]
    assert embedder.calls == ["query", "useful abstract"]


def test_vector_retriever_promotes_query_term_matches_when_embedding_scores_are_close(tmp_path):
    papers = [
        RetrievedPaperDocument(
            document_id="word2vec",
            title="Efficient Estimation of Word Representations in Vector Space",
            abstract="This paper learns word vectors from large text corpora.",
            cited_by_count=18105,
        ),
        RetrievedPaperDocument(
            document_id="kg-survey",
            title="A Survey on Knowledge Graphs: Representation, Acquisition, and Applications",
            abstract="Knowledge graphs represent structural relations between entities for applications.",
            cited_by_count=2594,
        ),
    ]
    embedder = FakeEmbedder(
        {
            "graph-based scientific idea generation": [1.0, 0.0],
            "This paper learns word vectors from large text corpora.": [0.596, 0.803],
            "Knowledge graphs represent structural relations between entities for applications.": [0.592, 0.806],
        }
    )
    retriever = VectorPaperRetriever(
        repository=FakeRepository(papers),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
    )

    results = retriever.retrieve("graph-based scientific idea generation", top_k=2)

    assert [paper.paper_id for paper in results] == ["kg-survey", "word2vec"]


def test_vector_retriever_uses_cached_paper_embeddings(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)
    cache.save("p1", "qwen3-embedding:4b", "abstract-1", [1.0, 0.0])
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Cached Paper",
        abstract="abstract-1",
        cited_by_count=10,
    )
    embedder = FakeEmbedder({"query": [1.0, 0.0]})
    retriever = VectorPaperRetriever(
        repository=FakeRepository([paper]),
        embedder=embedder,
        cache=cache,
    )

    results = retriever.retrieve("query", top_k=1)

    assert [paper.paper_id for paper in results] == ["p1"]
    assert embedder.calls == ["query"]


def test_vector_retriever_recomputes_embedding_when_cached_abstract_is_stale(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)
    cache.save("p1", "qwen3-embedding:4b", "abstract-old", [1.0, 0.0])
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Updated Paper",
        abstract="abstract-new",
        cited_by_count=10,
    )
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "abstract-new": [0.0, 1.0],
        }
    )
    retriever = VectorPaperRetriever(
        repository=FakeRepository([paper]),
        embedder=embedder,
        cache=cache,
    )

    results = retriever.retrieve("query", top_k=1)

    assert [paper.paper_id for paper in results] == ["p1"]
    assert embedder.calls == ["query", "abstract-new"]
    assert cache.load("p1", "qwen3-embedding:4b", "abstract-new") == [0.0, 1.0]


def test_vector_retriever_uses_configured_worker_count_for_item_embeddings(tmp_path):
    papers = [
        RetrievedPaperDocument(document_id="p1", title="Paper One", abstract="abstract-1"),
        RetrievedPaperDocument(document_id="p2", title="Paper Two", abstract="abstract-2"),
    ]
    embedder = OrderedFakeEmbedder(
        {
            "query": [1.0, 0.0],
            "abstract-1": [1.0, 0.0],
            "abstract-2": [0.6, 0.8],
        }
    )
    retriever = VectorPaperRetriever(
        repository=FakeRepository(papers),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
        worker_count=2,
    )

    results = retriever.retrieve("query", top_k=2)

    assert [paper.paper_id for paper in results] == ["p1", "p2"]
    assert embedder.calls[0] == "query"
    assert sorted(embedder.active_item_embeddings) == ["abstract-1", "abstract-2"]


def test_vector_entity_retriever_ranks_non_paper_entities_by_embedding_and_term_bonus(tmp_path):
    entities = [
        {
            "id": "graph rag",
            "entity_type": "Method",
            "aliases": ["graph retrieval augmented generation"],
        },
        {
            "id": "citation graph",
            "entity_type": "Concept",
            "aliases": [],
        },
        {
            "id": "Graph-guided discovery",
            "entity_type": "Paper",
            "aliases": ["should be ignored"],
        },
    ]
    embedder = FakeEmbedder(
        {
            "graph rag innovation": [1.0, 0.0],
            "graph rag\nMethod": [1.0, 0.0],
            "citation graph\nConcept": [0.6, 0.8],
        }
    )
    retriever = VectorEntityRetriever(
        repository=FakeRepository([], entities=entities),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
    )

    results = retriever.retrieve("graph rag innovation", top_k=10)

    assert [entity["id"] for entity in results] == ["graph rag", "citation graph"]
    assert results[0]["score"] > results[1]["score"]
    assert embedder.calls == [
        "graph rag innovation",
        "graph rag\nMethod",
        "citation graph\nConcept",
    ]


def test_vector_entity_retriever_uses_cached_entity_embeddings(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)
    entity_text = "graph rag\nMethod"
    cache.save("entity:graph rag", "qwen3-embedding:4b", entity_text, [1.0, 0.0])
    embedder = FakeEmbedder({"graph rag": [1.0, 0.0]})
    retriever = VectorEntityRetriever(
        repository=FakeRepository([], entities=[{"id": "graph rag", "entity_type": "Method", "aliases": []}]),
        embedder=embedder,
        cache=cache,
    )

    results = retriever.retrieve("graph rag", top_k=1)

    assert [entity["id"] for entity in results] == ["graph rag"]
    assert embedder.calls == ["graph rag"]


def test_vector_entity_retriever_uses_configured_worker_count_for_item_embeddings(tmp_path):
    entities = [
        {"id": "graph rag", "entity_type": "Method", "aliases": []},
        {"id": "citation graph", "entity_type": "Concept", "aliases": []},
    ]
    embedder = OrderedFakeEmbedder(
        {
            "graph rag": [1.0, 0.0],
            "graph rag\nMethod": [1.0, 0.0],
            "citation graph\nConcept": [0.6, 0.8],
        }
    )
    retriever = VectorEntityRetriever(
        repository=FakeRepository([], entities=entities),
        embedder=embedder,
        cache=PaperEmbeddingCache(cache_dir=tmp_path),
        worker_count=2,
    )

    results = retriever.retrieve("graph rag", top_k=2)

    assert [entity["id"] for entity in results] == ["graph rag", "citation graph"]
    assert embedder.calls[0] == "graph rag"
    assert sorted(embedder.active_item_embeddings) == ["citation graph\nConcept", "graph rag\nMethod"]
