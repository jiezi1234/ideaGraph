from pathlib import Path

from src.retrieval.chroma_retriever import (
    ChromaChunkRetriever,
    ChromaEntityRetriever,
    ChromaPaperRetriever,
    ChromaVectorIndex,
    chroma_persist_path,
)
from src.schemas.document import RetrievedPaperDocument


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    @property
    def model_name(self):
        return "fake-embedding"

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.mapping[text]


class CharacterTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

    def split_text(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
        if len(text) <= max_tokens:
            return [text.strip()]
        step = max(max_tokens - overlap_tokens, 1)
        parts = []
        for start in range(0, len(text), step):
            part = text[start : start + max_tokens].strip()
            if part:
                parts.append(part)
            if start + max_tokens >= len(text):
                break
        return parts


class FakeCollection:
    def __init__(self):
        self.points = {}
        self.query_calls = []

    def get(self, ids=None, include=None):
        del include
        selected_ids = ids if ids is not None else list(self.points)
        existing_ids = [point_id for point_id in selected_ids if point_id in self.points]
        return {
            "ids": existing_ids,
            "metadatas": [self.points[point_id]["metadata"] for point_id in existing_ids],
        }

    def upsert(self, *, ids, documents, embeddings, metadatas):
        for point_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas, strict=True):
            self.points[point_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def delete(self, *, ids):
        for point_id in ids:
            self.points.pop(point_id, None)

    def query(self, *, query_embeddings, n_results, include=None, where=None):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include,
                "where": where,
            }
        )
        rows = []
        for point_id, point in self.points.items():
            if where and not all(point["metadata"].get(key) == value for key, value in where.items()):
                continue
            distance = point["metadata"].get("_distance", 0.2)
            rows.append((distance, point_id, point))
        rows.sort(key=lambda row: row[0])
        rows = rows[:n_results]
        return {
            "ids": [[point_id for _, point_id, _ in rows]],
            "documents": [[point["document"] for _, _, point in rows]],
            "metadatas": [[point["metadata"] for _, _, point in rows]],
            "distances": [[distance for distance, _, _ in rows]],
        }


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.created = []

    def get_or_create_collection(self, name, metadata=None):
        self.created.append((name, metadata))
        return self.collections.setdefault(name, FakeCollection())


class FakeRepository:
    def __init__(self, papers=None, entities=None):
        self._papers = papers or []
        self._entities = entities or []

    def list_papers_with_abstracts(self):
        return list(self._papers)

    def list_papers_for_vector_index(self):
        return list(self._papers)

    def list_entities(self):
        return list(self._entities)


def test_chroma_persist_path_is_scoped_by_neo4j_database():
    assert chroma_persist_path("testidea") == Path(".cache/chroma/testidea")


def test_chroma_paper_retriever_indexes_title_and_abstract_then_returns_candidates():
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Graph-guided discovery",
        abstract="Knowledge graphs support scientific idea generation.",
        cited_by_count=7,
    )
    embedder = FakeEmbedder(
        {
            "graph ideas": [1.0, 0.0],
            "Title: Graph-guided discovery\nAbstract: Knowledge graphs support scientific idea generation.": [0.9, 0.1],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)
    retriever = ChromaPaperRetriever(
        repository=FakeRepository(papers=[paper]),
        index=index,
    )

    results = retriever.retrieve("graph ideas", top_k=1)

    collection = client.collections["papers"]
    stored = next(iter(collection.points.values()))
    assert stored["document"] == "Title: Graph-guided discovery\nAbstract: Knowledge graphs support scientific idea generation."
    assert stored["metadata"]["paper_id"] == "p1"
    assert stored["metadata"]["neo4j_database"] == "testidea"
    assert [candidate.paper_id for candidate in results] == ["p1"]
    assert results[0].title == "Graph-guided discovery"
    assert results[0].abstract == "Knowledge graphs support scientific idea generation."
    assert results[0].cited_by_count == 7


def test_chroma_paper_index_skips_papers_without_abstracts():
    useful_paper = RetrievedPaperDocument(
        document_id="p1",
        title="Useful Paper",
        abstract="A useful abstract.",
    )
    empty_paper = RetrievedPaperDocument(
        document_id="p2",
        title="Empty Paper",
        abstract="",
    )
    embedder = FakeEmbedder(
        {
            "Title: Useful Paper\nAbstract: A useful abstract.": [1.0, 0.0],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)

    index.sync_papers([useful_paper, empty_paper])

    collection = client.collections["papers"]
    assert [point["metadata"]["paper_id"] for point in collection.points.values()] == ["p1"]
    assert embedder.calls == ["Title: Useful Paper\nAbstract: A useful abstract."]


def test_chroma_entity_retriever_indexes_id_type_and_description_with_type_filter():
    entities = [
        {
            "id": "bus scheduling",
            "entity_type": "Problem",
            "description": "A scheduling problem for assigning buses to service plans.",
        },
        {
            "id": "multi-route",
            "entity_type": "Scenario",
            "description": "A condition involving more than one bus route.",
            "_distance": 0.1,
        },
    ]
    embedder = FakeEmbedder(
        {
            "bus dispatch": [1.0, 0.0],
            "Entity: bus scheduling\nType: Problem\nDescription: A scheduling problem for assigning buses to service plans.": [0.9, 0.1],
            "Entity: multi-route\nType: Scenario\nDescription: A condition involving more than one bus route.": [0.6, 0.4],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)
    retriever = ChromaEntityRetriever(
        repository=FakeRepository(entities=entities),
        index=index,
    )

    results = retriever.retrieve("bus dispatch", top_k=5, entity_type="Problem")

    collection = client.collections["entities"]
    stored_documents = {point["metadata"]["entity_id"]: point["document"] for point in collection.points.values()}
    assert stored_documents["bus scheduling"] == (
        "Entity: bus scheduling\n"
        "Type: Problem\n"
        "Description: A scheduling problem for assigning buses to service plans."
    )
    assert results == [
        {
            "id": "bus scheduling",
            "entity_type": "Problem",
            "score": 0.8,
        }
    ]
    assert collection.query_calls[0]["where"] == {"entity_type": "Problem"}


def test_chroma_index_skips_unchanged_documents_and_deletes_stale_points():
    first_paper = RetrievedPaperDocument(document_id="p1", title="Paper One", abstract="abstract one")
    second_paper = RetrievedPaperDocument(document_id="p2", title="Paper Two", abstract="abstract two")
    embedder = FakeEmbedder(
        {
            "Title: Paper One\nAbstract: abstract one": [1.0, 0.0],
            "Title: Paper Two\nAbstract: abstract two": [0.0, 1.0],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)

    index.sync_papers([first_paper, second_paper])
    embedder.calls.clear()
    index.sync_papers([first_paper])

    collection = client.collections["papers"]
    assert embedder.calls == []
    assert [point["metadata"]["paper_id"] for point in collection.points.values()] == ["p1"]


def test_chroma_chunk_retriever_indexes_paragraph_chunks_with_paper_metadata(tmp_path):
    first_paragraph = " ".join(["alpha"] * 800)
    second_paragraph = " ".join(["beta"] * 800)
    third_paragraph = " ".join(["gamma"] * 1200)
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        f"{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}\n",
        encoding="utf-8",
    )
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Paragraph Paper",
        abstract="Fallback abstract.",
        full_text_markdown_path=str(markdown_path),
        text_source="full_text_md",
    )
    first_chunk_document = f"Title: Paragraph Paper\nChunk:\n{first_paragraph}\n\n{second_paragraph}"
    second_chunk_document = f"Title: Paragraph Paper\nChunk:\n{third_paragraph}"
    embedder = FakeEmbedder(
        {
            "paragraph query": [1.0, 0.0],
            first_chunk_document: [0.9, 0.1],
            second_chunk_document: [0.8, 0.2],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)
    retriever = ChromaChunkRetriever(repository=FakeRepository(papers=[paper]), index=index)

    results = retriever.retrieve("paragraph query", top_k=2)

    collection = client.collections["chunks"]
    stored_by_index = {
        point["metadata"]["chunk_index"]: point
        for point in collection.points.values()
    }
    assert stored_by_index[0]["document"] == first_chunk_document
    assert stored_by_index[0]["metadata"]["paper_id"] == "p1"
    assert stored_by_index[0]["metadata"]["title"] == "Paragraph Paper"
    assert stored_by_index[0]["metadata"]["paragraph_start"] == 0
    assert stored_by_index[0]["metadata"]["paragraph_end"] == 1
    assert stored_by_index[0]["metadata"]["text_source"] == "full_text_md"
    assert stored_by_index[1]["document"] == second_chunk_document
    assert stored_by_index[1]["metadata"]["paragraph_start"] == 2
    assert stored_by_index[1]["metadata"]["paragraph_end"] == 2
    assert [chunk.paper_id for chunk in results] == ["p1", "p1"]
    assert results[0].title == "Paragraph Paper"
    assert results[0].chunk_index == 0
    assert results[0].paragraph_start == 0
    assert results[0].paragraph_end == 1
    assert results[0].text == first_chunk_document
    assert results[0].text_source == "full_text_md"


def test_chroma_chunker_packs_paragraphs_toward_max_token_limit(tmp_path):
    first_paragraph = "a" * 700
    second_paragraph = "b" * 700
    third_paragraph = "c" * 700
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        f"{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}",
        encoding="utf-8",
    )
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Long Paragraph Paper",
        abstract="Fallback abstract.",
        full_text_markdown_path=str(markdown_path),
        text_source="full_text_md",
    )
    first_chunk_document = f"Title: Long Paragraph Paper\nChunk:\n{first_paragraph}\n\n{second_paragraph}"
    second_chunk_document = f"Title: Long Paragraph Paper\nChunk:\n{third_paragraph}"
    embedder = FakeEmbedder(
        {
            first_chunk_document: [0.9, 0.1],
            second_chunk_document: [0.8, 0.2],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(
        client=client,
        database_name="testidea",
        embedder=embedder,
        tokenizer=CharacterTokenizer(),
        chunk_min_tokens=256,
        chunk_max_tokens=2048,
        chunk_overlap_tokens=0,
    )

    index.sync_chunks([paper])

    collection = client.collections["chunks"]
    stored_by_index = {
        point["metadata"]["chunk_index"]: point
        for point in collection.points.values()
    }
    assert stored_by_index[0]["document"] == first_chunk_document
    assert stored_by_index[0]["metadata"]["paragraph_start"] == 0
    assert stored_by_index[0]["metadata"]["paragraph_end"] == 1
    assert stored_by_index[1]["document"] == second_chunk_document
    assert stored_by_index[1]["metadata"]["paragraph_start"] == 2
    assert stored_by_index[1]["metadata"]["paragraph_end"] == 2


def test_chroma_chunker_splits_single_oversized_paragraph_with_tokenizer(tmp_path):
    long_paragraph = "x" * 2500
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(long_paragraph, encoding="utf-8")
    paper = RetrievedPaperDocument(
        document_id="p1",
        title="Oversized Paragraph Paper",
        abstract="Fallback abstract.",
        full_text_markdown_path=str(markdown_path),
        text_source="full_text_md",
    )
    first_chunk_document = f"Title: Oversized Paragraph Paper\nChunk:\n{long_paragraph[:2048]}"
    second_chunk_document = f"Title: Oversized Paragraph Paper\nChunk:\n{long_paragraph[2048:]}"
    embedder = FakeEmbedder(
        {
            first_chunk_document: [0.9, 0.1],
            second_chunk_document: [0.8, 0.2],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(
        client=client,
        database_name="testidea",
        embedder=embedder,
        tokenizer=CharacterTokenizer(),
        chunk_min_tokens=256,
        chunk_max_tokens=2048,
        chunk_overlap_tokens=0,
    )

    index.sync_chunks([paper])

    collection = client.collections["chunks"]
    stored_by_index = {
        point["metadata"]["chunk_index"]: point
        for point in collection.points.values()
    }
    assert stored_by_index[0]["document"] == first_chunk_document
    assert stored_by_index[0]["metadata"]["paragraph_start"] == 0
    assert stored_by_index[0]["metadata"]["paragraph_end"] == 0
    assert stored_by_index[1]["document"] == second_chunk_document
    assert stored_by_index[1]["metadata"]["paragraph_start"] == 0
    assert stored_by_index[1]["metadata"]["paragraph_end"] == 0


def test_chroma_chunk_index_uses_full_text_before_abstract_and_skips_empty_papers(tmp_path):
    full_text = " ".join(["orphan"] * 130)
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(full_text, encoding="utf-8")
    useful_paper = RetrievedPaperDocument(
        document_id="p1",
        title="Useful Paper",
        abstract="A useful abstract.",
    )
    empty_paper = RetrievedPaperDocument(
        document_id="p2",
        title="Empty Paper",
        abstract="",
        full_text_markdown_path=str(markdown_path),
        text_source="full_text_md",
    )
    no_text_paper = RetrievedPaperDocument(
        document_id="p3",
        title="No Text Paper",
        abstract="",
    )
    useful_document = "Title: Useful Paper\nChunk:\nAbstract: A useful abstract."
    full_text_document = f"Title: Empty Paper\nChunk:\n{full_text}"
    embedder = FakeEmbedder(
        {
            useful_document: [1.0, 0.0],
            full_text_document: [0.9, 0.1],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)

    index.sync_chunks([useful_paper, empty_paper, no_text_paper])

    collection = client.collections["chunks"]
    stored = list(collection.points.values())
    assert [point["metadata"]["paper_id"] for point in stored] == ["p1", "p2"]
    assert [point["metadata"]["text_source"] for point in stored] == ["title_abstract", "full_text_md"]
    assert embedder.calls == [useful_document, full_text_document]


def test_chroma_chunk_sync_skips_unchanged_chunks_and_deletes_stale_points(tmp_path):
    first_path = tmp_path / "first.md"
    first_text = " ".join(["alpha"] * 130)
    first_path.write_text(first_text, encoding="utf-8")
    second_path = tmp_path / "second.md"
    second_text = " ".join(["beta"] * 130)
    second_path.write_text(second_text, encoding="utf-8")
    first_paper = RetrievedPaperDocument(
        document_id="p1",
        title="First Paper",
        abstract="First abstract.",
        full_text_markdown_path=str(first_path),
        text_source="full_text_md",
    )
    second_paper = RetrievedPaperDocument(
        document_id="p2",
        title="Second Paper",
        abstract="Second abstract.",
        full_text_markdown_path=str(second_path),
        text_source="full_text_md",
    )
    first_document = f"Title: First Paper\nChunk:\n{first_text}"
    second_document = f"Title: Second Paper\nChunk:\n{second_text}"
    embedder = FakeEmbedder(
        {
            first_document: [1.0, 0.0],
            second_document: [0.0, 1.0],
        }
    )
    client = FakeClient()
    index = ChromaVectorIndex(client=client, database_name="testidea", embedder=embedder)

    index.sync_chunks([first_paper, second_paper])
    embedder.calls.clear()
    index.sync_chunks([first_paper])

    collection = client.collections["chunks"]
    assert embedder.calls == []
    assert [point["metadata"]["paper_id"] for point in collection.points.values()] == ["p1"]
