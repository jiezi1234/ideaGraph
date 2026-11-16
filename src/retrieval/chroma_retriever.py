from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Protocol

import chromadb
from transformers import AutoTokenizer

from src.retrieval.base import ContextChunkRetriever, EntityRetriever, PaperRetriever
from src.schemas.agent import CandidatePaper
from src.schemas.document import RetrievedContextChunk, RetrievedPaperDocument


CHUNK_MIN_TOKENS = 256
CHUNK_MAX_TOKENS = 2048
CHUNK_OVERLAP_TOKENS = 120


class ChunkTokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def split_text(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]: ...


class WhitespaceChunkTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def split_text(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokens = re.findall(r"\S+", text)
        if len(tokens) <= max_tokens:
            return [text.strip()]
        step = max(max_tokens - overlap_tokens, 1)
        parts = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + max_tokens]
            if not window:
                continue
            parts.append(" ".join(window))
            if start + max_tokens >= len(tokens):
                break
        return parts


class HuggingFaceChunkTokenizer:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_model_path(cls, model_path: str, *, local_files_only: bool = True) -> "HuggingFaceChunkTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            use_fast=True,
        )
        return cls(tokenizer)

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def split_text(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
        token_ids = self._encode(text)
        if len(token_ids) <= max_tokens:
            return [text.strip()]
        step = max(max_tokens - overlap_tokens, 1)
        parts = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + max_tokens]
            if not window:
                continue
            decoded = self._tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if decoded:
                parts.append(decoded)
            if start + max_tokens >= len(token_ids):
                break
        return parts

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


def chroma_persist_path(database_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in database_name)
    return Path(".cache") / "chroma" / safe_name


def build_chroma_client(database_name: str):
    return chromadb.PersistentClient(path=str(chroma_persist_path(database_name)))


class ChromaVectorIndex:
    def __init__(
        self,
        client,
        database_name: str,
        embedder,
        tokenizer: ChunkTokenizer | None = None,
        chunk_min_tokens: int = CHUNK_MIN_TOKENS,
        chunk_max_tokens: int = CHUNK_MAX_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._client = client
        self._database_name = database_name
        self._embedder = embedder
        self._tokenizer = tokenizer or WhitespaceChunkTokenizer()
        self._chunk_min_tokens = chunk_min_tokens
        self._chunk_max_tokens = chunk_max_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._papers = client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )
        self._entities = client.get_or_create_collection(
            name="entities",
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks = client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def sync_papers(self, papers: Iterable[RetrievedPaperDocument]) -> None:
        existing_metadata = self._existing_metadata(self._papers)
        active_ids: set[str] = set()
        upsert_ids = []
        upsert_documents = []
        upsert_embeddings = []
        upsert_metadatas = []
        for paper in papers:
            if not _has_abstract(paper):
                continue
            document = _paper_embedding_text(paper)
            point_id = _point_id("paper", paper.document_id)
            active_ids.add(point_id)
            metadata = {
                "kind": "paper",
                "neo4j_database": self._database_name,
                "paper_id": paper.document_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "cited_by_count": paper.cited_by_count or 0,
                "text_hash": _text_hash(document),
            }
            if existing_metadata.get(point_id, {}).get("text_hash") == metadata["text_hash"]:
                continue
            upsert_ids.append(point_id)
            upsert_documents.append(document)
            upsert_embeddings.append(self._embedder.embed(document))
            upsert_metadatas.append(metadata)
        self._delete_stale(self._papers, active_ids)
        if upsert_ids:
            self._papers.upsert(
                ids=upsert_ids,
                documents=upsert_documents,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
            )

    def sync_entities(self, entities: Iterable[dict]) -> None:
        existing_metadata = self._existing_metadata(self._entities)
        active_ids: set[str] = set()
        upsert_ids = []
        upsert_documents = []
        upsert_embeddings = []
        upsert_metadatas = []
        for entity in entities:
            entity_id = entity.get("id")
            entity_type = entity.get("entity_type") or ""
            if not entity_id or entity_type == "Paper":
                continue
            description = entity.get("description") or ""
            document = _entity_embedding_text(entity_id, entity_type, description)
            point_id = _point_id("entity", entity_id)
            active_ids.add(point_id)
            metadata = {
                "kind": "entity",
                "neo4j_database": self._database_name,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "description": description,
                "text_hash": _text_hash(document),
            }
            if "_distance" in entity:
                metadata["_distance"] = entity["_distance"]
            if existing_metadata.get(point_id, {}).get("text_hash") == metadata["text_hash"]:
                continue
            upsert_ids.append(point_id)
            upsert_documents.append(document)
            upsert_embeddings.append(self._embedder.embed(document))
            upsert_metadatas.append(metadata)
        self._delete_stale(self._entities, active_ids)
        if upsert_ids:
            self._entities.upsert(
                ids=upsert_ids,
                documents=upsert_documents,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
            )

    def sync_chunks(self, papers: Iterable[RetrievedPaperDocument]) -> None:
        existing_metadata = self._existing_metadata(self._chunks)
        active_ids: set[str] = set()
        upsert_ids = []
        upsert_documents = []
        upsert_embeddings = []
        upsert_metadatas = []
        for paper in papers:
            # chunk 向量以“全文优先、摘要兜底”为统一口径；不能因为摘要为空跳过已有全文的论文。
            for chunk in _paper_chunks(
                paper,
                tokenizer=self._tokenizer,
                min_tokens=self._chunk_min_tokens,
                max_tokens=self._chunk_max_tokens,
                overlap_tokens=self._chunk_overlap_tokens,
            ):
                document = _chunk_embedding_text(paper.title, chunk["text"])
                text_hash = _text_hash(document)
                chunk_id = _chunk_id(paper.document_id, chunk["chunk_index"], text_hash)
                point_id = _point_id("chunk", chunk_id)
                active_ids.add(point_id)
                metadata = {
                    "kind": "chunk",
                    "neo4j_database": self._database_name,
                    "paper_id": paper.document_id,
                    "title": paper.title,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk["chunk_index"],
                    "paragraph_start": chunk["paragraph_start"],
                    "paragraph_end": chunk["paragraph_end"],
                    "text_source": chunk["text_source"],
                    "text_hash": text_hash,
                }
                if existing_metadata.get(point_id, {}).get("text_hash") == metadata["text_hash"]:
                    continue
                upsert_ids.append(point_id)
                upsert_documents.append(document)
                upsert_embeddings.append(self._embedder.embed(document))
                upsert_metadatas.append(metadata)
        self._delete_stale(self._chunks, active_ids)
        if upsert_ids:
            self._chunks.upsert(
                ids=upsert_ids,
                documents=upsert_documents,
                embeddings=upsert_embeddings,
                metadatas=upsert_metadatas,
            )

    def search_papers(self, query: str, top_k: int) -> list[dict]:
        return self._query(self._papers, query=query, top_k=top_k, where=None)

    def search_entities(self, query: str, top_k: int, entity_type: str | None = None) -> list[dict]:
        where = {"entity_type": entity_type} if entity_type else None
        return self._query(self._entities, query=query, top_k=top_k, where=where)

    def search_chunks(self, query: str, top_k: int) -> list[dict]:
        return self._query(self._chunks, query=query, top_k=top_k, where=None)

    def _query(self, collection, *, query: str, top_k: int, where: dict | None) -> list[dict]:
        if top_k <= 0:
            return []
        query_embedding = self._embedder.embed(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        ids = _first_result_list(result.get("ids"))
        metadatas = _first_result_list(result.get("metadatas"))
        documents = _first_result_list(result.get("documents"))
        distances = _first_result_list(result.get("distances"))
        rows = []
        for index, point_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            rows.append(
                {
                    "id": point_id,
                    "metadata": dict(metadatas[index] if index < len(metadatas) else {}),
                    "document": documents[index] if index < len(documents) else "",
                    "score": _score_from_distance(distance),
                    "distance": distance,
                }
            )
        return rows

    def _existing_metadata(self, collection) -> dict[str, dict]:
        existing = collection.get(include=["metadatas"])
        return {
            point_id: metadata
            for point_id, metadata in zip(existing.get("ids") or [], existing.get("metadatas") or [], strict=False)
            if metadata.get("neo4j_database") == self._database_name
        }

    def _delete_stale(self, collection, active_ids: set[str]) -> None:
        existing = collection.get(include=["metadatas"])
        stale_ids = []
        for point_id, metadata in zip(existing.get("ids") or [], existing.get("metadatas") or [], strict=False):
            if metadata.get("neo4j_database") != self._database_name:
                continue
            if point_id not in active_ids:
                stale_ids.append(point_id)
        if stale_ids:
            collection.delete(ids=stale_ids)


class ChromaPaperRetriever(PaperRetriever):
    def __init__(self, repository, index: ChromaVectorIndex) -> None:
        self._repository = repository
        self._index = index

    def retrieve(self, query: str, top_k: int) -> list[CandidatePaper]:
        papers = list(self._repository.list_papers_with_abstracts())
        self._index.sync_papers(papers)
        rows = self._index.search_papers(query, top_k=top_k)
        return [
            CandidatePaper(
                paper_id=row["metadata"].get("paper_id") or row["id"],
                title=row["metadata"].get("title") or row["metadata"].get("paper_id") or row["id"],
                abstract=row["metadata"].get("abstract") or "",
                cited_by_count=_optional_int(row["metadata"].get("cited_by_count")),
                score=row["score"],
            )
            for row in rows
            if (row["metadata"].get("abstract") or "").strip()
        ]


class ChromaEntityRetriever(EntityRetriever):
    def __init__(self, repository, index: ChromaVectorIndex) -> None:
        self._repository = repository
        self._index = index

    def retrieve(self, query: str, top_k: int, entity_type: str | None = None) -> list[dict]:
        entities = list(self._repository.list_entities())
        self._index.sync_entities(entities)
        rows = self._index.search_entities(query, top_k=top_k, entity_type=entity_type)
        return [
            {
                "id": row["metadata"].get("entity_id") or row["id"],
                "entity_type": row["metadata"].get("entity_type"),
                "score": row["score"],
            }
            for row in rows
        ]


class ChromaChunkRetriever(ContextChunkRetriever):
    def __init__(self, repository, index: ChromaVectorIndex) -> None:
        self._repository = repository
        self._index = index

    def retrieve(self, query: str, top_k: int) -> list[RetrievedContextChunk]:
        papers = list(self._repository.list_papers_for_vector_index())
        self._index.sync_chunks(papers)
        rows = self._index.search_chunks(query, top_k=top_k)
        return [
            RetrievedContextChunk(
                paper_id=row["metadata"].get("paper_id") or "",
                title=row["metadata"].get("title") or row["metadata"].get("paper_id") or "",
                chunk_id=row["metadata"].get("chunk_id") or row["id"],
                chunk_index=_optional_int(row["metadata"].get("chunk_index")) or 0,
                paragraph_start=_optional_int(row["metadata"].get("paragraph_start")) or 0,
                paragraph_end=_optional_int(row["metadata"].get("paragraph_end")) or 0,
                score=row["score"],
                text_source=row["metadata"].get("text_source") or "title_abstract",
                text=row["document"],
            )
            for row in rows
        ]


def _paper_embedding_text(paper: RetrievedPaperDocument) -> str:
    return f"Title: {paper.title}\nAbstract: {paper.abstract}"


def _has_abstract(paper: RetrievedPaperDocument) -> bool:
    # 读取侧统一丢弃无摘要论文，避免向量索引残留只有标题的 Paper 噪声。
    return bool(paper.abstract.strip())


def _chunk_embedding_text(title: str, chunk_text: str) -> str:
    return f"Title: {title}\nChunk:\n{chunk_text}"


def _entity_embedding_text(entity_id: str, entity_type: str, description: str) -> str:
    return f"Entity: {entity_id}\nType: {entity_type}\nDescription: {description}"


def _point_id(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()


def _chunk_id(paper_id: str, chunk_index: int, text_hash: str) -> str:
    return hashlib.sha256(f"{paper_id}:{chunk_index}:{text_hash}".encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _score_from_distance(distance) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def _first_result_list(value) -> list:
    if not value:
        return []
    first = value[0]
    return list(first or [])


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _paper_chunks(
    paper: RetrievedPaperDocument,
    *,
    tokenizer: ChunkTokenizer | None = None,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    source_text, text_source = _paper_chunk_source(paper)
    return chunk_markdown_text(
        source_text,
        text_source=text_source,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def chunk_markdown_text(
    markdown: str,
    *,
    text_source: str,
    tokenizer: ChunkTokenizer | None = None,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    tokenizer = tokenizer or WhitespaceChunkTokenizer()
    source_text = markdown
    paragraphs = _markdown_paragraphs(source_text)
    chunks = []
    index = 0
    chunk_index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        token_count = tokenizer.count_tokens(paragraph)
        if token_count > max_tokens:
            for part in tokenizer.split_text(paragraph, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "paragraph_start": index,
                        "paragraph_end": index,
                        "text_source": text_source,
                        "text": part,
                    }
                )
                chunk_index += 1
            index += 1
            continue

        start = index
        parts = [paragraph]
        total_tokens = token_count
        index += 1
        while index < len(paragraphs):
            next_paragraph = paragraphs[index]
            next_text = "\n\n".join([*parts, next_paragraph])
            next_tokens = tokenizer.count_tokens(next_text)
            # 段落组尽量贴近上限，但不能为了凑满而跨过模型上下文预算。
            if next_tokens > max_tokens:
                break
            parts.append(next_paragraph)
            total_tokens = next_tokens
            index += 1
            if total_tokens >= max(min_tokens, max_tokens):
                break
        chunks.append(
            {
                "chunk_index": chunk_index,
                "paragraph_start": start,
                "paragraph_end": index - 1,
                "text_source": text_source,
                "text": "\n\n".join(parts),
            }
        )
        chunk_index += 1
    return chunks


def _paper_chunk_source(paper: RetrievedPaperDocument) -> tuple[str, str]:
    markdown_path = Path(paper.full_text_markdown_path) if paper.full_text_markdown_path else None
    if markdown_path is not None and markdown_path.exists():
        content = markdown_path.read_text(encoding="utf-8").strip()
        if content:
            return content, paper.text_source or "full_text_md"
    if paper.abstract.strip():
        return f"Abstract: {paper.abstract.strip()}", "title_abstract"
    return "", paper.text_source or "title_abstract"


def _markdown_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]
