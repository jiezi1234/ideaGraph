from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor

from src.retrieval.base import EmbeddingModel
from src.retrieval.embedding_cache import PaperEmbeddingCache
from src.schemas.agent import CandidatePaper

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class VectorPaperRetriever:
    def __init__(
        self,
        repository,
        embedder: EmbeddingModel,
        cache: PaperEmbeddingCache,
        worker_count: int = 1,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._cache = cache
        self._worker_count = max(worker_count, 1)

    def retrieve(self, query: str, top_k: int) -> list[CandidatePaper]:
        query_embedding = self._embedder.embed(query)
        # 仓储层通常已过滤；这里防御性跳过空摘要，避免向量侧生成无价值候选。
        papers = [
            paper
            for paper in self._repository.list_papers_with_abstracts()
            if paper.abstract.strip()
        ]
        with ThreadPoolExecutor(max_workers=min(len(papers), self._worker_count) or 1) as executor:
            candidates = list(
                executor.map(
                    lambda paper: self._score_paper(query, query_embedding, paper),
                    papers,
                )
            )

        candidates.sort(key=lambda paper: paper.score, reverse=True)
        return candidates[:top_k]

    def _score_paper(self, query: str, query_embedding: list[float], paper) -> CandidatePaper:
        paper_embedding = self._cache.load(
            paper_id=paper.document_id,
            model_name=self._embedder.model_name,
            abstract=paper.abstract,
        )
        if paper_embedding is None:
            paper_embedding = self._embedder.embed(paper.abstract)
            self._cache.save(
                paper_id=paper.document_id,
                model_name=self._embedder.model_name,
                abstract=paper.abstract,
                embedding=paper_embedding,
            )
        semantic_score = _cosine_similarity(query_embedding, paper_embedding)
        score = semantic_score + _query_term_bonus(query, paper.title, paper.abstract)
        return CandidatePaper(
            paper_id=paper.document_id,
            title=paper.title,
            abstract=paper.abstract,
            cited_by_count=paper.cited_by_count,
            score=score,
        )


class VectorEntityRetriever:
    def __init__(
        self,
        repository,
        embedder: EmbeddingModel,
        cache: PaperEmbeddingCache,
        worker_count: int = 1,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._cache = cache
        self._worker_count = max(worker_count, 1)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        query_embedding = self._embedder.embed(query)
        entities = [
            entity
            for entity in self._repository.list_entities()
            if entity.get("id") and entity.get("entity_type") != "Paper"
        ]
        with ThreadPoolExecutor(max_workers=min(len(entities), self._worker_count) or 1) as executor:
            candidates = list(
                executor.map(
                    lambda entity: self._score_entity(query, query_embedding, entity),
                    entities,
                )
            )

        candidates.sort(key=lambda entity: entity["score"], reverse=True)
        return candidates[:top_k]

    def _score_entity(self, query: str, query_embedding: list[float], entity: dict) -> dict:
        entity_id = entity["id"]
        entity_type = entity.get("entity_type")
        text = _entity_embedding_text(entity_id, entity_type)
        cache_id = f"entity:{entity_id}"
        entity_embedding = self._cache.load(
            paper_id=cache_id,
            model_name=self._embedder.model_name,
            abstract=text,
        )
        if entity_embedding is None:
            entity_embedding = self._embedder.embed(text)
            self._cache.save(
                paper_id=cache_id,
                model_name=self._embedder.model_name,
                abstract=text,
                embedding=entity_embedding,
            )
        semantic_score = _cosine_similarity(query_embedding, entity_embedding)
        score = semantic_score + _query_term_bonus(query, entity_id, entity_type or "")
        return {
            "id": entity_id,
            "entity_type": entity_type,
            "score": score,
        }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _query_term_bonus(query: str, title: str, abstract: str) -> float:
    query_terms = _token_set(query)
    if not query_terms:
        return 0.0

    title_terms = _token_set(title)
    abstract_terms = _token_set(abstract)
    title_matches = len(query_terms & title_terms)
    abstract_matches = len(query_terms & abstract_terms)
    return min(0.12, title_matches * 0.04 + abstract_matches * 0.02)


def _token_set(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS
    }


def _normalize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _entity_embedding_text(entity_id: str, entity_type: str | None) -> str:
    if entity_type:
        return f"{entity_id}\n{entity_type}"
    return entity_id
