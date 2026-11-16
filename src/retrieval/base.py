from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas.agent import CandidatePaper
from src.schemas.document import RetrievedContextChunk


class EmbeddingModel(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class PaperRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[CandidatePaper]: ...


@runtime_checkable
class EntityRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...


@runtime_checkable
class ContextChunkRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedContextChunk]: ...
