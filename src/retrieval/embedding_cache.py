from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PaperEmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)

    def load(self, paper_id: str, model_name: str, abstract: str) -> list[float] | None:
        path = self._path_for(paper_id, model_name)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("abstract_hash") != self._abstract_hash(abstract):
            return None
        return list(payload["embedding"])

    def save(self, paper_id: str, model_name: str, abstract: str, embedding: list[float]) -> None:
        path = self._path_for(paper_id, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_id": paper_id,
            "model_name": model_name,
            "abstract_hash": self._abstract_hash(abstract),
            "embedding": embedding,
        }
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _path_for(self, paper_id: str, model_name: str) -> Path:
        digest = hashlib.sha256(f"{model_name}:{paper_id}".encode("utf-8")).hexdigest()
        safe_model_name = self._safe_component(model_name)
        return self._cache_dir / safe_model_name / f"{digest}.json"

    @staticmethod
    def _safe_component(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)

    def _abstract_hash(self, abstract: str) -> str:
        return hashlib.sha256(abstract.encode("utf-8")).hexdigest()
