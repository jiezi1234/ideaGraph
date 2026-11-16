from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Settings
from src.models.openai_embeddings import OpenAIEmbeddingModel
from src.retrieval.embedding_cache import PaperEmbeddingCache
from src.retrieval.vector_retriever import _cosine_similarity, _entity_embedding_text
from src.storage.graph.neo4j_repository import Neo4jGraphRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank non-Paper graph entities by embedding cosine similarity.",
    )
    parser.add_argument("--query", default="进化神经网络")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def cached_embedding(
    *,
    cache: PaperEmbeddingCache,
    embedder: OpenAIEmbeddingModel,
    cache_id: str,
    text: str,
) -> tuple[list[float], bool]:
    embedding = cache.load(
        paper_id=cache_id,
        model_name=embedder.model_name,
        abstract=text,
    )
    if embedding is not None:
        return embedding, True
    legacy_embedding = load_legacy_embedding(
        cache_dir=cache._cache_dir,
        paper_id=cache_id,
        model_name=embedder.model_name,
        text=text,
    )
    if legacy_embedding is not None:
        return legacy_embedding, True
    embedding = embedder.embed(text)
    cache.save(
        paper_id=cache_id,
        model_name=embedder.model_name,
        abstract=text,
        embedding=embedding,
    )
    return embedding, False


def load_legacy_embedding(
    *,
    cache_dir: Path,
    paper_id: str,
    model_name: str,
    text: str,
) -> list[float] | None:
    digest = hashlib.sha256(f"{model_name}:{paper_id}".encode("utf-8")).hexdigest()
    path = cache_dir / model_name / f"{digest}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if payload.get("abstract_hash") != text_hash:
        return None
    return list(payload["embedding"])


def main() -> None:
    args = parse_args()
    settings = Settings()
    repository = Neo4jGraphRepository.from_connection(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    embedder = OpenAIEmbeddingModel(
        base_url=settings.resolved_embed_base_url,
        api_key=settings.resolved_embed_api_key,
        model_name=settings.embed_model,
        timeout=settings.llm_timeout,
    )
    cache = PaperEmbeddingCache(cache_dir=settings.paper_embedding_cache_dir)

    query_embedding, query_cache_hit = cached_embedding(
        cache=cache,
        embedder=embedder,
        cache_id=f"query:{args.query}",
        text=args.query,
    )

    entities = []
    for entity in repository.list_entities():
        entity_id = entity.get("id")
        if not entity_id:
            continue
        entities.append(entity)

    def score_entity(entity: dict) -> tuple[dict, bool]:
        entity_id = entity["id"]
        entity_type = entity.get("entity_type")
        text = _entity_embedding_text(entity_id, entity_type)
        entity_embedding, cache_hit = cached_embedding(
            cache=cache,
            embedder=embedder,
            cache_id=f"entity:{entity_id}",
            text=text,
        )
        return (
            {
                "id": entity_id,
                "entity_type": entity_type or "",
                "embedding_text": text,
                "cosine": _cosine_similarity(query_embedding, entity_embedding),
            },
            cache_hit,
        )

    with ThreadPoolExecutor(max_workers=min(len(entities), settings.embedding_worker_count) or 1) as executor:
        results = list(executor.map(score_entity, entities))

    scored = [item for item, _ in results]
    cache_hits = sum(1 for _, cache_hit in results if cache_hit)
    cache_misses = len(results) - cache_hits
    scored.sort(key=lambda item: item["cosine"], reverse=True)
    print(f"query: {args.query}")
    print(f"neo4j_database: {settings.neo4j_database}")
    print(f"embedding_model: {settings.embed_model}")
    print(f"worker_count: {settings.embedding_worker_count}")
    print(f"entity_count: {len(scored)}")
    print(f"query_cache_hit: {query_cache_hit}")
    print(f"entity_cache_hits: {cache_hits}")
    print(f"entity_cache_misses: {cache_misses}")
    print()
    for index, item in enumerate(scored[: args.top_k], start=1):
        print(
            f"{index}. cosine={item['cosine']:.6f} "
            f"type={item['entity_type']} id={item['id']}"
        )
        print(f"   embedding_text={item['embedding_text']!r}")


if __name__ == "__main__":
    main()
