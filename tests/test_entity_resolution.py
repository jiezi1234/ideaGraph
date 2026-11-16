# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from src.knowledge.entity_resolver import EntityResolutionService
from src.knowledge.normalizer import normalize_relations
from src.schemas.entity import ExtractedEntity, ConceptResolutionResult
from src.schemas.relation import ExtractedRelation

class StructuredResult:
    def __init__(self, parsed, usage=None):
        self.parsed = parsed
        self.usage = usage or {}

class FakeModel:
    def __init__(self, is_same=True, canonical_id="graph neural network", merged_description="Merged GNN description"):
        self.is_same = is_same
        self.canonical_id = canonical_id
        self.merged_description = merged_description
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return StructuredResult(
            ConceptResolutionResult(
                is_same=self.is_same,
                canonical_id=self.canonical_id,
                merged_description=self.merged_description
            )
        )

class FakeRepo:
    def __init__(self, entities=None):
        self.entities = entities or []

    def list_entities(self) -> list[dict]:
        return self.entities

class FakeEmbedder:
    def __init__(self):
        self.model_name = "test-embed-model"
        self.calls = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # 根据文本内容返回虚拟的 embedding
        if "gnn" in text.lower() or "graph neural network" in text.lower():
            return [1.0, 0.0, 0.0]
        if "cnn" in text.lower() or "convolutional neural network" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

class FakeEmbeddingCache:
    def __init__(self):
        self.cache = {}

    def load(self, paper_id: str, model_name: str, abstract: str) -> list[float] | None:
        return self.cache.get((paper_id, model_name, abstract))

    def save(self, paper_id: str, model_name: str, abstract: str, embedding: list[float]) -> None:
        self.cache[(paper_id, model_name, abstract)] = embedding


def test_entity_resolver_aligns_synonyms():
    # 测试字面和嵌入对齐：将 "GNN" 合并进已有实体 "graph neural network"
    async def run_test():
        model = FakeModel(is_same=True, canonical_id="graph neural network", merged_description="A unified concept description.")
        repo = FakeRepo(entities=[
            {"id": "graph neural network", "entity_type": "Method", "description": "GNN description", "aliases": []}
        ])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()
        
        resolver = EntityResolutionService(model, repo, embedder, cache)
        
        new_entities = [
            ExtractedEntity(entity_id="gnn", entity_type="Method", description="Graph neural networks definition", source_refs=["paper-1"])
        ]
        
        resolved, id_map = await resolver.resolve_entities(new_entities)
        
        assert len(resolved) == 1
        assert resolved[0].entity_id == "graph neural network"
        assert id_map["gnn"] == "graph neural network"
        assert resolved[0].description == "A unified concept description."
        assert "gnn" in resolved[0].aliases

    asyncio.run(run_test())


def test_entity_resolver_no_match_keeps_separate():
    # 测试未对齐：如果大模型判定为不同实体，则单独创建保留
    async def run_test():
        model = FakeModel(is_same=False)
        repo = FakeRepo(entities=[
            {"id": "cnn", "entity_type": "Method", "description": "CNN description", "aliases": []}
        ])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()
        
        resolver = EntityResolutionService(model, repo, embedder, cache)
        
        new_entities = [
            ExtractedEntity(entity_id="gnn", entity_type="Method", description="Graph neural networks", source_refs=["paper-1"])
        ]
        
        resolved, id_map = await resolver.resolve_entities(new_entities)
        
        assert len(resolved) == 1
        assert resolved[0].entity_id == "gnn"
        assert id_map["gnn"] == "gnn"

    asyncio.run(run_test())


def test_entity_resolver_schema_missing_fields():
    # 测试当 canonical_id 和 merged_description 为 None/未提供时 Schema 能正常解析（不抛出 ValidationError）
    result = ConceptResolutionResult(is_same=False)
    assert result.is_same is False
    assert result.canonical_id is None
    assert result.merged_description is None


def test_entity_resolver_forces_merge_to_candidate_id_case_mismatch():
    # 测试大小写/命名不匹配时：将 "Graph Neural Network" 强制合并进已有实体 "graph neural network"
    async def run_test():
        model = FakeModel(is_same=True, canonical_id="Graph Neural Network", merged_description="Unified GNN description.")
        repo = FakeRepo(entities=[
            {"id": "graph neural network", "entity_type": "Method", "description": "GNN description", "aliases": ["gnn_alias"]}
        ])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()
        
        resolver = EntityResolutionService(model, repo, embedder, cache)
        
        new_entities = [
            ExtractedEntity(entity_id="GNN", entity_type="Method", description="Graph neural networks definition", source_refs=["paper-1"])
        ]
        
        resolved, id_map = await resolver.resolve_entities(new_entities)
        
        assert len(resolved) == 1
        # 必须是已有的 top_candidate ID，不能是 LLM 返回的 "Graph Neural Network"
        assert resolved[0].entity_id == "graph neural network"
        assert id_map["GNN"] == "graph neural network"
        assert "gnn_alias" in resolved[0].aliases
        assert "Graph Neural Network" in resolved[0].aliases
        assert "GNN" in resolved[0].aliases

    asyncio.run(run_test())


def test_entity_resolver_does_not_match_entities_created_in_same_call():
    # resolver 只对齐历史图谱实体；当前论文内部一致性由抽取模型维护。
    async def run_test():
        model = FakeModel(is_same=True, canonical_id="graph neural network", merged_description="Unified GNN description.")
        repo = FakeRepo(entities=[])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()

        resolver = EntityResolutionService(model, repo, embedder, cache)

        new_entities = [
            ExtractedEntity(entity_id="GNN", entity_type="Method", description="Abbrev description.", source_refs=["paper-1"]),
            ExtractedEntity(
                entity_id="graph neural network",
                entity_type="Method",
                description="Full-name description.",
                source_refs=["paper-1"],
            ),
        ]

        resolved, id_map = await resolver.resolve_entities(new_entities)

        assert [entity.entity_id for entity in resolved] == ["gnn", "graph neural network"]
        assert id_map["GNN"] == "gnn"
        assert id_map["graph neural network"] == "graph neural network"
        assert len(model.calls) == 0

    asyncio.run(run_test())


def test_entity_resolver_maps_normalized_relation_endpoints_to_existing_canonical_id():
    async def run_test():
        model = FakeModel(is_same=True, canonical_id="graph neural network", merged_description="Unified GNN description.")
        repo = FakeRepo(entities=[
            {"id": "graph neural network", "entity_type": "Method", "description": "GNN description", "aliases": []}
        ])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()

        resolver = EntityResolutionService(model, repo, embedder, cache)

        _, id_map = await resolver.resolve_entities(
            [ExtractedEntity(entity_id="GNN", entity_type="Method", description="Graph neural networks.", source_refs=["paper-1"])]
        )
        normalized = normalize_relations(
            [
                ExtractedRelation(
                    source_entity_id="gnn",
                    target_entity_id="idea generation",
                    relation_type="Used-For",
                    source_refs=["paper-1"],
                )
            ],
            id_map,
        )

        assert normalized[0].source_entity_id == "graph neural network"

    asyncio.run(run_test())


def test_entity_resolver_does_not_cache_unpersisted_entities_between_calls():
    # 缓存只能代表已写入图谱的实体；resolve 阶段不能提前污染候选池。
    async def run_test():
        model = FakeModel(is_same=True, canonical_id="graph neural network", merged_description="Unified GNN description.")
        repo = FakeRepo(entities=[])
        embedder = FakeEmbedder()
        cache = FakeEmbeddingCache()

        resolver = EntityResolutionService(model, repo, embedder, cache)

        await resolver.resolve_entities(
            [ExtractedEntity(entity_id="GNN", entity_type="Method", description="Abbrev description.", source_refs=["paper-1"])]
        )
        resolved, id_map = await resolver.resolve_entities(
            [
                ExtractedEntity(
                    entity_id="graph neural network",
                    entity_type="Method",
                    description="Full-name description.",
                    source_refs=["paper-2"],
                )
            ]
        )

        assert resolved[0].entity_id == "graph neural network"
        assert id_map["graph neural network"] == "graph neural network"
        assert len(model.calls) == 0

    asyncio.run(run_test())
