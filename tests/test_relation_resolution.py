from __future__ import annotations

import asyncio

from src.knowledge.relation_resolver import RelationResolutionService
from src.schemas.relation import ExtractedRelation, RelationResolutionResult


class StructuredResult:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeModel:
    def __init__(self, merged_description: str = "Merged relation description."):
        self.merged_description = merged_description
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return StructuredResult(RelationResolutionResult(merged_description=self.merged_description))


class FakeRepo:
    def __init__(self, relation=None):
        self.relation = relation
        self.calls = []

    def get_entity_relation(self, source_entity_id: str, relation_type: str, target_entity_id: str):
        self.calls.append((source_entity_id, relation_type, target_entity_id))
        return self.relation


def test_relation_resolver_merges_existing_relation_description_with_model():
    async def run_test():
        model = FakeModel(merged_description="Merged graph relation evidence.")
        repo = FakeRepo(
            relation={
                "description": "Existing relation evidence.",
                "source_refs": ["paper-1"],
            }
        )
        resolver = RelationResolutionService(model=model, repository=repo)

        resolved = await resolver.resolve_relations(
            [
                ExtractedRelation(
                    source_entity_id="graph neural network",
                    relation_type="Used-For",
                    target_entity_id="idea generation",
                    description="Current relation evidence.",
                    source_refs=["paper-2"],
                )
            ]
        )

        assert len(resolved) == 1
        assert resolved[0].description == "Merged graph relation evidence."
        assert resolved[0].source_refs == ["paper-1", "paper-2"]
        assert resolved[0].description_update_mode == "replace"
        assert repo.calls == [("graph neural network", "Used-For", "idea generation")]
        assert model.calls[0][1] is RelationResolutionResult

    asyncio.run(run_test())


def test_relation_resolver_keeps_new_relation_without_model_call():
    async def run_test():
        model = FakeModel()
        repo = FakeRepo(relation=None)
        resolver = RelationResolutionService(model=model, repository=repo)

        relation = ExtractedRelation(
            source_entity_id="graph neural network",
            relation_type="Used-For",
            target_entity_id="idea generation",
            description="Current relation evidence.",
            source_refs=["paper-2"],
        )
        resolved = await resolver.resolve_relations([relation])

        assert resolved == [relation]
        assert model.calls == []

    asyncio.run(run_test())


def test_relation_resolver_falls_back_to_append_mode_when_model_fails():
    class FailingModel(FakeModel):
        def structured_json(self, *, messages, schema, name: str, metadata=None):
            raise RuntimeError("model unavailable")

    async def run_test():
        repo = FakeRepo(
            relation={
                "description": "Existing relation evidence.",
                "source_refs": ["paper-1"],
            }
        )
        resolver = RelationResolutionService(model=FailingModel(), repository=repo)

        resolved = await resolver.resolve_relations(
            [
                ExtractedRelation(
                    source_entity_id="graph neural network",
                    relation_type="Used-For",
                    target_entity_id="idea generation",
                    description="Current relation evidence.",
                    source_refs=["paper-2"],
                )
            ]
        )

        assert resolved[0].description == "Current relation evidence."
        assert resolved[0].source_refs == ["paper-2"]
        assert resolved[0].description_update_mode == "append"

    asyncio.run(run_test())
