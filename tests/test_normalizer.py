from src.knowledge.normalizer import merge_entities, normalize_relations
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def test_merge_entities_merges_same_entity_id_case_insensitively():
    entities = [
        ExtractedEntity(entity_id="Graph Reasoning", entity_type="Concept", source_refs=["d1"], description="Uses graph paths."),
        ExtractedEntity(entity_id="graph reasoning", entity_type="Concept", source_refs=["d2"], description="Supports inference."),
    ]

    merged = merge_entities(entities)

    assert len(merged) == 1
    assert merged[0].entity_id == "graph reasoning"
    assert sorted(merged[0].source_refs) == ["d1", "d2"]
    assert merged[0].description == "Uses graph paths.\nSupports inference."


def test_normalize_relations_rewrites_ids_to_merged_entity_ids():
    relations = [
        ExtractedRelation(
            source_entity_id="Graph Reasoning",
            target_entity_id="Idea Generation",
            relation_type="Used-For",
            source_refs=["d1"],
        )
    ]
    id_map = {
        "Graph Reasoning": "graph reasoning",
        "Idea Generation": "idea generation",
    }

    normalized = normalize_relations(relations, id_map)

    assert normalized[0].source_entity_id == "graph reasoning"
    assert normalized[0].target_entity_id == "idea generation"
