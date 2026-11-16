from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def merge_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    merged: dict[str, ExtractedEntity] = {}
    for entity in entities:
        normalized_id = _normalize_name(entity.entity_id)
        if normalized_id not in merged:
            merged[normalized_id] = entity.model_copy(
                update={
                    "entity_id": normalized_id,
                    "aliases": [entity.entity_id] if entity.entity_id != normalized_id else [],
                }
            )
            continue
        current = merged[normalized_id]
        current.source_refs = sorted(set(current.source_refs + entity.source_refs))
        current.aliases = sorted(set(current.aliases + [entity.entity_id]))
        current.description = _merge_description(current.description, entity.description)
    return list(merged.values())


def normalize_relations(relations: list[ExtractedRelation], id_map: dict[str, str]) -> list[ExtractedRelation]:
    normalized: list[ExtractedRelation] = []
    for relation in relations:
        normalized.append(
            relation.model_copy(
                update={
                    "source_entity_id": id_map.get(relation.source_entity_id, _normalize_name(relation.source_entity_id)),
                    "target_entity_id": id_map.get(relation.target_entity_id, _normalize_name(relation.target_entity_id)),
                }
            )
        )
    return normalized


def _merge_description(left: str, right: str) -> str:
    descriptions = []
    seen = set()
    for description in [left, right]:
        normalized = " ".join((description or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        descriptions.append(normalized)
    return "\n".join(descriptions)
