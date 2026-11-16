from __future__ import annotations


EMPTY_GRAPH_CONTEXT = {
    "entities": [],
    "relationships": [],
    "page": 1,
    "page_size": 20,
    "total_entities": 0,
    "total_relationships": 0,
}


def empty_graph_context() -> dict:
    return dict(EMPTY_GRAPH_CONTEXT)


def has_graph_context(graph_context: dict) -> bool:
    return bool(
        graph_context.get("entities")
        or graph_context.get("relationships")
        or graph_context.get("lookup_notes")
    )


def merge_graph_contexts(contexts: list[dict]) -> dict:
    if not contexts:
        return empty_graph_context()
    entities = []
    relationships = []
    lookup_notes = []
    seen_entities = set()
    seen_relationships = set()
    seen_lookup_notes = set()
    total_entities = 0
    total_relationships = 0
    for context in contexts:
        total_entities += int(context.get("total_entities", 0) or 0)
        total_relationships += int(context.get("total_relationships", 0) or 0)
        for note in context.get("lookup_notes", []):
            key = (
                note.get("status"),
                note.get("requested_node_id"),
                note.get("resolved_node_id"),
            )
            if key in seen_lookup_notes:
                continue
            seen_lookup_notes.add(key)
            lookup_notes.append(note)
        for entity in context.get("entities", []):
            entity_id = entity.get("id")
            if not entity_id or entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            entities.append(entity)
        for relationship in context.get("relationships", []):
            key = (
                relationship.get("source_id"),
                relationship.get("target_id"),
                relationship.get("relationship_type"),
            )
            if key in seen_relationships:
                continue
            seen_relationships.add(key)
            relationships.append(relationship)
    merged = {
        "entities": entities,
        "relationships": relationships,
        "page": 1,
        "page_size": 20,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
    }
    if lookup_notes:
        merged["lookup_notes"] = lookup_notes
    return merged
