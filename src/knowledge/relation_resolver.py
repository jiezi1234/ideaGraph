# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

from src.prompts.relation_resolution import PROMPT as RELATION_RESOLUTION_PROMPT
from src.schemas.relation import ExtractedRelation, RelationResolutionResult

logger = logging.getLogger(__name__)


def _merge_refs(left: list[str], right: list[str]) -> list[str]:
    refs = []
    for ref in [*left, *right]:
        if ref and ref not in refs:
            refs.append(ref)
    return refs


class RelationResolutionService:
    def __init__(self, model, repository):
        self._model = model
        self._repository = repository

    async def resolve_relations(self, relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
        resolved: list[ExtractedRelation] = []
        for relation in relations:
            existing = self._repository.get_entity_relation(
                source_entity_id=relation.source_entity_id,
                relation_type=relation.relation_type,
                target_entity_id=relation.target_entity_id,
            )
            if existing is None:
                resolved.append(relation)
                continue

            existing_description = existing.get("description") or ""
            if not existing_description or not relation.description:
                resolved.append(relation)
                continue

            try:
                result = self._model.structured_json(
                    messages=[
                        {"role": "system", "content": RELATION_RESOLUTION_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"source_entity_id: {relation.source_entity_id}\n"
                                f"relation_type: {relation.relation_type}\n"
                                f"target_entity_id: {relation.target_entity_id}\n\n"
                                f"已有关系描述：\n{existing_description}\n\n"
                                f"当前论文关系描述：\n{relation.description}\n"
                            ),
                        },
                    ],
                    schema=RelationResolutionResult,
                    name="relation_resolution",
                    metadata={"stage": "relation_resolution"},
                )
            except Exception as exc:
                logger.warning(
                    "关系描述融合失败，退化为追加写入: %s -[%s]-> %s, error=%s",
                    relation.source_entity_id,
                    relation.relation_type,
                    relation.target_entity_id,
                    exc,
                )
                resolved.append(relation)
                continue

            resolved.append(
                relation.model_copy(
                    update={
                        "description": result.parsed.merged_description,
                        "source_refs": _merge_refs(existing.get("source_refs") or [], relation.source_refs),
                        "description_update_mode": "replace",
                    }
                )
            )
        return resolved
