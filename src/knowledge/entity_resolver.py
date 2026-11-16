# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import difflib
import math
from src.knowledge.normalizer import merge_entities
from src.schemas.entity import ExtractedEntity, ConceptResolutionResult
from src.prompts.entity_resolution import PROMPT as ENTITY_RESOLUTION_PROMPT

logger = logging.getLogger(__name__)


def _normalize_name(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


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


def _add_id_mapping(id_map: dict[str, str], value: str, canon_id: str) -> None:
    if value:
        id_map[value] = canon_id
    normalized = _normalize_name(value)
    if normalized:
        id_map[normalized] = canon_id


class EntityResolutionService:
    def __init__(self, model, repository, embedder, embedding_cache):
        self._model = model
        self._repository = repository
        self._embedder = embedder
        self._cache = embedding_cache
        # 内存缓存避免频繁查 Neo4j；单篇论文处理时会先拍快照，避免论文内重消歧。
        self._entities_cache = None

    def _get_all_entities(self) -> list[dict]:
        if self._entities_cache is None:
            # 启动时一次性载入已有实体，便于快速做相似度对比
            self._entities_cache = self._repository.list_entities()
        return self._entities_cache

    def _add_to_cache(self, entity_id: str, entity_type: str, description: str, aliases: list[str]):
        if self._entities_cache is None:
            self._entities_cache = []
        for ent in self._entities_cache:
            if ent["id"] == entity_id:
                ent["description"] = description
                # 合并别名
                ent["aliases"] = sorted(list(set(ent.get("aliases", []) + aliases)))
                return
        self._entities_cache.append({
            "id": entity_id,
            "entity_type": entity_type,
            "description": description,
            "aliases": aliases
        })

    def remember_entities(self, entities: list[ExtractedEntity]) -> None:
        for entity in entities:
            self._add_to_cache(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=entity.aliases,
            )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        return dot / (left_norm * right_norm)

    def _get_entity_embedding(self, entity_id: str, entity_type: str) -> list[float]:
        text = f"{entity_id}\n{entity_type}"
        cache_id = f"entity:{entity_id}"
        # 复用项目自带的向量缓存机制，避免重复调用 Embedding 接口
        emb = self._cache.load(
            paper_id=cache_id,
            model_name=self._embedder.model_name,
            abstract=text
        )
        if emb is None:
            emb = self._embedder.embed(text)
            self._cache.save(
                paper_id=cache_id,
                model_name=self._embedder.model_name,
                abstract=text,
                embedding=emb
            )
        return emb

    async def resolve_entities(self, entities: list[ExtractedEntity]) -> tuple[list[ExtractedEntity], dict[str, str]]:
        """
        在线实体对齐核心逻辑：
        1. 粗筛：比对内存缓存中的实体，计算字面距离 (SequenceMatcher) 和语义嵌入距离。
        2. 精筛：对相似候选调用大模型，判定是否合并，并融合描述。
        3. 返回映射字典 `{ 原始ID: 规范ID }` 与对齐后的新实体列表。
        """
        id_map = {}
        resolved_entities = []
        
        # 只用本篇论文开始处理前的图谱快照做候选，避免当前论文内部实体互相重消歧。
        existing_list = [
            {
                "id": existing.get("id"),
                "entity_type": existing.get("entity_type"),
                "description": existing.get("description") or "",
                "aliases": list(existing.get("aliases") or []),
            }
            for existing in self._get_all_entities()
            if existing.get("id")
        ]
        
        for ent in merge_entities(entities):
            # 归一化去空格
            normalized_name = _normalize_name(ent.entity_id)
            if not normalized_name:
                continue
                
            candidates = []
            
            # 第一阶段：轻量表征粗筛
            for existing in existing_list:
                existing_id = existing["id"]
                # A. 如果字面完全一致
                if _normalize_name(existing_id) == normalized_name:
                    candidates.append((existing, 1.0))
                    continue
                    
                # B. 如果别名中已经存在
                if any(_normalize_name(alias) == normalized_name for alias in existing.get("aliases") or []):
                    candidates.append((existing, 1.0))
                    continue

                # C. 字面相似度 (编辑距离)
                edit_sim = difflib.SequenceMatcher(None, normalized_name, existing_id.lower()).ratio()
                if edit_sim > 0.8:
                    candidates.append((existing, edit_sim))
                    continue
                
                # D. 语义向量相似度
                try:
                    ent_emb = self._get_entity_embedding(normalized_name, ent.entity_type)
                    ex_emb = self._get_entity_embedding(existing_id, existing.get("entity_type") or "")
                    sem_sim = self._cosine_similarity(ent_emb, ex_emb)
                    if sem_sim > 0.85:
                        candidates.append((existing, sem_sim))
                except Exception as ex:
                    logger.warning("Error computing embedding similarity for %s and %s: %s", normalized_name, existing_id, ex)
            
            matched_canon = None
            if candidates:
                # 降序排列，取相似度最高的一个进行 LLM 判断
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidate = candidates[0][0]
                
                # 第二阶段：大模型精筛与融合
                try:
                    result = self._model.structured_json(
                        messages=[
                            {"role": "system", "content": ENTITY_RESOLUTION_PROMPT},
                            {
                                "role": "user",
                                "content": (
                                    f"待判定实体：\n"
                                    f"ID: {normalized_name}\n"
                                    f"Type: {ent.entity_type}\n"
                                    f"Description: {ent.description}\n\n"
                                    f"候选匹配实体：\n"
                                    f"ID: {top_candidate['id']}\n"
                                    f"Type: {top_candidate.get('entity_type')}\n"
                                    f"Description: {top_candidate.get('description')}\n"
                                )
                            }
                        ],
                        schema=ConceptResolutionResult,
                        name="entity_resolution",
                        metadata={"stage": "entity_resolution"}
                    )
                    
                    if result.parsed.is_same and result.parsed.canonical_id:
                        matched_canon = result.parsed
                        logger.info("在线对齐成功: %s -> %s", normalized_name, matched_canon.canonical_id)
                except Exception as ex:
                    logger.warning("大模型对齐判定失败，退化为独立写入。错误: %s", ex)
            
            if matched_canon:
                canon_id = top_candidate["id"]
                
                # 合并别名 (包含已有候选别名、新提取实体ID/归一化ID以及LLM建议的规范ID)
                existing_aliases = top_candidate.get("aliases") or []
                aliases = sorted(list(set(
                    [normalized_name, ent.entity_id, canon_id]
                    + ent.aliases
                    + existing_aliases 
                    + ([matched_canon.canonical_id] if matched_canon.canonical_id else [])
                )))
                for alias in aliases:
                    _add_id_mapping(id_map, alias, canon_id)
                
                desc = matched_canon.merged_description or _merge_description(
                    top_candidate.get("description") or "",
                    ent.description,
                )
                resolved_ent = ExtractedEntity(
                    entity_id=canon_id,
                    entity_type=ent.entity_type,
                    description=desc,
                    source_refs=sorted(list(set(ent.source_refs))),
                    aliases=aliases,
                    description_update_mode="replace",
                )
                resolved_entities.append(resolved_ent)
            else:
                # 判定为独立的新概念，规范 ID 采用 lowercase
                canon_id = normalized_name
                
                aliases = sorted(list(set([normalized_name, ent.entity_id] + ent.aliases)))
                for alias in aliases:
                    _add_id_mapping(id_map, alias, canon_id)
                
                resolved_ent = ExtractedEntity(
                    entity_id=canon_id,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_refs=sorted(list(set(ent.source_refs))),
                    aliases=aliases
                )
                resolved_entities.append(resolved_ent)
                
        return resolved_entities, id_map
